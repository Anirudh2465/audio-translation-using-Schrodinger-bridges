"""
models/ot_solver.py
===================
Entropic Optimal Transport (EOT) solver using the Sinkhorn algorithm.
This defines the coupling between the source (zh-CN) and target (bn)
speech embedding distributions -- the mathematical core of the project.

Requires: pip install POT
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def cost_matrix(x: torch.Tensor, y: torch.Tensor,
                p: int = 2) -> torch.Tensor:
    """
    Compute squared pairwise Euclidean cost matrix.

    Args:
        x: (n, d) source embeddings
        y: (m, d) target embeddings
        p: cost power (2 = squared Euclidean)
    Returns:
        C: (n, m) cost matrix
    """
    x_sq = (x ** 2).sum(-1, keepdim=True)    # (n, 1)
    y_sq = (y ** 2).sum(-1, keepdim=True).T  # (1, m)
    C    = x_sq + y_sq - 2 * x @ y.T
    return C.clamp(min=0) ** (p / 2)


def sinkhorn(C: torch.Tensor, reg: float = 0.05,
             n_iter: int = 100,
             a: Optional[torch.Tensor] = None,
             b: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Sinkhorn algorithm for balanced Entropic Optimal Transport.

    Solves: min_{P} <C, P> + reg * KL(P || ab^T)
            s.t.    P1 = a, P^T1 = b, P >= 0

    Args:
        C:      (n, m) cost matrix
        reg:    Entropic regularization epsilon (>0, smaller = closer to OT)
        n_iter: Number of Sinkhorn iterations
        a:      (n,) source marginal weights (uniform if None)
        b:      (m,) target marginal weights (uniform if None)
    Returns:
        P: (n, m) optimal transport plan
    """
    n, m = C.shape
    if a is None: a = torch.ones(n, device=C.device) / n
    if b is None: b = torch.ones(m, device=C.device) / m

    log_K = -C / reg        # (n, m) log kernel
    log_u = torch.zeros(n, device=C.device)  # log dual variable u
    log_v = torch.zeros(m, device=C.device)  # log dual variable v
    log_a = a.log()
    log_b = b.log()

    for _ in range(n_iter):
        log_u = log_a - torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u[:, None], dim=0)

    log_P = log_K + log_u[:, None] + log_v[None, :]
    return log_P.exp()


def sinkhorn_unbalanced(C: torch.Tensor, reg: float = 0.05,
                         reg_m: float = 1.0, n_iter: int = 100,
                         a: Optional[torch.Tensor] = None,
                         b: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Unbalanced Sinkhorn (KL marginal relaxation).
    Essential for our case: zh-CN has ~4.4x more validated hours than Bengali.

    Solves: min_{P} <C,P> + reg*KL(P||ab^T) + reg_m*(KL(P1||a) + KL(P^T1||b))

    Args:
        C:     (n, m) cost matrix
        reg:   Entropic regularization epsilon
        reg_m: Marginal relaxation weight (lower = more mass destruction allowed)
        n_iter: Iterations
        a, b:  Marginal distributions (uniform if None)
    Returns:
        P: (n, m) unbalanced transport plan
    """
    n, m = C.shape
    if a is None: a = torch.ones(n, device=C.device) / n
    if b is None: b = torch.ones(m, device=C.device) / m

    K    = (-C / reg).exp()   # (n, m)
    u    = torch.ones(n, device=C.device)
    v    = torch.ones(m, device=C.device)
    tau  = reg_m / (reg_m + reg)

    for _ in range(n_iter):
        u = (a / ((K @ v).clamp(min=1e-12))).pow(tau)
        v = (b / ((K.T @ u).clamp(min=1e-12))).pow(tau)

    P = u[:, None] * K * v[None, :]
    return torch.nan_to_num(P, nan=1.0 / (n * m))


def wasserstein_distance(C: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Compute W distance given cost matrix C and transport plan P."""
    return (C * P).sum()


def minibatch_eot_loss(x: torch.Tensor, y: torch.Tensor,
                        reg: float = 0.05, reg_m: Optional[float] = None,
                        n_iter: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mini-batch EOT loss between a batch of source (x) and target (y) embeddings.
    Used as the training objective for the Schrodinger Bridge drift network.

    Args:
        x:     (B, D) source embeddings (e.g., zh-CN features)
        y:     (B, D) target embeddings (e.g., Bengali features)
        reg:   Entropic regularisation epsilon
        reg_m: If set, use unbalanced OT with this marginal weight
        n_iter: Sinkhorn iterations
    Returns:
        Tuple of (Scalar EOT loss, Transport Plan P of shape (B, B))
    """
    C = cost_matrix(x, y)
    scale = C.max().clamp(min=1e-8)
    C_scaled = C / scale

    if reg_m is not None:
        P = sinkhorn_unbalanced(C_scaled, reg=reg, reg_m=reg_m, n_iter=n_iter)
    else:
        P = sinkhorn(C_scaled, reg=reg, n_iter=n_iter)
    return wasserstein_distance(C_scaled, P), P


def sample_transport_pairs(P: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
    """
    Sample target assignment indices j(i) for each source item i from the
    Entropic Optimal Transport plan P (shape B, B).
    Follows Diffusion Schrodinger Bridge Matching (Shi et al., ICML 2024),
    ensuring each source utterance is coupled with an actual, coherent target
    utterance rather than a blurred convex combination.

    Args:
        P: (B, B) transport plan matrix from Sinkhorn algorithm
        deterministic: If True, uses argmax (MAP / Monge assignment).
                       If False, samples from conditional categorical distribution.
    Returns:
        target_indices: (B,) indices of matched target samples
    """
    p_cond = P / P.sum(dim=1, keepdim=True).clamp(min=1e-12)
    if deterministic:
        return torch.argmax(p_cond, dim=1)
    else:
        return torch.multinomial(p_cond, num_samples=1).squeeze(-1)

