"""
models/schrodinger_bridge.py
============================
Schrodinger Bridge drift network for cross-lingual speech morphing.

The SB defines a stochastic process X_t, t in [0,1] that interpolates
from a source speech distribution (zh-CN) to a target (Bengali) while
minimizing entropic transport cost.

Architecture: Transformer-based drift network f_theta(x_t, t) that
predicts the score/drift at each time step.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class SinusoidalTimeEmbedding(nn.Module):
    """Encodes scalar time t in [0,1] as a sinusoidal vector."""
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) -> (B, dim)"""
        t   = t.unsqueeze(-1)                           # (B, 1)
        arg = t * self.freqs.unsqueeze(0)               # (B, half)
        return torch.cat([arg.sin(), arg.cos()], dim=-1) # (B, dim)


class DriftNetwork(nn.Module):
    """
    Transformer-based drift (score) network for the Schrodinger Bridge.

    Predicts the drift direction f_theta(z_t, t) in the latent embedding
    space at time t, given the noisy interpolated embedding z_t.

    Input:  z_t  -- (B, D) embedding at time t
            t    -- (B,) scalar time in [0, 1]
    Output: drift -- (B, D) predicted drift vector
    """
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 512,
                 n_heads: int = 8, n_layers: int = 6,
                 dropout: float = 0.1, time_dim: int = 128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.input_proj = nn.Linear(embedding_dim + time_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z_t: (B, D) embedding at time t  (or (B, T, D) for frame-level)
        t:   (B,) time in [0, 1]
        Returns: (B, D) drift
        """
        t_emb = self.time_embed(t)          # (B, time_dim)

        if z_t.dim() == 2:
            # Utterance-level: (B, D)
            x = torch.cat([z_t, t_emb], dim=-1)    # (B, D+time_dim)
            x = self.input_proj(x).unsqueeze(1)    # (B, 1, hidden)
            x = self.transformer(x)                # (B, 1, hidden)
            return self.output_proj(x.squeeze(1))  # (B, D)
        else:
            # Frame-level: (B, T, D)
            B, T, D = z_t.shape
            t_exp = t_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, time_dim)
            x = torch.cat([z_t, t_exp], dim=-1)            # (B, T, D+time_dim)
            x = self.input_proj(x)                         # (B, T, hidden)
            x = self.transformer(x)                        # (B, T, hidden)
            return self.output_proj(x)                     # (B, T, D)


class SchrodingerBridgeSDE:
    """
    Schrodinger Bridge SDE simulator.

    Forward process: dX_t = f(X_t, t) dt + sigma(t) dW_t
    where f is parameterized by DriftNetwork.

    Inference generates a morphing trajectory from x_0 (zh-CN) to x_1 (bn).
    """
    def __init__(self, drift_net: DriftNetwork,
                 sigma_min: float = 0.01, sigma_max: float = 1.0,
                 n_steps: int = 50):
        self.net      = drift_net
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.n_steps   = n_steps

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise schedule: linear interpolation sigma_min -> sigma_max."""
        return self.sigma_min + t * (self.sigma_max - self.sigma_min)

    def sde_step(self, x_t: torch.Tensor, t: torch.Tensor,
                 dt: float, sigma_scale: float = 0.0) -> torch.Tensor:
        """One Euler-Maruyama or ODE step."""
        drift = self.net(x_t, t)
        if sigma_scale <= 0.0:
            # Deterministic Probability Flow ODE (no white noise accumulation)
            return x_t + drift * dt

        sigma = self.sigma(t) * sigma_scale
        if x_t.dim() == 3:
            sigma = sigma.view(-1, 1, 1)
        else:
            sigma = sigma.view(-1, 1)
        noise = torch.randn_like(x_t) * sigma * math.sqrt(abs(dt))
        return x_t + drift * dt + noise

    @torch.no_grad()
    def morph(self, x_source: torch.Tensor,
              return_trajectory: bool = False,
              target_t: float = 1.0,
              sigma_scale: float = 0.0):
        """
        Generate a morphing trajectory from source to target distribution.

        Args:
            x_source: (B, D) or (B, T, D) source embeddings (zh-CN)
            return_trajectory: If True, return all intermediate states
            target_t: Target time in [0.0, 1.0] (e.g., 0.5 for intermediate, 1.0 for full target)
            sigma_scale: Noise scale (0.0 = deterministic ODE flow, >0 = stochastic SDE)

        Returns:
            x_t: final morphed embedding at target_t
            trajectory: List of states along trajectory (if return_trajectory=True)
        """
        x = x_source.clone()
        traj = [x.clone()] if return_trajectory else None
        
        n_steps = max(int(self.n_steps * target_t), 1)
        dt = target_t / n_steps

        for step in range(n_steps):
            t_val = torch.full((x.shape[0],), step * dt,
                               device=x.device, dtype=x.dtype)
            x = self.sde_step(x, t_val, dt, sigma_scale=sigma_scale)
            if return_trajectory:
                traj.append(x.clone())

        if return_trajectory:
            return x, traj
        return x


def build_sb_model(cfg: dict) -> tuple:
    """Build drift network and SDE from config dict."""
    sb_cfg  = cfg.get("schrodinger_bridge", {})
    enc_dim = cfg.get("encoder", {}).get("embedding_dim", 768)

    drift = DriftNetwork(
        embedding_dim = enc_dim,
        hidden_dim    = sb_cfg.get("hidden_dim", 512),
        n_heads       = sb_cfg.get("n_heads", 8),
        n_layers      = sb_cfg.get("n_layers", 6),
        dropout       = sb_cfg.get("dropout", 0.1),
    )
    sde = SchrodingerBridgeSDE(
        drift_net  = drift,
        sigma_min  = sb_cfg.get("sigma_min", 0.01),
        sigma_max  = sb_cfg.get("sigma_max", 1.0),
        n_steps    = sb_cfg.get("n_timesteps", 50),
    )
    return drift, sde
