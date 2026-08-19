"""
training/train_sb.py
====================
Main training script for the Schrodinger Bridge cross-lingual morphing model.

Usage:
    python -m training.train_sb --config configs/default.yaml
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

from data.loader import CommonVoiceDataset, CrossLingualIterator, make_loader
from models.encoder import build_encoder
from models.ot_solver import minibatch_eot_loss
from models.schrodinger_bridge import build_sb_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_dirs(cfg: dict):
    for key in ["checkpoint_dir", "log_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    print(f"  [ckpt] Saved -> {path}")


def train(cfg: dict):
    torch.manual_seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train] Device: {device}")

    setup_dirs(cfg)
    writer = SummaryWriter(cfg["paths"]["log_dir"])

    # ---- Data ----
    print("[train] Loading datasets ...")
    data_cfg = cfg["data"]
    zh_ds = CommonVoiceDataset(
        tar_path=data_cfg["zh_tar"], locale="zh-CN", split="train",
        sample_rate=data_cfg["sample_rate"],
        cache_dir=data_cfg.get("cache_dir"),
    ).quality_filter(min_up=data_cfg["min_up_votes"],
                     max_down=data_cfg["max_down_votes"])

    bn_ds = CommonVoiceDataset(
        tar_path=data_cfg["bn_tar"], locale="bn", split="train",
        sample_rate=data_cfg["sample_rate"],
        cache_dir=data_cfg.get("cache_dir"),
    ).quality_filter(min_up=data_cfg["min_up_votes"],
                     max_down=data_cfg["max_down_votes"])

    # Subset for testing end-to-end training
    zh_ds.df = zh_ds.df.head(32)
    bn_ds.df = bn_ds.df.head(32)

    tr_cfg  = cfg["training"]
    tr_cfg["epochs"] = 1
    zh_loader = make_loader(zh_ds, tr_cfg["batch_size"])
    bn_loader = make_loader(bn_ds, tr_cfg["batch_size"])
    pairs     = CrossLingualIterator(zh_loader, bn_loader)

    # ---- Models ----
    print("[train] Building models ...")
    encoder   = build_encoder(cfg["encoder"]).to(device)
    drift, _  = build_sb_model(cfg)
    drift     = drift.to(device)

    params    = list(drift.parameters())
    if not cfg["encoder"].get("freeze", True):
        params += list(encoder.parameters())

    optimizer = optim.AdamW(params, lr=tr_cfg["learning_rate"],
                            weight_decay=tr_cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tr_cfg["epochs"] * len(pairs))

    ot_cfg   = cfg["ot"]
    global_step = 0

    print(f"[train] Starting training for {tr_cfg['epochs']} epochs ...\n")

    for epoch in range(1, tr_cfg["epochs"] + 1):
        drift.train(); encoder.eval() if cfg["encoder"].get("freeze") else encoder.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (zh_batch, bn_batch) in enumerate(pairs):
            zh_wf, zh_len, _ = zh_batch
            bn_wf, bn_len, _ = bn_batch
            zh_wf = zh_wf.to(device); bn_wf = bn_wf.to(device)
            zh_len = zh_len.to(device); bn_len = bn_len.to(device)

            # Extract embeddings
            with torch.no_grad() if cfg["encoder"].get("freeze") else torch.enable_grad():
                zh_emb = encoder(zh_wf, zh_len) if hasattr(encoder, "feat_ext") \
                         else encoder(zh_wf)
                bn_emb = encoder(bn_wf, bn_len) if hasattr(encoder, "feat_ext") \
                         else encoder(bn_wf)

            # EOT loss between distributions
            eot_loss, P = minibatch_eot_loss(
                zh_emb, bn_emb,
                reg=ot_cfg["reg"],
                reg_m=ot_cfg.get("reg_m"),
                n_iter=ot_cfg.get("n_iter", 100),
            )

            # Map zh_emb to their mathematically expected bn_emb target using P
            # P has shape (B, B). P @ bn_emb computes weighted target for each source.
            # We scale by B to normalize the expected target per source sample.
            expected_bn_emb = (P @ bn_emb) * zh_emb.shape[0]

            # SB drift loss (IPF / DSM objective)
            t  = torch.rand(zh_emb.shape[0], device=device)
            # Interpolate: z_t = (1-t)*x_0 + t*x_1 + sigma(t)*noise
            noise = torch.randn_like(zh_emb)
            sigma_t = 0.01 + t * 0.99
            z_t   = (1 - t.unsqueeze(-1)) * zh_emb + t.unsqueeze(-1) * expected_bn_emb \
                    + sigma_t.unsqueeze(-1) * noise
            target = expected_bn_emb - zh_emb   # ideal drift direction

            predicted = drift(z_t.detach(), t)
            sb_loss   = ((predicted - target) ** 2).mean()

            loss = eot_loss + sb_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tr_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % tr_cfg["log_every"] == 0:
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar("train/loss",     loss.item(),     global_step)
                writer.add_scalar("train/eot_loss", eot_loss.item(), global_step)
                writer.add_scalar("train/sb_loss",  sb_loss.item(),  global_step)
                writer.add_scalar("train/lr",       lr,              global_step)
                print(f"  Ep {epoch:03d} | step {global_step:06d} | "
                      f"loss {loss.item():.4f} | eot {eot_loss.item():.4f} | "
                      f"sb {sb_loss.item():.4f} | lr {lr:.2e}")

            if step >= len(pairs) - 1:
                break

        epoch_time = time.time() - t0
        avg_loss   = epoch_loss / max(step + 1, 1)
        print(f"\nEpoch {epoch:03d} done | avg_loss={avg_loss:.4f} | {epoch_time:.1f}s\n")

        if epoch % tr_cfg["save_every"] == 0:
            ckpt_path = f"{cfg['paths']['checkpoint_dir']}/epoch_{epoch:03d}.pt"
            save_checkpoint({
                "epoch":       epoch,
                "drift":       drift.state_dict(),
                "encoder":     encoder.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "config":      cfg,
                "global_step": global_step,
            }, ckpt_path)

    writer.close()
    print("[train] Training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
