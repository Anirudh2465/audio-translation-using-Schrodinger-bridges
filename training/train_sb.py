"""
training/train_sb.py
====================
Main training script for the Schrodinger Bridge cross-lingual morphing model.
Implements the architecture from Slides 6, 8, 16, and 24:
- Pretrained WavLM (d=768) latent manifold
- Entropic Optimal Transport (Sinkhorn coupling)
- Diffusion Schrodinger Bridge Matching (Shi et al., ICML 2024)
- Decoupled Acoustic Style Manifold Transport (preserving speech intelligibility)
- Neural Mel Decoder calibrated to SpeechT5 HiFi-GAN (zero static noise)

Usage:
    python -m training.train_sb --config configs/default.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

from data.loader import CommonVoiceDataset, CrossLingualIterator, make_loader
from models.encoder import build_encoder
from models.decoder import WavLMDecoder, MelSpectrogramExtractor
from models.ot_solver import minibatch_eot_loss, sample_transport_pairs
from models.schrodinger_bridge import build_sb_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_dirs(cfg: dict):
    for key in ["checkpoint_dir", "log_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    print(f"  [ckpt] Saved -> {path}", flush=True)


def train(cfg: dict):
    torch.manual_seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train] Device: {device}", flush=True)

    setup_dirs(cfg)
    writer = SummaryWriter(cfg["paths"]["log_dir"])

    # ---- Data ----
    print("[train] Loading datasets ...", flush=True)
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

    tr_cfg  = cfg["training"]
    zh_loader = make_loader(zh_ds, tr_cfg["batch_size"])
    bn_loader = make_loader(bn_ds, tr_cfg["batch_size"])
    pairs     = CrossLingualIterator(zh_loader, bn_loader)

    # ---- Models ----
    print("[train] Building models (WavLM d=768 + DriftNetwork + Calibrated MelDecoder) ...", flush=True)
    encoder   = build_encoder(cfg["encoder"]).to(device)
    drift, _  = build_sb_model(cfg)
    drift     = drift.to(device)

    decoder   = WavLMDecoder(in_dim=cfg["encoder"]["embedding_dim"],
                             out_dim=cfg.get("features", {}).get("n_mels", 80),
                             hidden_dim=512).to(device)
    mel_extractor = MelSpectrogramExtractor().to(device)

    params = list(drift.parameters()) + list(decoder.parameters())
    if not cfg["encoder"].get("freeze", True):
        params += list(encoder.parameters())

    optimizer = optim.AdamW(params, lr=tr_cfg["learning_rate"],
                            weight_decay=tr_cfg["weight_decay"])
    total_steps = tr_cfg["epochs"] * len(pairs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    ot_cfg   = cfg["ot"]
    global_step = 0

    print(f"[train] Starting training for {tr_cfg['epochs']} epochs ({len(pairs)} steps/epoch) ...\n", flush=True)

    for epoch in range(1, tr_cfg["epochs"] + 1):
        drift.train()
        decoder.train()
        encoder.eval() if cfg["encoder"].get("freeze", True) else encoder.train()

        epoch_loss = 0.0
        t0 = time.time()

        for step, (zh_batch, bn_batch) in enumerate(pairs):
            zh_wf, zh_len, _ = zh_batch
            bn_wf, bn_len, _ = bn_batch
            zh_wf = zh_wf.to(device); bn_wf = bn_wf.to(device)
            zh_len = zh_len.to(device); bn_len = bn_len.to(device)

            # 1. Extract WavLM representations and ground-truth calibrated Mel
            with torch.no_grad() if cfg["encoder"].get("freeze", True) else torch.enable_grad():
                zh_feats = encoder(zh_wf, zh_len, frame_level=True)   # (B, T_zh, 768)
                bn_feats = encoder(bn_wf, bn_len, frame_level=True)   # (B, T_bn, 768)
                zh_mel_gt = mel_extractor(zh_wf)                     # (B, 80, T_mel_zh)
                bn_mel_gt = mel_extractor(bn_wf)                     # (B, 80, T_mel_bn)

            # 2. Decoder Mel reconstruction loss (calibrated to SpeechT5 HiFi-GAN)
            zh_mel_pred = decoder(zh_feats, target_len=zh_mel_gt.shape[-1])
            bn_mel_pred = decoder(bn_feats, target_len=bn_mel_gt.shape[-1])
            dec_loss = 0.5 * (F.l1_loss(zh_mel_pred, zh_mel_gt) + F.l1_loss(bn_mel_pred, bn_mel_gt))

            # 3. Extract language acoustic style manifold vectors (Slide 16: z in R^768)
            s_zh = encoder(zh_wf, zh_len, frame_level=False)  # (B, 768)
            s_bn = encoder(bn_wf, bn_len, frame_level=False)  # (B, 768)

            # 4. Entropic Optimal Transport on speech manifold
            eot_loss, P = minibatch_eot_loss(
                s_zh, s_bn,
                reg=ot_cfg["reg"],
                reg_m=ot_cfg.get("reg_m"),
                n_iter=ot_cfg.get("n_iter", 100),
            )

            # 5. Sample discrete paired target style endpoints from Sinkhorn plan (Shi et al., ICML 2024)
            tgt_idx = sample_transport_pairs(P, deterministic=False)
            matched_s_bn = s_bn[tgt_idx]  # (B, 768)

            # 6. Diffusion Schrodinger Bridge Matching (DSBM) in R^768
            B = s_zh.shape[0]
            t = torch.rand(B, device=device)
            t_exp = t.view(-1, 1)

            # Stochastic bridge noise schedule: sigma(t) = sigma_b * sqrt(t * (1 - t))
            sigma_bridge = 0.05
            sigma_t = (sigma_bridge * torch.sqrt(t * (1.0 - t) + 1e-4)).view(-1, 1)
            noise = torch.randn_like(s_zh)

            # Interpolated state along the bridge trajectory
            s_t = (1.0 - t_exp) * s_zh + t_exp * matched_s_bn + sigma_t * noise
            target_drift = matched_s_bn - s_zh   # Geodesic velocity vector

            pred_drift = drift(s_t.detach(), t)
            sb_loss = F.mse_loss(pred_drift, target_drift)

            # Minimal kinetic energy regularization (Riemannian action integral)
            energy_loss = 0.5 * (pred_drift ** 2).mean()

            # Total objective
            loss = sb_loss + 0.1 * eot_loss + 1.0 * dec_loss + 0.01 * energy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tr_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % tr_cfg["log_every"] == 0:
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar("train/total_loss", loss.item(), global_step)
                writer.add_scalar("train/sb_loss",    sb_loss.item(), global_step)
                writer.add_scalar("train/dec_loss",   dec_loss.item(), global_step)
                writer.add_scalar("train/eot_loss",   eot_loss.item(), global_step)
                writer.add_scalar("train/energy_loss", energy_loss.item(), global_step)
                writer.add_scalar("train/lr",          lr, global_step)
                print(f"  Ep {epoch:03d} | step {global_step:06d} | "
                      f"loss {loss.item():.4f} | sb {sb_loss.item():.4f} | "
                      f"dec {dec_loss.item():.4f} | eot {eot_loss.item():.4f} | "
                      f"lr {lr:.2e}", flush=True)

            if step >= len(pairs) - 1:
                break

        epoch_time = time.time() - t0
        avg_loss   = epoch_loss / max(step + 1, 1)
        print(f"\nEpoch {epoch:03d} done | avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s\n", flush=True)

        if epoch % tr_cfg["save_every"] == 0 or epoch == tr_cfg["epochs"]:
            ckpt_path = f"{cfg['paths']['checkpoint_dir']}/epoch_{epoch:03d}.pt"
            save_checkpoint({
                "epoch":       epoch,
                "drift":       drift.state_dict(),
                "decoder":     decoder.state_dict(),
                "encoder":     encoder.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "config":      cfg,
                "global_step": global_step,
            }, ckpt_path)

    writer.close()
    print("[train] Training complete.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
