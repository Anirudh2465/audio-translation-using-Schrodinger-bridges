"""
inference.py
============
Inference pipeline for Schrodinger Bridge cross-lingual morphing.
Takes a source audio file (zh-CN), extracts WavLM latent features (d=768),
morphs the acoustic style manifold vector along the continuous Schrodinger Bridge
trajectory to target (Bengali) while preserving phonetic intelligibility,
decodes to calibrated log-Mel spectrogram, and synthesizes crisp audio using HiFi-GAN.

Usage:
    python inference.py \
        --input sample_input_zh.wav \
        --checkpoint outputs/checkpoints/epoch_005.pt \
        --output morphed.wav \
        --target_t 1.0 \
        --save_intermediates
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torchaudio
import torchaudio.transforms as T

from models.encoder import build_encoder
from models.decoder import WavLMDecoder, SpeechT5Vocoder
from models.schrodinger_bridge import build_sb_model


def main():
    parser = argparse.ArgumentParser(description="Schrodinger Bridge Audio Morphing Inference")
    parser.add_argument("--input", required=True, help="Input audio file (.mp3, .wav)")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--output", default="morphed.wav", help="Output audio file path")
    parser.add_argument("--target_t", type=float, default=1.0, help="Morphing target time in [0, 1] (default: 1.0)")
    parser.add_argument("--sigma_scale", type=float, default=0.0, help="SDE noise scale (0.0 = deterministic ODE flow)")
    parser.add_argument("--save_intermediates", action="store_true", help="Save intermediate steps at t=0.0, 0.5, 1.0")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Loading checkpoint: {args.checkpoint}", flush=True)

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt["config"]

    # ---- Build Models ----
    print("[inference] Initializing WavLM encoder, DriftNetwork, MelDecoder, and HiFi-GAN ...", flush=True)
    encoder  = build_encoder(cfg["encoder"]).to(device)
    drift, sde = build_sb_model(cfg)
    drift    = drift.to(device)

    decoder = WavLMDecoder(
        in_dim=cfg["encoder"].get("embedding_dim", 768),
        out_dim=cfg.get("features", {}).get("n_mels", 80),
        hidden_dim=512,
    ).to(device)

    vocoder = SpeechT5Vocoder(device=device)

    # Load weights
    encoder.load_state_dict(ckpt["encoder"])
    drift.load_state_dict(ckpt["drift"])
    if "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"])
    else:
        print("[inference] Warning: 'decoder' weights not found in checkpoint. Using initialized decoder.")

    encoder.eval()
    drift.eval()
    decoder.eval()

    # ---- Load Input Audio ----
    print(f"[inference] Loading audio: {args.input}", flush=True)
    wf, sr = torchaudio.load(args.input)
    target_sr = cfg["data"]["sample_rate"]
    if sr != target_sr:
        wf = T.Resample(sr, target_sr)(wf)
    if wf.shape[0] > 1:
        wf = wf.mean(0, keepdim=True)
    wf = wf.to(device)  # (1, T)

    # ---- Feature Extraction & Style-Content Decomposition ----
    print("[inference] Extracting WavLM latent manifold representation ...", flush=True)
    with torch.no_grad():
        source_feats = encoder(wf.unsqueeze(0) if wf.dim() == 2 else wf, frame_level=True)
        # Separate time-varying phonetic trajectory from global acoustic style vector
        s_0 = source_feats.mean(dim=1)               # (1, 768) style manifold coordinate
        c_0 = source_feats - s_0.unsqueeze(1)        # (1, T, 768) phonetic content trajectory
        print(f"  Source content shape: {c_0.shape}, style shape: {s_0.shape}", flush=True)

        # Helper synthesis function
        def synthesize_from_style(s_target: torch.Tensor, out_path: str):
            z_conditioned = c_0 + s_target.unsqueeze(1) # (1, T, 768)
            mel = decoder(z_conditioned)                # (1, 80, T_mel)
            synth_audio = vocoder(mel)                  # (1, T_audio)
            synth_audio = synth_audio.squeeze(0).cpu()
            if synth_audio.dim() == 1:
                synth_audio = synth_audio.unsqueeze(0)
            torchaudio.save(out_path, synth_audio, target_sr)
            rms = synth_audio.pow(2).mean().sqrt().item()
            print(f"  [synthesized] -> {out_path} (RMS: {rms:.4f})", flush=True)

        if args.save_intermediates:
            stem = Path(args.output).stem
            ext = Path(args.output).suffix or ".wav"
            out_dir = Path(args.output).parent

            # t = 0.0 (Native source speech reconstructed on manifold)
            synthesize_from_style(s_0, str(out_dir / f"{stem}_t0.0{ext}"))

            # t = 0.5 (Midpoint geodesic cross-lingual morph)
            s_mid = sde.morph(s_0, target_t=0.5, sigma_scale=args.sigma_scale)
            synthesize_from_style(s_mid, str(out_dir / f"{stem}_t0.5{ext}"))

            # t = 1.0 (Full target Bengali acoustic manifold)
            s_end = sde.morph(s_0, target_t=1.0, sigma_scale=args.sigma_scale)
            synthesize_from_style(s_end, str(out_dir / f"{stem}_t1.0{ext}"))

        # Main output at target_t
        print(f"[inference] Integrating Schrodinger Bridge to t={args.target_t} (sigma_scale={args.sigma_scale}) ...", flush=True)
        s_morphed = sde.morph(s_0, target_t=args.target_t, sigma_scale=args.sigma_scale)
        synthesize_from_style(s_morphed, args.output)

    print(f"\n[inference] Complete! Output saved to: {args.output}\n", flush=True)


if __name__ == "__main__":
    main()
