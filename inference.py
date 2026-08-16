"""
inference.py
============
Inference pipeline for Schrodinger Bridge cross-lingual morphing.
Takes a source audio file (zh-CN), extracts features, morphs them using
the trained SDE, and reconstructs the audio using a Griffin-Lim vocoder.

Usage:
    python inference.py \
        --input test_audio.mp3 \
        --checkpoint outputs/checkpoints/epoch_100.pt \
        --output morphed.wav
"""

import argparse
import yaml
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T

from models.encoder import build_encoder
from models.schrodinger_bridge import build_sb_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class GriffinLimVocoder(torch.nn.Module):
    """Converts DB Mel-Spectrograms back into waveforms."""
    def __init__(self, sample_rate: int = 16000, n_mels: int = 80,
                 n_fft: int = 1024, hop_length: int = 256,
                 win_length: int = 1024, f_min: float = 80.0,
                 f_max: float = 7600.0, max_iter: int = 32):
        super().__init__()
        self.inverse_mel = T.InverseMelScale(
            n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate,
            f_min=f_min, f_max=f_max, norm="slaney", mel_scale="slaney"
        )
        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft, n_iter=max_iter, win_length=win_length,
            hop_length=hop_length
        )

    def forward(self, mel_db: torch.Tensor) -> torch.Tensor:
        """
        mel_db: (B, n_mels, T_frames)
        Returns: (B, 1, T_samples)
        """
        # 1. Convert DB back to power (amplitude squared)
        # torchaudio AmplitudeToDB uses 10 * log10(x) for power
        mel_power = 10.0 ** (mel_db / 10.0)
        
        # 2. Inverse Mel Scale (Mel Power -> Linear Power)
        lin_power = self.inverse_mel(mel_power)
        
        # 3. Griffin-Lim to estimate phase and reconstruct waveform
        # Griffin-Lim expects linear amplitude, not power.
        lin_amp = torch.sqrt(lin_power + 1e-9)
        waveform = self.griffin_lim(lin_amp)
        return waveform.unsqueeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input audio file (.mp3, .wav)")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--output", default="morphed.wav", help="Output audio file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Loading checkpoint: {args.checkpoint}")
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        print("Please train the model first or provide a valid checkpoint path.")
        return

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt["config"]

    # ---- Models ----
    print("[inference] Building models ...")
    encoder  = build_encoder(cfg["encoder"]).to(device)
    drift, sde = build_sb_model(cfg)
    drift    = drift.to(device)

    encoder.load_state_dict(ckpt["encoder"])
    drift.load_state_dict(ckpt["drift"])
    encoder.eval(); drift.eval()

    # ---- Vocoder ----
    # Assumes we are using Mel features
    f_cfg = cfg.get("features", {})
    vocoder = GriffinLimVocoder(
        sample_rate=cfg["data"]["sample_rate"],
        n_mels=f_cfg.get("n_mels", 80),
        n_fft=f_cfg.get("n_fft", 1024),
        hop_length=f_cfg.get("hop_length", 256),
        win_length=f_cfg.get("win_length", 1024),
        f_min=f_cfg.get("f_min", 80.0),
        f_max=f_cfg.get("f_max", 7600.0)
    ).to(device)

    # ---- Process Audio ----
    print(f"[inference] Loading input: {args.input}")
    wf, sr = torchaudio.load(args.input)
    if sr != cfg["data"]["sample_rate"]:
        wf = torchaudio.transforms.Resample(sr, cfg["data"]["sample_rate"])(wf)
    if wf.shape[0] > 1:
        wf = wf.mean(0, keepdim=True)
    wf = wf.unsqueeze(0).to(device)  # (1, 1, T)

    print("[inference] Extracting features ...")
    with torch.no_grad():
        # Get frame-level features: (1, T_frames, 80)
        source_feats = encoder(wf, frame_level=True)

        print("[inference] Running Schrodinger Bridge Morphing SDE ...")
        # Generate the trajectory x_0 -> x_1
        morphed_feats, trajectory = sde.morph(source_feats, return_trajectory=True)
        
        # Transpose back to (1, 80, T_frames) for the vocoder
        morphed_mel = morphed_feats.transpose(1, 2)

        print("[inference] Reconstructing audio via Griffin-Lim vocoder ...")
        out_wf = vocoder(morphed_mel)

    # Save
    out_wf = out_wf.squeeze(0).cpu()
    torchaudio.save(args.output, out_wf, cfg["data"]["sample_rate"])
    print(f"\n✅ Success! Morphed audio saved to: {args.output}")

if __name__ == "__main__":
    main()
