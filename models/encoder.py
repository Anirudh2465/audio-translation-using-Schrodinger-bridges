"""
models/encoder.py
=================
Speech encoder wrapper -- extracts utterance/frame-level representations.
Supports mel-spectrogram baseline and WavLM (pretrained transformer).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchaudio.transforms as T


class MelEncoder(nn.Module):
    """
    Simple CNN encoder over log mel-spectrograms.
    Produces a fixed-dim embedding per utterance.
    Output: (B, embedding_dim)
    """
    def __init__(self, n_mels: int = 80, embedding_dim: int = 256,
                 n_fft: int = 1024, hop_length: int = 256, sample_rate: int = 16000):
        super().__init__()
        self.mel = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                                     hop_length=hop_length, n_mels=n_mels)
        self.to_db = T.AmplitudeToDB(top_db=80)
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, 256, 5, padding=2), nn.GELU(), nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, 5, padding=2),    nn.GELU(), nn.BatchNorm1d(256),
            nn.Conv1d(256, embedding_dim, 5, padding=2), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, waveform: torch.Tensor, lengths: Optional[torch.Tensor] = None, frame_level: bool = False) -> torch.Tensor:
        """
        waveform: (B, 1, T)
        frame_level=False -> embedding: (B, embedding_dim)
        frame_level=True  -> mel spec: (B, T_frames, n_mels)
        """
        mel = self.to_db(self.mel(waveform.squeeze(1)))  # (B, n_mels, T_frames)
        if frame_level:
            return mel.transpose(1, 2)                    # (B, T_frames, n_mels)
            
        h   = self.cnn(mel)                               # (B, emb, T_frames)
        return self.pool(h).squeeze(-1)                   # (B, emb)


class WavLMEncoder(nn.Module):
    """
    WavLM frame-level encoder with mean-pooling for utterance embeddings.
    Requires: pip install transformers

    Output: (B, hidden_dim) or (B, T_frames, hidden_dim) with frame_level=True
    """
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus",
                 freeze: bool = True, layer: Optional[int] = None):
        super().__init__()
        try:
            from transformers import WavLMModel, AutoFeatureExtractor
        except ImportError:
            raise ImportError("pip install transformers")

        self.feat_ext = AutoFeatureExtractor.from_pretrained(model_name)
        self.model    = WavLMModel.from_pretrained(model_name)
        self.layer    = layer

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, waveform: torch.Tensor,
                lengths: Optional[torch.Tensor] = None,
                frame_level: bool = False) -> torch.Tensor:
        """
        waveform: (B, 1, T) at 16 kHz
        Returns:
            frame_level=False -> (B, 768) utterance embedding
            frame_level=True  -> (B, T_frames, 768)
        """
        wav_np = [waveform[i,0].cpu().numpy() for i in range(len(waveform))]
        inputs = self.feat_ext(wav_np, sampling_rate=16000,
                               return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad() if not self.training else torch.enable_grad():
            outs = self.model(**inputs,
                              output_hidden_states=(self.layer is not None))

        feats = outs.hidden_states[self.layer] if self.layer is not None \
                else outs.last_hidden_state     # (B, T_frames, D)

        if frame_level:
            return feats

        # Mean pool with length mask
        if lengths is not None:
            mask = torch.zeros(feats.shape[:2], device=device)
            for i, l in enumerate(lengths):
                # Convert sample lengths to frame lengths
                frame_len = min(int(l.item() / 320), feats.shape[1])
                mask[i, :frame_len] = 1.0
            mask = mask.unsqueeze(-1)
            return (feats * mask).sum(1) / mask.sum(1).clamp(min=1)
        return feats.mean(1)


def build_encoder(cfg: dict) -> nn.Module:
    """Factory function -- builds encoder from config dict."""
    enc_type = cfg.get("type", "mel")
    if enc_type == "mel":
        return MelEncoder(
            n_mels=cfg.get("n_mels", 80),
            embedding_dim=cfg.get("embedding_dim", 256),
        )
    elif enc_type == "wavlm":
        return WavLMEncoder(
            model_name=cfg.get("pretrained", "microsoft/wavlm-base-plus"),
            freeze=cfg.get("freeze", True),
            layer=cfg.get("layer", None),
        )
    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")
