"""
data/preprocessing.py
=====================
Audio feature extraction utilities for the EOT morphing pipeline.
Converts raw waveforms -> mel-spectrograms or WavLM embeddings.
"""

from typing import Optional, Tuple
import torch
import torchaudio
import torchaudio.transforms as T


class MelExtractor(torch.nn.Module):
    """
    Converts waveform -> log mel-spectrogram.
    Output shape: (B, n_mels, T_frames)
    """
    def __init__(self, sample_rate: int = 16000, n_mels: int = 80,
                 n_fft: int = 1024, hop_length: int = 256,
                 win_length: int = 1024, f_min: float = 80.0,
                 f_max: float = 7600.0, norm: str = "slaney"):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, n_mels=n_mels, f_min=f_min, f_max=f_max,
            norm=norm, mel_scale="slaney",
        )
        self.to_db = T.AmplitudeToDB(top_db=80)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, 1, T) -> log_mel: (B, n_mels, T_frames)"""
        mel = self.mel(waveform.squeeze(1))  # (B, n_mels, T_frames)
        return self.to_db(mel)

    def normalize(self, mel: torch.Tensor) -> torch.Tensor:
        """Normalize mel to zero mean, unit variance (per utterance)."""
        mean = mel.mean(dim=-1, keepdim=True)
        std  = mel.std(dim=-1, keepdim=True).clamp(min=1e-5)
        return (mel - mean) / std


class WavLMExtractor(torch.nn.Module):
    """
    Extracts frame-level embeddings from a pretrained WavLM model.
    Requires: pip install transformers

    Output shape: (B, T_frames, hidden_dim)  -- hidden_dim=768 for base
    """
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus",
                 freeze: bool = True, layer: Optional[int] = None):
        super().__init__()
        try:
            from transformers import WavLMModel, AutoFeatureExtractor
        except ImportError:
            raise ImportError("Run: pip install transformers")

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name)
        self.layer = layer  # None = last hidden state; int = specific layer

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        waveform: (B, 1, T) torch tensor at 16 kHz
        Returns:  (B, T_frames, 768) frame embeddings
        """
        # WavLM expects numpy or list of 1-D arrays
        wav_list = [waveform[i, 0].cpu().numpy() for i in range(waveform.shape[0])]
        inputs   = self.feature_extractor(wav_list, sampling_rate=sample_rate,
                                           return_tensors="pt", padding=True)
        inputs   = {k: v.to(next(self.model.parameters()).device)
                    for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=(self.layer is not None))

        if self.layer is not None:
            return outputs.hidden_states[self.layer]   # (B, T, 768)
        return outputs.last_hidden_state               # (B, T, 768)


def extract_speaker_embedding(wavlm_feats: torch.Tensor,
                              lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mean-pool WavLM frame embeddings over time -> speaker embedding.

    Args:
        wavlm_feats: (B, T, D) frame embeddings
        lengths:     (B,) valid frame counts (None = use all)
    Returns:
        (B, D) utterance-level embeddings
    """
    if lengths is None:
        return wavlm_feats.mean(dim=1)

    mask = torch.zeros(wavlm_feats.shape[:2], device=wavlm_feats.device)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0
    mask = mask.unsqueeze(-1)
    return (wavlm_feats * mask).sum(1) / mask.sum(1).clamp(min=1)
