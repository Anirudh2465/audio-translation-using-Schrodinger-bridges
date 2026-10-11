"""
models/decoder.py
=================
Neural vocoder and decoder components for cross-lingual speech morphing.
Maps WavLM latent representations (d=768) to log-Mel spectrograms (80 bins),
calibrated exactly to SpeechT5HifiGan specifications (log10, amplitude power, 80-7600Hz),
and synthesizes 16kHz audio waveforms with zero static.

Aligns with Slides 6, 8, 16, and 23 of the project presentation.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MelSpectrogramExtractor(nn.Module):
    """
    Extracts 80-channel log10-Mel spectrograms on GPU, exactly calibrated
    to SpeechT5HifiGan specifications:
    - Sample rate: 16 kHz
    - FFT: 1024, Win: 1024, Hop: 256
    - Power: 1.0 (amplitude)
    - Filters: SpeechT5 80-channel filterbank (80 Hz to 7600 Hz)
    - Mel floor: 1e-10
    - Log scale: log10
    """
    def __init__(self):
        super().__init__()
        from transformers import SpeechT5FeatureExtractor
        feat_ext = SpeechT5FeatureExtractor.from_pretrained("microsoft/speecht5_tts")
        # mel_filters shape: (513, 80)
        self.register_buffer("filters", torch.from_numpy(feat_ext.mel_filters).float())
        self.register_buffer("window", torch.hann_window(1024))
        self.mel_floor = float(feat_ext.mel_floor)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (B, T) or (B, 1, T) at 16kHz on GPU
        Returns: (B, 80, T_mel) log10 Mel spectrogram calibrated to SpeechT5HiFiGAN
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        stft = torch.stft(
            waveform, n_fft=1024, hop_length=256, win_length=1024,
            window=self.window, return_complex=True, center=True, pad_mode="reflect"
        )  # (B, 513, T_frames)
        mag = torch.abs(stft)  # amplitude spectrum (power=1.0)
        mel = torch.einsum("fm,bft->bmt", self.filters, mag)  # (B, 80, T_frames)
        log_mel = torch.log10(torch.clamp(mel, min=self.mel_floor))
        return log_mel


class WavLMDecoder(nn.Module):
    """
    Convolutional decoder that maps continuous latent WavLM representations (d=768, 50 fps)
    to 80-channel log-Mel spectrograms (80 bins, 62.5 fps) suitable for HiFi-GAN.

    Input:  z -- (B, T_wavlm, 768)
    Output: mel -- (B, 80, T_mel)
    """
    def __init__(self, in_dim: int = 768, out_dim: int = 80, hidden_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.act1  = nn.GELU()

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.act2  = nn.GELU()

        self.conv3 = nn.Conv1d(hidden_dim, 256, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm1d(256)
        self.act3  = nn.GELU()

        self.out_proj = nn.Conv1d(256, out_dim, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        """
        z: (B, T_wavlm, in_dim)
        target_len: Optional explicit Mel frame length.
                    If None, resamples by 1.25x (stride 320 -> stride 256).
        Returns: (B, out_dim, T_mel)
        """
        # Transpose to (B, in_dim, T_wavlm)
        h = z.transpose(1, 2)
        if target_len is not None:
            h = F.interpolate(h, size=target_len, mode="linear", align_corners=False)
        else:
            h = F.interpolate(h, scale_factor=1.25, mode="linear", align_corners=False)

        h = self.act1(self.norm1(self.conv1(h)))
        h = self.act2(self.norm2(self.conv2(h)))
        h = self.act3(self.norm3(self.conv3(h)))
        mel = self.out_proj(h)  # (B, out_dim, T_mel)
        return mel


class SpeechT5Vocoder(nn.Module):
    """
    Neural Vocoder wrapper around pretrained SpeechT5HifiGan.
    Synthesizes crisp 16kHz audio directly from 80-channel log-Mel spectrograms.
    """
    def __init__(self, model_name: str = "microsoft/speecht5_hifigan", device: Optional[str] = None):
        super().__init__()
        from transformers import SpeechT5HifiGan
        self.hifigan = SpeechT5HifiGan.from_pretrained(model_name)
        if device is not None:
            self.hifigan = self.hifigan.to(device)
        self.hifigan.eval()
        for p in self.hifigan.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: (B, 80, T_mel) or (80, T_mel) or (T_mel, 80)
        Returns: waveform (B, T_audio) or (T_audio,)
        """
        if mel.dim() == 2:
            if mel.shape[0] == 80:
                mel = mel.transpose(0, 1)  # (T, 80)
            return self.hifigan(mel)
        elif mel.dim() == 3:
            # (B, 80, T) -> (B, T, 80)
            mel = mel.transpose(1, 2)
            return self.hifigan(mel)
        else:
            raise ValueError(f"Unexpected mel tensor dimensions: {mel.shape}")
