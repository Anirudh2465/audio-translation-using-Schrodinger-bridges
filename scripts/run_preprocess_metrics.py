import sys
import os
import yaml
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchaudio

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.loader import CommonVoiceDataset

def compute_metrics(wf, sr):
    wf_np = wf.squeeze().numpy()
    if wf_np.size == 0 or np.all(wf_np == 0):
        return {"duration": 0.0, "rms": 0.0, "zcr": 0.0, "snr": 0.0}
        
    duration = len(wf_np) / sr
    rms = np.sqrt(np.mean(wf_np**2))
    zcr = np.mean(np.abs(np.diff(np.sign(wf_np)))) / 2
    
    # Simple SNR estimation: assuming lowest 10% of energy frames are noise
    frame_len = int(sr * 0.02) # 20ms frames
    frames = [wf_np[i:i+frame_len] for i in range(0, len(wf_np), frame_len)]
    if frames:
        rms_frames = [np.mean(f**2) for f in frames if len(f) == frame_len]
        if rms_frames:
            sorted_rms = np.sort(rms_frames)
            noise_idx = max(1, int(len(sorted_rms) * 0.10))
            noise_power = np.mean(sorted_rms[:noise_idx]) + 1e-10
            signal_power = np.mean(sorted_rms)
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 0.0
    else:
        snr = 0.0

    return {"duration": duration, "rms": rms, "zcr": zcr, "snr": snr}

def plot_waveform_mel(wf, sr, title, save_path, mel_transform=None):
    plt.figure(figsize=(12, 6))
    
    # Waveform
    plt.subplot(2, 1, 1)
    time_axis = np.linspace(0, wf.shape[-1] / sr, wf.shape[-1])
    plt.plot(time_axis, wf.squeeze().numpy(), color="teal", alpha=0.8)
    plt.title(f"Waveform - {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # Mel-Spectrogram
    plt.subplot(2, 1, 2)
    if mel_transform is None:
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80)
    mel_spec = mel_transform(wf)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    plt.imshow(mel_spec_db.squeeze().numpy(), aspect='auto', origin='lower', cmap='magma',
               extent=[0, wf.shape[-1] / sr, 0, 80])
    plt.title(f"Mel-Spectrogram (dB) - {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel Bins")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    os.makedirs("plots", exist_ok=True)
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    data_cfg = cfg["data"]
    
    print("[metrics] Loading datasets ...")
    zh_ds = CommonVoiceDataset(
        tar_path=data_cfg["zh_tar"], locale="zh-CN", split="train",
        sample_rate=data_cfg["sample_rate"], cache_dir=data_cfg.get("cache_dir"),
        max_samples=500
    ).quality_filter(min_up=data_cfg["min_up_votes"], max_down=data_cfg["max_down_votes"])
    
    bn_ds = CommonVoiceDataset(
        tar_path=data_cfg["bn_tar"], locale="bn", split="train",
        sample_rate=data_cfg["sample_rate"], cache_dir=data_cfg.get("cache_dir"),
        max_samples=500
    ).quality_filter(min_up=data_cfg["min_up_votes"], max_down=data_cfg["max_down_votes"])
    
    n_samples = min(len(zh_ds), len(bn_ds))
    # Ensure they have the exact same number of samples for parallel processing
    zh_ds.df = zh_ds.df.head(n_samples)
    bn_ds.df = bn_ds.df.head(n_samples)
    
    datasets = {"zh-CN": zh_ds, "bn": bn_ds}
    all_metrics = []
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=data_cfg["sample_rate"], n_fft=1024, hop_length=256, n_mels=80)
    
    print(f"[metrics] Computing metrics on {n_samples} samples per locale...")
    
    for loc, ds in datasets.items():
        for i in range(len(ds)):
            wf, sr, meta = ds[i]
            metrics = compute_metrics(wf, sr)
            metrics["locale"] = loc
            all_metrics.append(metrics)
            
            # Plot one example per locale
            if i == 0:
                plot_waveform_mel(wf, sr, f"{loc} Sample", f"plots/sample_{loc}.png", mel_transform)
                
            if (i+1) % 100 == 0:
                print(f"[{loc}] Processed {i+1}/{n_samples} clips")

    df = pd.DataFrame(all_metrics)
    
    print("[metrics] Generating distribution plots ...")
    sns.set_theme(style="whitegrid")
    
    metrics_to_plot = ["duration", "rms", "zcr", "snr"]
    titles = ["Duration (s)", "RMS Energy", "Zero-Crossing Rate", "Estimated SNR (dB)"]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        sns.violinplot(data=df, x="locale", y=metric, ax=ax, palette="muted", inner="quartile")
        ax.set_title(f"Distribution of {title}")
        ax.set_ylabel(title)
        
    plt.tight_layout()
    plt.savefig("plots/metrics_distributions.png", dpi=150)
    plt.close()
    
    print("[metrics] Done! Plots saved to codes/plots/")

if __name__ == "__main__":
    main()
