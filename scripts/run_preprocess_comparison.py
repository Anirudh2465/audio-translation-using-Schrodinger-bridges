"""
scripts/run_preprocess_comparison.py
====================================
Generates a before/after comparison of the preprocessing pipeline.
Produces plots and a Markdown report.
"""
import sys
import os
import yaml
import numpy as np
import torch
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.loader import (
    CommonVoiceDataset, preprocess, _denoise, _trim_silence, _peak_normalize,
)

N_EXAMPLES = 5  # number of before/after examples per locale


def compute_snr(wf_np, sr):
    """Estimate SNR in dB using lowest-10%-energy-frames as noise proxy."""
    frame_len = int(sr * 0.02)
    frames = [wf_np[i:i+frame_len] for i in range(0, len(wf_np), frame_len)]
    rms_frames = [np.mean(f**2) for f in frames if len(f) == frame_len]
    if not rms_frames:
        return 0.0
    sorted_rms = np.sort(rms_frames)
    noise_idx = max(1, int(len(sorted_rms) * 0.10))
    noise_power = np.mean(sorted_rms[:noise_idx]) + 1e-10
    signal_power = np.mean(sorted_rms)
    return 10 * np.log10(signal_power / noise_power)


def compute_metrics(wf_np, sr):
    """Compute duration, RMS, ZCR, SNR."""
    if wf_np.size == 0 or np.all(wf_np == 0):
        return {"duration": 0.0, "rms": 0.0, "zcr": 0.0, "snr_db": 0.0}
    duration = len(wf_np) / sr
    rms = float(np.sqrt(np.mean(wf_np**2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(wf_np)))) / 2)
    snr = compute_snr(wf_np, sr)
    return {"duration": round(duration, 3), "rms": round(rms, 5),
            "zcr": round(zcr, 5), "snr_db": round(snr, 2)}


def plot_comparison(raw_wf, clean_wf, sr, title, save_path):
    """4-panel plot: raw waveform, raw mel, clean waveform, clean mel."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80)
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Raw waveform
    t_raw = np.linspace(0, raw_wf.shape[-1] / sr, raw_wf.shape[-1])
    axes[0, 0].plot(t_raw, raw_wf.squeeze().numpy(), color="#e74c3c", alpha=0.8)
    axes[0, 0].set_title("Raw Waveform")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")

    # Raw mel
    raw_mel = amp_to_db(mel_transform(raw_wf)).squeeze().numpy()
    axes[0, 1].imshow(raw_mel, aspect="auto", origin="lower", cmap="magma",
                      extent=[0, raw_wf.shape[-1] / sr, 0, 80])
    axes[0, 1].set_title("Raw Mel-Spectrogram (dB)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Mel Bins")

    # Clean waveform
    t_clean = np.linspace(0, clean_wf.shape[-1] / sr, clean_wf.shape[-1])
    axes[1, 0].plot(t_clean, clean_wf.squeeze().numpy(), color="#2ecc71", alpha=0.8)
    axes[1, 0].set_title("Preprocessed Waveform (Denoised + Trimmed + Normalized)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Amplitude")

    # Clean mel
    clean_mel = amp_to_db(mel_transform(clean_wf)).squeeze().numpy()
    axes[1, 1].imshow(clean_mel, aspect="auto", origin="lower", cmap="magma",
                      extent=[0, clean_wf.shape[-1] / sr, 0, 80])
    axes[1, 1].set_title("Preprocessed Mel-Spectrogram (dB)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Mel Bins")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    os.makedirs("plots", exist_ok=True)
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    sr = data_cfg["sample_rate"]

    print("[comparison] Loading datasets ...")
    zh_ds = CommonVoiceDataset(
        tar_path=data_cfg["zh_tar"], locale="zh-CN", split="train",
        sample_rate=sr, cache_dir=data_cfg.get("cache_dir"),
        max_samples=50,
    ).quality_filter(min_up=data_cfg["min_up_votes"],
                     max_down=data_cfg["max_down_votes"])

    bn_ds = CommonVoiceDataset(
        tar_path=data_cfg["bn_tar"], locale="bn", split="train",
        sample_rate=sr, cache_dir=data_cfg.get("cache_dir"),
        max_samples=50,
    ).quality_filter(min_up=data_cfg["min_up_votes"],
                     max_down=data_cfg["max_down_votes"])

    datasets = {"zh-CN": zh_ds, "bn": bn_ds}

    # ---- Collect before/after metrics for all samples ----
    md_lines = [
        "# Audio Preprocessing & Denoising Report\n",
        "## Pipeline Steps\n",
        "1. **Spectral-Gating Denoising** — removes stationary background noise "
        "(hum, hiss, fan) using the `noisereduce` library with `prop_decrease=0.75`.\n",
        "2. **Silence Trimming** — trims leading and trailing silence below -40 dB "
        "using frame-level energy thresholding (20 ms frames).\n",
        "3. **Peak Normalization** — scales the waveform so `max(|amplitude|) = 1.0` "
        "for consistent volume across all speakers.\n",
        "---\n",
    ]

    for loc, ds in datasets.items():
        print(f"\n[comparison] Processing {loc} ...")
        md_lines.append(f"## Locale: `{loc}`\n")

        before_snrs, after_snrs = [], []
        before_durs, after_durs = [], []
        n_examples = min(N_EXAMPLES, len(ds))
        n_total = min(len(ds), 30)  # compute aggregate metrics on up to 30

        for i in range(n_total):
            try:
                # Load raw (bypass preprocessing in __getitem__)
                row = ds.df.iloc[i]
                raw_wf, raw_sr = ds._load(row["archive_path"])
                if raw_sr != sr:
                    raw_wf = torchaudio.transforms.Resample(raw_sr, sr)(raw_wf)
                if raw_wf.shape[0] > 1:
                    raw_wf = raw_wf.mean(0, keepdim=True)

                clean_wf = preprocess(raw_wf.clone(), sr)

                raw_np = raw_wf.squeeze().numpy()
                clean_np = clean_wf.squeeze().numpy()

                before_m = compute_metrics(raw_np, sr)
                after_m = compute_metrics(clean_np, sr)

                before_snrs.append(before_m["snr_db"])
                after_snrs.append(after_m["snr_db"])
                before_durs.append(before_m["duration"])
                after_durs.append(after_m["duration"])

                # Generate comparison plots for first N examples
                if i < n_examples:
                    plot_path = f"plots/comparison_{loc}_{i}.png"
                    plot_comparison(raw_wf, clean_wf, sr,
                                   f"{loc} Sample {i+1}: Before vs After", plot_path)
                    md_lines.append(f"### Sample {i+1}\n")
                    md_lines.append(f"| Metric | Before | After |\n")
                    md_lines.append(f"|--------|--------|-------|\n")
                    md_lines.append(f"| Duration (s) | {before_m['duration']} | {after_m['duration']} |\n")
                    md_lines.append(f"| RMS Energy | {before_m['rms']} | {after_m['rms']} |\n")
                    md_lines.append(f"| ZCR | {before_m['zcr']} | {after_m['zcr']} |\n")
                    md_lines.append(f"| SNR (dB) | {before_m['snr_db']:.2f} | {after_m['snr_db']:.2f} |\n")
                    md_lines.append(f"\n![{loc} Sample {i+1}]({os.path.abspath(plot_path)})\n\n")

                if (i + 1) % 10 == 0:
                    print(f"  [{loc}] {i+1}/{n_total} processed")
            except Exception as e:
                print(f"  Skipping index {i}: {e}")

        # Aggregate stats
        if before_snrs:
            md_lines.append(f"### Aggregate Statistics ({len(before_snrs)} clips)\n")
            md_lines.append(f"| Metric | Before (mean) | After (mean) | Improvement |\n")
            md_lines.append(f"|--------|---------------|--------------|-------------|\n")
            b_snr = np.mean(before_snrs)
            a_snr = np.mean(after_snrs)
            md_lines.append(f"| SNR (dB) | {b_snr:.2f} | {a_snr:.2f} | +{a_snr - b_snr:.2f} dB |\n")
            b_dur = np.mean(before_durs)
            a_dur = np.mean(after_durs)
            md_lines.append(f"| Duration (s) | {b_dur:.3f} | {a_dur:.3f} | -{b_dur - a_dur:.3f}s trimmed |\n")
            md_lines.append("\n---\n")

    # Write the report
    report_path = "preprocessing_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)

    print(f"\n[comparison] Done! Report saved to: {os.path.abspath(report_path)}")
    print(f"[comparison] Plots saved to: plots/")


if __name__ == "__main__":
    main()
