import sys
import os
import yaml
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.loader import CommonVoiceDataset

def compute_noise_metrics(wf, sr):
    wf_np = wf.squeeze().numpy()
    if wf_np.size == 0 or np.all(wf_np == 0):
        return {"duration": 0.0, "rms": 0.0, "zcr": 0.0, "snr_db": 0.0}
        
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

    return {"duration": duration, "rms": rms, "zcr": zcr, "snr_db": snr}

def main():
    os.makedirs("outputs", exist_ok=True)
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    data_cfg = cfg["data"]
    
    print("[noise_export] Loading datasets...")
    # Using max_samples to limit the export time. Set to None to process entire dataset.
    MAX_SAMPLES = 1000 
    
    zh_ds = CommonVoiceDataset(
        tar_path=data_cfg["zh_tar"], locale="zh-CN", split="train",
        sample_rate=data_cfg["sample_rate"], cache_dir=data_cfg.get("cache_dir"),
        max_samples=MAX_SAMPLES
    ).quality_filter(min_up=data_cfg["min_up_votes"], max_down=data_cfg["max_down_votes"])
    
    bn_ds = CommonVoiceDataset(
        tar_path=data_cfg["bn_tar"], locale="bn", split="train",
        sample_rate=data_cfg["sample_rate"], cache_dir=data_cfg.get("cache_dir"),
        max_samples=MAX_SAMPLES
    ).quality_filter(min_up=data_cfg["min_up_votes"], max_down=data_cfg["max_down_votes"])
    
    datasets = {"zh-CN": zh_ds, "bn": bn_ds}
    all_data = []
    
    for loc, ds in datasets.items():
        print(f"\n[noise_export] Computing noise ratios for {loc}...")
        for i in tqdm(range(len(ds)), desc=loc):
            try:
                wf, sr, meta = ds[i]
                metrics = compute_noise_metrics(wf, sr)
                metrics["locale"] = loc
                metrics["path"] = meta.path
                metrics["client_id"] = meta.client_id
                all_data.append(metrics)
            except Exception as e:
                print(f"Skipping index {i} due to error: {e}")

    df = pd.DataFrame(all_data)
    
    # Save to CSV
    csv_path = "outputs/noise_ratios_export.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[noise_export] Done! Exported {len(df)} records to {csv_path}")
    
    # Print summary
    print("\n--- Noise Ratio (SNR) Summary ---")
    for loc in df['locale'].unique():
        loc_df = df[df['locale'] == loc]
        print(f"Locale: {loc}")
        print(f"  Mean SNR: {loc_df['snr_db'].mean():.2f} dB")
        print(f"  Median SNR: {loc_df['snr_db'].median():.2f} dB")
        print(f"  Min SNR: {loc_df['snr_db'].min():.2f} dB")
        print(f"  Max SNR: {loc_df['snr_db'].max():.2f} dB")
        print()

if __name__ == "__main__":
    main()
