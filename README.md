# EOT / Schrodinger Bridge Cross-Lingual Speech Morphing

**Dataset:** Mozilla Common Voice v26.0 — Chinese (zh-CN) + Bengali (bn)  
**Goal:** Continuous, trajectory-based speech morphing between language manifolds  
using Entropic Optimal Transport and Schrodinger Bridges.

---

## Setup

```powershell
# venv activates automatically when you open a terminal in this folder
# First time only -- install dependencies:
pip install -r requirements.txt
```

## Project Structure

```
codes/
├── .venv/                    # Python virtual environment (auto-activated)
├── configs/
│   └── default.yaml          # All hyperparameters
├── data/
│   ├── loader.py             # TAR streaming data loader (no extraction)
│   ├── preprocessing.py      # Mel / WavLM feature extraction
│   └── explore.py            # Dataset statistics & plots
├── models/
│   ├── encoder.py            # Speech encoder (MelCNN / WavLM)
│   ├── ot_solver.py          # Sinkhorn EOT solver (balanced + unbalanced)
│   └── schrodinger_bridge.py # SB drift network + SDE simulator
├── training/
│   └── train_sb.py           # Main training loop
├── evaluation/
│   └── metrics.py            # MCD, perceptual metrics
├── scripts/
│   ├── run_explore.py        # Quick-launch: data exploration
│   └── run_train.py          # Quick-launch: training
├── notebooks/                # Jupyter notebooks for EDA
└── outputs/
    ├── checkpoints/          # Model weights
    ├── features/             # Extracted embeddings
    ├── exploration/          # Generated plots
    └── logs/                 # TensorBoard logs
```

## Quick Commands

```powershell
# Explore the dataset (generates plots in outputs/exploration/)
python -m data.explore --zh-tar ../Dataset/Chinese/Chinese --bn-tar ../Dataset/bengali/bengali

# Start training
python -m training.train_sb --config configs/default.yaml

# View TensorBoard
tensorboard --logdir outputs/logs
```

## Key Dataset Stats (CV v26.0, June 2026)

| Metric | Chinese (zh-CN) | Bengali (bn) |
|---|---|---|
| Validated clips | 190,017 | 44,739 |
| Validated hours | 239.59 h | 54.32 h |
| Train clips | 29,699 | 21,618 |
| Speakers | 7,580 | 23,010 |
| Avg duration | 4.54 s | 4.37 s |
