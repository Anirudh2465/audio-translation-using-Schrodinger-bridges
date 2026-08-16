"""
data/explore.py
===============
Streams TSV metadata from the TAR archives and generates statistics + plots.

Usage:
    python -m data.explore \
        --zh-tar ../../Dataset/Chinese/Chinese \
        --bn-tar ../../Dataset/bengali/bengali \
        --out-dir ../outputs/exploration
"""

import argparse
import tarfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

CORPUS  = "cv-corpus-26.0-2026-06-12"
PALETTE = {"zh-CN": "#4A90D9", "bn": "#E87040"}
sns.set_theme(style="darkgrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "#1a1a2e"})


def load_tsv(tar_path: str, locale: str, split: str) -> pd.DataFrame:
    tsv = f"{CORPUS}/{locale}/{split}.tsv"
    with tarfile.open(tar_path, "r|*") as tar:
        for m in tar:
            if m.name == tsv:
                f = tar.extractfile(m)
                import io
                return pd.read_csv(io.BytesIO(f.read()), sep="\t", dtype=str).fillna("")
    warnings.warn(f"{split}.tsv not found for {locale}")
    return pd.DataFrame()


def _bar(ax, x, width, locale, vals, labels, color):
    i = list(PALETTE).index(locale)
    bars = ax.bar(x + i * width, vals, width, label=locale, color=color,
                  alpha=0.88, edgecolor="white", linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(vals)*0.01,
                f"{v:,}", ha="center", va="bottom", fontsize=8, color="white")


def plot_split_sizes(dfs: dict, out: Path):
    splits = ["train","dev","test","validated"]
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    x = np.arange(len(splits)); w = 0.35
    for locale, color in PALETTE.items():
        vals = [len(dfs.get(locale,{}).get(s, pd.DataFrame())) for s in splits]
        _bar(ax, x, w, locale, vals, splits, color)
    ax.set_xticks(x + w/2); ax.set_xticklabels(splits, color="white")
    ax.set_ylabel("Clips", color="white"); ax.tick_params(colors="white")
    ax.legend(facecolor="#16213e", labelcolor="white")
    ax.set_title("Clips per Split — zh-CN vs Bengali", color="white", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    plt.tight_layout(); plt.savefig(out/"split_sizes.png", bbox_inches="tight",
                                    facecolor="#1a1a2e"); plt.close()
    print(f"  Saved: {out/'split_sizes.png'}")


def plot_gender(dfs: dict, out: Path):
    fig, axes = plt.subplots(1,2,figsize=(12,6),facecolor="#1a1a2e")
    label_map = {"male_masculine":"Male","female_feminine":"Female",
                 "non-binary":"Non-binary","do_not_wish_to_say":"Prefer not say","":"Unspecified"}
    for ax, (locale, split_dfs) in zip(axes, dfs.items()):
        df = split_dfs.get("validated", pd.DataFrame())
        if df.empty: continue
        gc = df["gender"].replace("","Unspecified").map(
            lambda g: label_map.get(g, g)).value_counts()
        colors = ["#4A90D9","#E87040","#50C878","#FFD700","#888"][:len(gc)]
        ax.pie(gc.values, labels=gc.index, autopct="%1.1f%%", colors=colors,
               startangle=90, textprops={"color":"white","fontsize":9})
        ax.set_title(f"{locale} — Gender (Validated)", color="white")
        ax.set_facecolor("#1a1a2e")
    fig.suptitle("Gender Distribution", color="white", fontsize=13)
    plt.tight_layout(); plt.savefig(out/"gender.png", bbox_inches="tight",
                                    facecolor="#1a1a2e"); plt.close()
    print(f"  Saved: {out/'gender.png'}")


def plot_age(dfs: dict, out: Path):
    order  = ["teens","twenties","thirties","fourties","fifties","sixties"]
    labels = ["Teens","20s","30s","40s","50s","60s"]
    fig, axes = plt.subplots(1,2,figsize=(13,5),facecolor="#1a1a2e")
    for ax, (locale, split_dfs) in zip(axes, dfs.items()):
        df = split_dfs.get("validated", pd.DataFrame())
        if df.empty: continue
        vc = df["age"].value_counts()
        counts = [vc.get(a,0) for a in order]
        ax.set_facecolor("#16213e")
        bars = ax.bar(labels, counts, color=PALETTE[locale], alpha=0.88,
                      edgecolor="white", linewidth=0.4)
        for b,c in zip(bars,counts):
            if c: ax.text(b.get_x()+b.get_width()/2, b.get_height()+10,
                          f"{c:,}", ha="center", va="bottom", fontsize=8, color="white")
        ax.set_title(f"{locale} — Age (Validated)", color="white")
        ax.tick_params(colors="white"); ax.set_ylabel("Clips", color="white")
    fig.suptitle("Age Distribution (Declared Only)", color="white", fontsize=13)
    plt.tight_layout(); plt.savefig(out/"age.png", bbox_inches="tight",
                                    facecolor="#1a1a2e"); plt.close()
    print(f"  Saved: {out/'age.png'}")


def plot_sentence_len(dfs: dict, out: Path):
    fig, axes = plt.subplots(1,2,figsize=(12,5),facecolor="#1a1a2e")
    for ax, (locale, split_dfs) in zip(axes, dfs.items()):
        df = split_dfs.get("train", pd.DataFrame())
        if df.empty: continue
        lens = df["sentence"].str.len()
        ax.set_facecolor("#16213e")
        ax.hist(lens.clip(0,150), bins=50, color=PALETTE[locale], alpha=0.88,
                edgecolor="white", linewidth=0.3)
        ax.axvline(lens.mean(), color="yellow", linestyle="--", linewidth=1.5,
                   label=f"Mean {lens.mean():.0f} chars")
        ax.set_title(f"{locale} — Sentence Length (Train)", color="white")
        ax.set_xlabel("Characters", color="white"); ax.set_ylabel("Clips", color="white")
        ax.tick_params(colors="white"); ax.legend(facecolor="#16213e", labelcolor="white")
    fig.suptitle("Sentence Character Length", color="white", fontsize=13)
    plt.tight_layout(); plt.savefig(out/"sentence_len.png", bbox_inches="tight",
                                    facecolor="#1a1a2e"); plt.close()
    print(f"  Saved: {out/'sentence_len.png'}")


def print_summary(dfs: dict):
    print("\n" + "="*55)
    print("  DATASET SUMMARY  --  Common Voice v26.0")
    print("="*55)
    for locale, split_dfs in dfs.items():
        v  = split_dfs.get("validated", pd.DataFrame())
        tr = split_dfs.get("train",     pd.DataFrame())
        dv = split_dfs.get("dev",       pd.DataFrame())
        te = split_dfs.get("test",      pd.DataFrame())
        print(f"\n  [{locale}]")
        print(f"    Validated clips  : {len(v):>8,}")
        print(f"    Train / Dev / Test: {len(tr):>6,} / {len(dv):>6,} / {len(te):>6,}")
        if not v.empty:
            print(f"    Unique speakers  : {v['client_id'].nunique():>8,}")
            up = pd.to_numeric(v['up_votes'], errors='coerce')
            print(f"    Mean up-votes    : {up.mean():>8.2f}")
            print(f"    Unique sentences : {v['sentence_id'].nunique():>8,}")
    print("="*55)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zh-tar",  required=True)
    parser.add_argument("--bn-tar",  required=True)
    parser.add_argument("--out-dir", default="../outputs/exploration")
    args = parser.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    archives = {"zh-CN": args.zh_tar, "bn": args.bn_tar}
    splits   = ["train","dev","test","validated"]

    print("Loading TSV metadata ...")
    dfs = {}
    for locale, tar in archives.items():
        dfs[locale] = {}
        for sp in splits:
            df = load_tsv(tar, locale, sp)
            dfs[locale][sp] = df
            if not df.empty:
                print(f"  {locale}/{sp}: {len(df):,} rows")

    print("\nGenerating plots ...")
    plot_split_sizes(dfs, out)
    plot_gender(dfs, out)
    plot_age(dfs, out)
    plot_sentence_len(dfs, out)
    print_summary(dfs)
    print(f"\nDone. Plots saved to {out.resolve()}")


if __name__ == "__main__":
    main()
