"""
data/loader.py
==============
Streaming data loader for Mozilla Common Voice v26.0.
Reads .mp3 clips directly from TAR archives -- no extraction needed.
"""

import io
import os
import tarfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple

import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CORPUS_VERSION = "cv-corpus-26.0-2026-06-12"
Split = Literal["train", "dev", "test", "validated", "invalidated", "other"]


@dataclass
class ClipMeta:
    """Per-clip metadata from a TSV row."""
    client_id:  str
    path:       str
    sentence:   str
    sentence_id: str
    sentence_domain: str
    up_votes:   int
    down_votes: int
    age:        str
    gender:     str
    accents:    str
    locale:     str
    segment:    str


def load_tsv_from_tar(tar_path: str, locale: str, split: str) -> pd.DataFrame:
    """Stream a split TSV from inside the TAR archive."""
    tsv = f"{CORPUS_VERSION}/{locale}/{split}.tsv"
    prefix = f"{CORPUS_VERSION}/{locale}/clips/"
    with tarfile.open(tar_path, "r|*") as tar:
        for m in tar:
            if m.name == tsv:
                f = tar.extractfile(m)
                df = pd.read_csv(f, sep="\t", dtype=str).fillna("")
                df["archive_path"] = prefix + df["path"]
                df["locale"] = locale
                return df
    raise FileNotFoundError(f"'{tsv}' not found in {tar_path}")


def build_tar_index(tar_path: str, locale: str,
                    paths: List[str], cache_dir: Optional[str] = None) -> Dict[str, int]:
    """Build byte-offset index for O(1) random access. Cached to disk."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = Path(cache_dir) / f"idx_{locale}.pkl"
        if cache.exists():
            import pickle
            with open(cache, "rb") as f:
                return pickle.load(f)

    targets = set(paths)
    index: Dict[str, int] = {}
    with tarfile.open(tar_path, "r") as tar:
        for m in tqdm(tar.getmembers(), desc=f"Indexing {locale} TAR"):
            if m.name in targets:
                index[m.name] = m.offset_data
                if len(index) == len(targets):
                    break

    if cache_dir:
        import pickle
        with open(cache, "wb") as f:
            pickle.dump(index, f)
    return index


class CommonVoiceDataset(Dataset):
    """
    PyTorch Dataset -- streams audio from TAR archives on-the-fly.

    Args:
        tar_path:    Path to the .tar file.
        locale:      'zh-CN' or 'bn'.
        split:       'train' | 'dev' | 'test' | 'validated'.
        sample_rate: Target sample rate (default 16000 Hz).
        cache_dir:   Directory for TAR index cache.
        use_index:   Use random-access byte-offset index (faster after first scan).
        transform:   Optional callable applied to waveform tensor.
    """
    def __init__(self, tar_path: str, locale: str, split: str = "train",
                 sample_rate: int = 16_000, cache_dir: Optional[str] = None,
                 use_index: bool = True, transform=None):
        self.tar_path    = str(tar_path)
        self.locale      = locale
        self.sample_rate = sample_rate
        self.transform   = transform
        self.use_index   = use_index

        print(f"[Dataset] Loading {locale}/{split} ...")
        self.df = load_tsv_from_tar(self.tar_path, locale, split)
        self.df["up_votes"]   = pd.to_numeric(self.df["up_votes"],   errors="coerce").fillna(0).astype(int)
        self.df["down_votes"] = pd.to_numeric(self.df["down_votes"], errors="coerce").fillna(0).astype(int)
        self.df = self.df.reset_index(drop=True)
        print(f"  -> {len(self.df):,} clips")

        self._index: Optional[Dict[str, int]] = None
        if use_index:
            self._index = build_tar_index(self.tar_path, locale,
                                          self.df["archive_path"].tolist(), cache_dir)

    def _load(self, path: str) -> Tuple[torch.Tensor, int]:
        if self.use_index and self._index:
            with open(self.tar_path, "rb") as f:
                f.seek(self._index[path])
                buf = io.BytesIO(f.read(12 * 1024 * 1024))
            return torchaudio.load(buf, format="mp3")
        with tarfile.open(self.tar_path, "r|*") as tar:
            for m in tar:
                if m.name == path:
                    buf = io.BytesIO(tar.extractfile(m).read())
                    return torchaudio.load(buf, format="mp3")
        raise FileNotFoundError(path)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, ClipMeta]:
        row = self.df.iloc[idx]
        try:
            wf, sr = self._load(row["archive_path"])
        except Exception as e:
            warnings.warn(f"Load failed {row['archive_path']}: {e}")
            wf, sr = torch.zeros(1, self.sample_rate), self.sample_rate

        if sr != self.sample_rate:
            wf = torchaudio.transforms.Resample(sr, self.sample_rate)(wf)
        if wf.shape[0] > 1:
            wf = wf.mean(0, keepdim=True)
        if self.transform:
            wf = self.transform(wf)

        meta = ClipMeta(
            client_id=row.get("client_id",""), path=row.get("path",""),
            sentence=row.get("sentence",""),   sentence_id=row.get("sentence_id",""),
            sentence_domain=row.get("sentence_domain",""),
            up_votes=int(row.get("up_votes",0)), down_votes=int(row.get("down_votes",0)),
            age=row.get("age",""),             gender=row.get("gender",""),
            accents=row.get("accents",""),     locale=row.get("locale",self.locale),
            segment=row.get("segment",""),
        )
        return wf, self.sample_rate, meta

    def quality_filter(self, min_up: int = 2, max_down: int = 0):
        """Return a quality-filtered copy."""
        clone = CommonVoiceDataset.__new__(CommonVoiceDataset)
        clone.__dict__.update(self.__dict__)
        mask = (self.df["up_votes"] >= min_up) & (self.df["down_votes"] <= max_down)
        clone.df = self.df[mask].reset_index(drop=True)
        print(f"[quality_filter] {len(clone.df):,} clips remain.")
        return clone


def collate_fn(batch):
    """Pad variable-length waveforms to the longest in the batch."""
    wfs, _, metas = zip(*batch)
    lengths = torch.tensor([w.shape[-1] for w in wfs])
    padded  = torch.zeros(len(wfs), 1, int(lengths.max()))
    for i, w in enumerate(wfs):
        padded[i, :, :w.shape[-1]] = w
    return padded, lengths, list(metas)


def make_loader(dataset: CommonVoiceDataset, batch_size=16,
                shuffle=True, num_workers=4) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      collate_fn=collate_fn)


class CrossLingualIterator:
    """
    Yields (zh_batch, bn_batch) unpaired mini-batches for EOT training.
    The unpaired setting is the core of the Schrodinger Bridge problem.
    """
    def __init__(self, src_loader: DataLoader, tgt_loader: DataLoader):
        self.src, self.tgt = src_loader, tgt_loader

    def __iter__(self) -> Iterator:
        si, ti = iter(self.src), iter(self.tgt)
        while True:
            try:   s = next(si)
            except StopIteration: si = iter(self.src); s = next(si)
            try:   t = next(ti)
            except StopIteration: ti = iter(self.tgt); t = next(ti)
            yield s, t

    def __len__(self): return max(len(self.src), len(self.tgt))
