"""ASVspoof 2019 LA: protocol parsing, waveform loading, fixed-length crops.

Layout after extracting LA.zip into data/:
  data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train.trn,dev.trl,eval.trl}.txt
  data/LA/ASVspoof2019_LA_{train,dev,eval}/flac/<utt>.flac
Protocol line: SPEAKER UTT_ID - SYSTEM_ID KEY   (KEY = bonafide | spoof, SYSTEM_ID = '-' or A01..A19)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
LA = ROOT / "data" / "LA"
SR = 16000
N_SAMPLES = 64600  # ~4.04 s, the AASIST / RawNet2 input length
PROTO = {"train": "ASVspoof2019.LA.cm.train.trn.txt", "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
         "eval": "ASVspoof2019.LA.cm.eval.trl.txt"}


@dataclass(frozen=True)
class Utt:
    utt: str
    speaker: str
    attack: str      # "-" for bona fide, else A01..A19
    label: int       # 1 = bona fide (real), 0 = spoof (fake)
    path: Path


def read_protocol(split: str, root: Path = LA) -> list[Utt]:
    proto = root / "ASVspoof2019_LA_cm_protocols" / PROTO[split]
    audio = root / f"ASVspoof2019_LA_{split}" / "flac"
    out = []
    for line in proto.read_text().splitlines():
        spk, utt, _, attack, key = line.split()
        out.append(Utt(utt, spk, attack, int(key == "bonafide"), audio / f"{utt}.flac"))
    return out


def fix_length(x: np.ndarray, n: int = N_SAMPLES, rng: np.random.Generator | None = None) -> np.ndarray:
    """Crop (random if rng, else from the start) or repeat-pad to exactly n samples.
    Repeat-padding (not zeros) follows the AASIST reference: silence padding is itself a spoof cue."""
    if len(x) >= n:
        start = int(rng.integers(0, len(x) - n + 1)) if rng is not None else 0
        return x[start:start + n]
    reps = int(np.ceil(n / max(len(x), 1)))
    return np.tile(x, reps)[:n]


def load_wave(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32")
    assert sr == SR, f"{path}: {sr} Hz"
    return x if x.ndim == 1 else x.mean(1)


class ASVspoofLA(Dataset):
    def __init__(self, split: str, train: bool = False, limit: int | None = None, seed: int = 0, root: Path = LA):
        self.items = read_protocol(split, root)
        if limit:
            self.items = self.items[:limit]
        self.train, self.seed, self.epoch = train, seed, 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        u = self.items[i]
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.train else None
        x = fix_length(load_wave(u.path), rng=rng)
        return torch.from_numpy(x), u.label, i

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights [spoof, bonafide]; train set is ~9:1 spoof."""
        n1 = sum(u.label for u in self.items)
        n0 = len(self.items) - n1
        return torch.tensor([len(self.items) / (2 * n0), len(self.items) / (2 * n1)], dtype=torch.float32)
