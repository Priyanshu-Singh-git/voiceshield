import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from voiceshield.data import N_SAMPLES, ASVspoofLA, fix_length, read_protocol  # noqa: E402
from voiceshield.features import LFCC, dct_matrix, linear_filterbank  # noqa: E402
from voiceshield.metrics import eer, min_tdcf  # noqa: E402
from voiceshield.models import SincConv, build  # noqa: E402


def fake_la(root: Path, n=12):
    """Tiny ASVspoof-shaped tree: bona fide = low tone, spoof = high tone (trivially separable)."""
    rng = np.random.default_rng(0)
    (root / "ASVspoof2019_LA_cm_protocols").mkdir(parents=True)
    for split, tag in (("train", "T"), ("dev", "D"), ("eval", "E")):
        d = root / f"ASVspoof2019_LA_{split}" / "flac"; d.mkdir(parents=True)
        lines = []
        for i in range(n):
            bona = i % 3 == 0
            t = np.arange(int(16000 * rng.uniform(1.5, 5))) / 16000
            x = 0.3 * np.sin(2 * np.pi * (300 if bona else 5000) * t) + 0.01 * rng.standard_normal(len(t))
            utt = f"LA_{tag}_{i:07d}"
            sf.write(d / f"{utt}.flac", x.astype(np.float32), 16000)
            lines.append(f"LA_0001 {utt} - {'-' if bona else 'A0' + str(1 + i % 6)} {'bonafide' if bona else 'spoof'}")
        name = {"train": "ASVspoof2019.LA.cm.train.trn.txt", "dev": "ASVspoof2019.LA.cm.dev.trl.txt", "eval": "ASVspoof2019.LA.cm.eval.trl.txt"}[split]
        (root / "ASVspoof2019_LA_cm_protocols" / name).write_text("\n".join(lines) + "\n")
    return root


def test_eer_known_values():
    assert eer(np.array([2.0, 3.0]), np.array([0.0, 1.0]))[0] == 0.0            # perfectly separated
    rng = np.random.default_rng(0)
    b, s = rng.normal(1, 1, 20000), rng.normal(-1, 1, 20000)                      # d' = 2 -> EER = Phi(-1) = 0.1587
    assert eer(b, s)[0] == pytest.approx(0.1587, abs=0.01)


def test_min_tdcf_is_zero_for_perfect_cm_and_at_most_one_otherwise():
    b, s = np.array([5.0, 6.0]), np.array([-5.0, -6.0])
    assert min_tdcf(b, s, 0.01, 0.02, 0.5) == pytest.approx(0.0, abs=1e-9)
    rng = np.random.default_rng(1)
    assert min_tdcf(rng.normal(0, 1, 500), rng.normal(0, 1, 500), 0.01, 0.02, 0.5) <= 1.0 + 1e-9


def test_dct_is_orthonormal_and_filterbank_covers_band():
    m = dct_matrix(20, 20)
    assert torch.allclose(m.T @ m, torch.eye(20), atol=1e-5)
    fb = linear_filterbank(512, 20, 16000)
    assert fb.shape == (257, 20) and (fb.sum(1)[5:-5] > 0).all()


def test_lfcc_shape():
    f = LFCC()(torch.randn(2, 16000))
    assert f.shape == (2, 60, 101)


def test_sinc_filters_pass_their_own_band():
    sc = SincConv(10, 257)
    h = sc.filters().squeeze(1)
    H = torch.fft.rfft(h, n=8192).abs()
    freqs = torch.fft.rfftfreq(8192, 1 / 16000)
    for i in (2, 6):
        f1, f2 = float(sc.f1[i]), float(sc.f1[i] + sc.band[i])
        centre = H[i, (freqs >= f1) & (freqs <= f2)].mean()
        far = H[i, freqs > f2 + 2000].mean() if (freqs > f2 + 2000).any() else torch.tensor(0.0)
        assert centre > 0.8 and far < 0.1


def test_fix_length_crop_and_repeat():
    x = np.arange(100, dtype=np.float32)
    assert np.array_equal(fix_length(x, 250), np.tile(x, 3)[:250])
    y = fix_length(np.arange(1000, dtype=np.float32), 300, rng=np.random.default_rng(0))
    assert len(y) == 300 and np.all(np.diff(y) == 1)


@pytest.mark.parametrize("name", ["lcnn", "rawnet2", "aasist_l", "aasist"])
def test_models_forward_and_backward(name):
    m = build(name)
    y = m(torch.randn(2, N_SAMPLES))
    assert y.shape == (2, 2)
    y.sum().backward()


def test_dataset_reads_protocol(tmp_path):
    root = fake_la(tmp_path / "LA")
    items = read_protocol("train", root)
    assert len(items) == 12 and sum(u.label for u in items) == 4
    x, lab, i = ASVspoofLA("train", train=True, root=root)[0]
    assert x.shape == (N_SAMPLES,) and lab == 1


def test_train_script_end_to_end(tmp_path):
    root = fake_la(tmp_path / "LA", n=24)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "train" / "train_cm.py"), "--model", "lcnn", "--root", str(root),
                        "--out", str(out), "--epochs", "8", "--bs", "6", "--workers", "0", "--lr", "1e-3"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    final = json.loads(r.stdout.strip().splitlines()[-1])
    assert final["eval_eer"] <= 0.25                     # tones are trivially separable
    lines = (out / "scores_eval.txt").read_text().splitlines()
    assert len(lines) == 24 and len(lines[0].split()) == 4
