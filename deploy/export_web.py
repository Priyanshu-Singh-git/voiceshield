"""Export the browser demo: AASIST -> ONNX (wave[1,64600] -> logits[1,2]), demo clips, meta.json.

  python deploy/export_web.py --ckpt kaggle_runs/voiceshield-train/vs/out/aasist/best.pt
Parity: ONNX Runtime vs PyTorch on every demo clip (max |score diff| < 1e-3).
Threshold: the dev-set EER threshold (chosen without looking at eval).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from voiceshield.data import LA, N_SAMPLES, fix_length, load_wave, read_protocol  # noqa: E402
from voiceshield.metrics import eer  # noqa: E402
from voiceshield.models import build, load_official_aasist  # noqa: E402

OUT = ROOT / "docs"
NAMES = {"A07": "TTS + WORLD vocoder", "A08": "neural TTS", "A09": "TTS (LSTM)", "A10": "Tacotron-style TTS",
         "A11": "Tacotron + Griffin-Lim", "A12": "WaveNet TTS", "A13": "TTS + voice conversion", "A14": "TTS + VC",
         "A15": "TTS + VC (WaveNet)", "A16": "TTS (seen type A04)", "A17": "voice conversion", "A18": "voice conversion (i-vector)",
         "A19": "voice conversion (seen type A06)"}


def dev_threshold(scores_dev: Path):
    rows = [l.split() for l in scores_dev.read_text().splitlines()]
    s = np.array([float(r[3]) for r in rows]); lab = np.array([r[2] == "bonafide" for r in rows])
    e, thr = eer(s[lab], s[~lab])
    return float(thr), float(np.std(s)), float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--results", default=str(ROOT / "eval" / "results.json"))
    ap.add_argument("--official", default=None, help="TEST ONLY: authors' AASIST.pth instead of a trained checkpoint (threshold 0)")
    args = ap.parse_args()
    if args.official:
        ck, net = {"model": "aasist"}, load_official_aasist(args.official)
    else:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        net = build(ck["model"]).eval()
        net.load_state_dict(ck["state"])
    for sub in ("models", "samples"):
        shutil.rmtree(OUT / sub, ignore_errors=True); (OUT / sub).mkdir(parents=True)
    onnx_path = OUT / "models" / f"voiceshield_{ck['model']}.onnx"
    torch.onnx.export(net, torch.randn(1, N_SAMPLES), str(onnx_path), opset_version=17,
                      input_names=["wave"], output_names=["logits"], do_constant_folding=True)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    samples, worst = [], 0.0
    proto = {u.utt: u for u in read_protocol("eval")}
    for f in sorted((LA / "ASVspoof2019_LA_eval" / "flac").glob("*.flac"), key=lambda p: (proto[p.stem].attack != "-", proto[p.stem].attack, p.stem)):
        u = proto[f.stem]
        x = fix_length(load_wave(f))[None]
        with torch.no_grad():
            ref = net(torch.from_numpy(x)).numpy()[0]
        got = sess.run(None, {"wave": x})[0][0]
        worst = max(worst, abs(float((got[1] - got[0]) - (ref[1] - ref[0]))))
        shutil.copy(f, OUT / "samples" / f.name)
        kind = "VC" if "conversion" in NAMES.get(u.attack, "") else "TTS"
        label = "Real" if u.attack == "-" else f"{u.attack} {kind}"
        samples.append({"file": f"samples/{f.name}", "label": label, "title": "Real speaker" if u.attack == "-" else f"{u.attack}: {NAMES.get(u.attack, 'spoof')}",
                        "attack": u.attack, "py_score": float(ref[1] - ref[0])})
    print(f"ONNX vs PyTorch max |score diff|: {worst:.2e}")
    assert worst < 1e-3, "ONNX export does not match PyTorch"
    # number real clips 1..n
    k = 0
    for s in samples:
        if s["attack"] == "-":
            k += 1; s["label"] = f"Real {k}"

    thr, scale, dev_e = (0.0, 8.0, None) if args.official else dev_threshold(Path(args.ckpt).parent / "scores_dev.txt")
    res = json.loads(Path(args.results).read_text()) if Path(args.results).exists() else {}
    r = res.get(ck["model"], {})
    rows = []
    if r:
        rows = [{"label": "Equal error rate (lower is better)", "value": f"{100 * r['eer']:.2f}%"},
                {"label": "min t-DCF (official ASVspoof metric)", "value": f"{r['min_tdcf']:.3f}"},
                {"label": "Test clips (real / AI)", "value": f"{r['n_bona']:,} / {r['n_spoof']:,}"},
                {"label": "Model size", "value": f"{r.get('meta', {}).get('params', 0):,} parameters"}]
    meta = {"model": onnx_path.name, "threshold": thr, "scale": max(scale, 1e-3), "dev_eer": dev_e, "samples": samples,
            "stats": {"rows": rows, "note": "ASVspoof 2019 LA evaluation set: 71,237 clips, 13 attack systems, 11 of them never seen in training."}}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"wrote {OUT} ({size:.1f} MB), threshold {thr:.3f}, dev EER {dev_e}")


if __name__ == "__main__":
    main()
