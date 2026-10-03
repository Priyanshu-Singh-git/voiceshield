"""Fill the deck from measured results and build it: deck_data.json -> deck.pptx -> slide PNGs + PDF.

  python presentation/build_deck.py --run kaggle_runs/voiceshield-train/vs/out --main aasist
Every number on a slide comes from eval/results.json, the score files and eval/browser_check.json.
Business numbers (AI clips caught, real voices wrongly flagged) use the DEV-set EER threshold applied to EVAL.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation"
FIG = ROOT / "assets" / "figures"
sys.path.insert(0, str(ROOT / "src"))
from voiceshield.metrics import eer  # noqa: E402

DEMO = "https://priyanshu-singh-git.github.io/voiceshield/"
REPO = "https://github.com/Priyanshu-Singh-git/voiceshield"
LABEL = {"lcnn": "LFCC + LCNN", "rawnet2": "RawNet2", "aasist_l": "AASIST-L", "aasist": "AASIST"}
PAPER = [("AASIST (official model)", "297k", "0.83%", "0.0275"), ("AASIST-L (official model)", "85k", "0.99%", "0.0309")]


def load_scores(p):
    rows = [l.split() for l in Path(p).read_text().splitlines() if l.strip()]
    return np.array([float(r[3]) for r in rows]), np.array([r[2] == "bonafide" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder with <model>/scores_{dev,eval}.txt")
    ap.add_argument("--main", default="aasist")
    a = ap.parse_args()
    run = Path(a.run)
    res = json.loads((ROOT / "eval" / "results.json").read_text())
    m = res[a.main]
    ds, dl = load_scores(run / a.main / "scores_dev.txt")
    _, thr = eer(ds[dl], ds[~dl])
    es, el = load_scores(run / a.main / "scores_eval.txt")
    caught = float((es[~el] <= thr).mean())          # AI clips flagged as AI
    false_alarm = float((es[el] <= thr).mean())      # real voices flagged as AI
    browser = json.loads((ROOT / "eval" / "browser_check.json").read_text())
    unseen = [k for k, v in m["per_attack"].items() if not v["seen_in_training"]]
    worst = max(m["per_attack"].items(), key=lambda kv: kv[1]["eer"])

    bench = [{"model": f"{LABEL[k]} (mine)", "params": f"{res[k]['meta']['params']:,}", "eer": f"{100 * res[k]['eer']:.2f}%",
              "tdcf": f"{res[k]['min_tdcf']:.4f}", "source": f"this work · {len(res[k].get('log', []))} epochs", "mine": True}
             for k in ("lcnn", "rawnet2", "aasist_l", "aasist") if k in res]
    bench += [{"model": n, "params": p, "eer": e, "tdcf": t, "source": "clovaai/aasist (100 epochs)", "mine": False} for n, p, e, t in PAPER]

    data = {
        "cover_stats": [[f"{100 * m['eer']:.2f}%", "equal error rate"], [f"{m['n_bona'] + m['n_spoof']:,}", "test clips"]],
        "result_stats": [[f"{100 * m['eer']:.2f}%", "equal error rate", "lower is better"],
                         [f"{100 * caught:.1f}%", "AI clips caught", f"{false_alarm * 100:.1f}% of real voices flagged"],
                         [f"{m['min_tdcf']:.3f}", "min t-DCF", "official ASVspoof metric"],
                         [f"{browser['median_ms'] / 1000:.1f} s", "per clip in a browser", "no server, no upload"]],
        "measure_note": (f"ASVspoof 2019 LA evaluation set: {m['n_bona']:,} real and {m['n_spoof']:,} AI clips from 13 attack systems, "
                         f"{len(unseen)} of them never seen in training. The pass/fail threshold is fixed on the separate development set."),
        "how_note": (f"{LABEL[a.main]} learns from raw audio which frequency bands and moments give a synthetic voice away, "
                     f"then lets the two graphs vote through a shared master node. {res[a.main]['meta']['params']:,} parameters: "
                     "small enough to run in a web browser."),
        "attack_note": (f"Error rate per attack system on the evaluation set. Hardest: {worst[0]} at {100 * worst[1]['eer']:.1f}%. "
                        "Systems marked grey reuse an algorithm from the training set; the rest were never seen."),
        "per_attack": [{"attack": k, "value": 100 * v["eer"], "seen": v["seen_in_training"]} for k, v in m["per_attack"].items()],
        "attack_chart": {"series": "EER %", "fmt": "0.0", "legend": "Yellow = never seen in training",
                         "title": "Equal error rate per attack system (%) · lower is better"},
        "bench": bench,
        "bench_note": ("My models: trained on Kaggle T4 GPUs for fewer epochs than the official recipe (12-hour session limit). "
                       "Official numbers: the authors' released pretrained models, from github.com/clovaai/aasist."),
        "demo_url": DEMO, "repo_url": REPO,
        "assets": {"render_wave": str(FIG / "render_wave.png"), "render_ribbon": str(FIG / "render_ribbon.png"),
                   "shot": str(FIG / "demo_screenshot.png"), "qr": str(OUT / "qr.png")},
    }
    import qrcode
    q = qrcode.QRCode(border=2, box_size=12); q.add_data(DEMO); q.make(fit=True)
    q.make_image(fill_color="#111111", back_color="white").convert("RGB").save(OUT / "qr.png")
    (OUT / "deck_data.json").write_text(json.dumps(data, indent=2))
    subprocess.run(["node", "build_pptx.js"], check=True, cwd=OUT)
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(OUT / "export_slides.ps1")], check=True, cwd=OUT)


if __name__ == "__main__":
    main()
