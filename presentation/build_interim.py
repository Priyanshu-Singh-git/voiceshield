"""INTERIM deck (not for publishing): authors' published AASIST numbers + measurements of their released model on
the 16 demo clips, until my own trained model's results replace it (presentation/build_deck.py)."""
import json, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "presentation"; FIG = ROOT / "assets" / "figures"
meta = json.loads((ROOT / "docs" / "meta.json").read_text())
bc = json.loads((ROOT / "eval" / "browser_check.json").read_text())
fake = [s for s in meta["samples"] if s["attack"] != "-"]
correct = sum((r["real"]) == (s["attack"] == "-") for r, s in zip(bc["rows"], meta["samples"]))
d = json.loads((OUT / "deck_data.json").read_text())
d.update({
  "cover_stats": [["0.83%", "error rate (AASIST, published)"], ["71,237", "test clips"]],
  "result_stats": [["0.83%", "equal error rate", "AASIST, published by its authors"], [f"{correct}/{len(meta['samples'])}", "demo clips correct", "measured in the browser"],
                   ["0.0275", "min t-DCF", "official ASVspoof metric (published)"], [f"{bc['median_ms'] / 1000:.1f} s", "per clip in a browser", "no server, no upload"]],
  "measure_note": "Interim: error rates are the AASIST authors' published results on the ASVspoof 2019 LA evaluation set (71,237 clips, 13 attack systems). "
                  "Demo-clip results are measured on their released model running in my implementation.",
  "how_note": "AASIST learns from raw audio which frequency bands and moments give a synthetic voice away, then lets the two graphs vote through a shared master node. 297,866 parameters: small enough to run in a web browser.",
  "attack_note": "Detector score for one test clip from each attack system (real voices score about +14). Every clip lands far below the threshold, so every AI voice is flagged. Grey = a system whose algorithm also appears in training; yellow = never seen.",
  "per_attack": [{"attack": s["attack"], "value": s["py_score"], "seen": s["attack"] in ("A16", "A19")} for s in fake],
  "attack_chart": {"series": "score", "fmt": "0.0", "legend": "Below 0 = flagged as AI", "title": "Detector score per test clip (real voices ≈ +14; below 0 = AI)"},
  "bench": [{"model": "AASIST (official model)", "params": "297k", "eer": "0.83%", "tdcf": "0.0275", "source": "clovaai/aasist", "mine": False},
            {"model": "AASIST-L (official model)", "params": "85k", "eer": "0.99%", "tdcf": "0.0309", "source": "clovaai/aasist", "mine": False},
            {"model": "My AASIST, AASIST-L, RawNet2, LCNN", "params": "–", "eer": "training", "tdcf": "training", "source": "results tonight", "mine": True}],
  "bench_note": "Official numbers: the authors' released pretrained models (github.com/clovaai/aasist). My implementation reproduces their model exactly (score difference < 0.0001).",
  "assets": {**d["assets"], "shot": str(FIG / "demo_screenshot.png")},
})
(OUT / "deck_data.json").write_text(json.dumps(d, indent=2))
subprocess.run(["node", "build_pptx.js"], check=True, cwd=OUT)
subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(OUT / "export_slides.ps1")], check=True, cwd=OUT)
