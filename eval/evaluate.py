"""Official-style evaluation of countermeasure score files on ASVspoof 2019 LA eval.

  python eval/evaluate.py kaggle_runs/<job>/out/<model>/scores_eval.txt [...more score files]
Per file: pooled EER, min t-DCF (ASVspoof 2019 cost model + organisers' ASV scores), per-attack EER
(bona fide vs each of A07-A19; A16 = A04 and A19 = A06 algorithms seen in training, the rest unseen).
Writes eval/results.json (merged by model name).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from voiceshield.metrics import eer, min_tdcf  # noqa: E402

ASV = ROOT / "data" / "LA" / "ASVspoof2019_LA_asv_scores" / "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
KNOWN = {"A16": "A04", "A19": "A06"}   # eval attacks that reuse a training-set algorithm


def asv_error_rates(path=ASV):
    rows = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    key = np.array([r[1] for r in rows]); s = np.array([float(r[2]) for r in rows])
    tar, non, spf = s[key == "target"], s[key == "nontarget"], s[key == "spoof"]
    _, thr = eer(tar, non)
    return {"pfa_asv": float((non >= thr).mean()), "pmiss_asv": float((tar < thr).mean()),
            "pmiss_spoof_asv": float((spf < thr).mean())}


def evaluate(scores_path, asv):
    rows = [l.split() for l in Path(scores_path).read_text().splitlines() if l.strip()]
    attack = np.array([r[1] for r in rows]); lab = np.array([r[2] == "bonafide" for r in rows]); s = np.array([float(r[3]) for r in rows])
    bona, spoof = s[lab], s[~lab]
    pooled, thr = eer(bona, spoof)
    per = {}
    for a in sorted(set(attack[~lab])):
        per[a] = {"eer": eer(bona, s[attack == a])[0], "seen_in_training": a in KNOWN}
    return {"n_bona": int(lab.sum()), "n_spoof": int((~lab).sum()), "eer": pooled, "threshold": thr,
            "min_tdcf": min_tdcf(bona, spoof, **asv), "per_attack": per}


def main():
    asv = asv_error_rates()
    out = ROOT / "eval" / "results.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    res["_asv"] = asv
    for p in sys.argv[1:]:
        name = Path(p).parent.name
        r = evaluate(p, asv)
        meta = Path(p).parent / "meta.json"
        if meta.exists():
            r["meta"] = json.loads(meta.read_text())
        log = Path(p).parent / "log.jsonl"
        if log.exists():
            r["log"] = [json.loads(l) for l in log.read_text().splitlines()]
        res[name] = r
        print(f"{name:10s} EER {100 * r['eer']:.2f}%  min t-DCF {r['min_tdcf']:.4f}  "
              + " ".join(f"{a}:{100 * v['eer']:.1f}" for a, v in r["per_attack"].items()))
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
