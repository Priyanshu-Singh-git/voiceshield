"""EER and min t-DCF for spoofing countermeasures (scores: higher = more bona fide).

EER: the operating point where the false-accept rate (spoof accepted) equals the false-reject rate
(bona fide rejected). min t-DCF: the ASVspoof 2019 tandem cost with the official ASV error rates and
cost model (Kinnunen et al., 2018, 2020), normalised and minimised over thresholds.
"""
from __future__ import annotations

import numpy as np


def det_curve(bona: np.ndarray, spoof: np.ndarray):
    """Miss (bona fide rejected) and false-alarm (spoof accepted) rates at every threshold."""
    scores = np.concatenate([bona, spoof])
    labels = np.concatenate([np.ones(len(bona)), np.zeros(len(spoof))])
    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]
    tar_cum = np.cumsum(labels)
    non_cum = len(spoof) - (np.arange(1, len(labels) + 1) - tar_cum)
    frr = np.concatenate([[0], tar_cum / len(bona)])
    far = np.concatenate([[1], non_cum / len(spoof)])
    thr = np.concatenate([[scores[order[0]] - 1e-3], scores[order]])
    return frr, far, thr


def eer(bona: np.ndarray, spoof: np.ndarray) -> tuple[float, float]:
    frr, far, thr = det_curve(np.asarray(bona, float), np.asarray(spoof, float))
    i = int(np.nanargmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2), float(thr[i])


# ASVspoof 2019 official t-DCF cost model and the ASV system's error rates at its EER threshold
# on the LA evaluation set (from the organisers' ASV scores).
COST = {"Pspoof": 0.05, "Ptar": 0.95 * 0.99, "Pnon": 0.95 * 0.01,
        "Cmiss_asv": 1, "Cfa_asv": 10, "Cmiss_cm": 1, "Cfa_cm": 10}


def min_tdcf(bona: np.ndarray, spoof: np.ndarray, pfa_asv: float, pmiss_asv: float, pmiss_spoof_asv: float) -> float:
    """Normalised min t-DCF (ASVspoof 2019 formulation) given the ASV error rates."""
    c = COST
    frr, far, _ = det_curve(np.asarray(bona, float), np.asarray(spoof, float))
    C1 = c["Ptar"] * (c["Cmiss_cm"] - c["Cmiss_asv"] * pmiss_asv) - c["Pnon"] * c["Cfa_asv"] * pfa_asv
    C2 = c["Cfa_cm"] * c["Pspoof"] * (1 - pmiss_spoof_asv)
    tdcf = C1 * frr + C2 * far
    norm = tdcf / min(C1, C2)
    return float(norm.min())
