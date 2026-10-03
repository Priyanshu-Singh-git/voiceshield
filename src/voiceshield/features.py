"""LFCC (linear-frequency cepstral coefficients), the ASVspoof baseline front end, in torch.

LFCC = DCT( log( |STFT|^2 @ linear triangular filterbank ) ), plus deltas.
Linear (not mel) spacing keeps resolution in the high frequencies, where vocoder artefacts live.
Defaults follow the ASVspoof 2019/2021 baseline: 20 ms window, 10 ms hop, 512-pt FFT, 20 filters, 20 coeffs.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def linear_filterbank(n_fft: int, n_filters: int, sr: int) -> torch.Tensor:
    """(n_fft//2+1, n_filters) triangular filters equally spaced in Hz from 0 to sr/2."""
    n_bins = n_fft // 2 + 1
    freqs = torch.linspace(0, sr / 2, n_bins)
    edges = torch.linspace(0, sr / 2, n_filters + 2)
    fb = torch.zeros(n_bins, n_filters)
    for m in range(n_filters):
        lo, ce, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (freqs - lo) / (ce - lo)
        down = (hi - freqs) / (hi - ce)
        fb[:, m] = torch.clamp(torch.minimum(up, down), min=0)
    return fb


def dct_matrix(n: int, k: int) -> torch.Tensor:
    """Orthonormal DCT-II as an (n, k) matrix: c_k = sum_n x_n cos(pi k (n + 1/2) / N)."""
    ns = torch.arange(n).float()
    ks = torch.arange(k).float()
    m = torch.cos(math.pi * ks[None, :] * (ns[:, None] + 0.5) / n) * math.sqrt(2 / n)
    m[:, 0] /= math.sqrt(2)
    return m


def deltas(x: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Regression deltas along time (last dim): d_t = sum_k k (x_{t+k} - x_{t-k}) / (2 sum k^2)."""
    n = width // 2
    pad = torch.nn.functional.pad(x, (n, n), mode="replicate")
    num = sum(k * (pad[..., n + k:pad.shape[-1] - n + k] - pad[..., n - k:pad.shape[-1] - n - k]) for k in range(1, n + 1))
    return num / (2 * sum(k * k for k in range(1, n + 1)))


class LFCC(nn.Module):
    def __init__(self, sr=16000, win=320, hop=160, n_fft=512, n_filters=20, n_coef=20, with_deltas=True):
        super().__init__()
        self.win, self.hop, self.n_fft, self.with_deltas = win, hop, n_fft, with_deltas
        self.register_buffer("window", torch.hamming_window(win))
        self.register_buffer("fb", linear_filterbank(n_fft, n_filters, sr))
        self.register_buffer("dct", dct_matrix(n_filters, n_coef))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, T) waveform -> (B, C, frames) with C = n_coef * 3 if deltas else n_coef."""
        spec = torch.stft(wav, self.n_fft, self.hop, self.win, self.window, center=True, return_complex=True)
        power = spec.abs().pow(2).transpose(1, 2)                    # B, frames, bins
        logfb = torch.log(power @ self.fb + torch.finfo(power.dtype).eps)
        c = (logfb @ self.dct).transpose(1, 2)                       # B, coef, frames
        if self.with_deltas:
            d1 = deltas(c)
            c = torch.cat([c, d1, deltas(d1)], 1)
        return c
