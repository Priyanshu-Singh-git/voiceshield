"""Spoofing countermeasure models, written from the papers.

SincConv   - band-pass filters h[n] = 2 f2 sinc(2 f2 n) - 2 f1 sinc(2 f1 n), mel-spaced edges, Hamming window
             (Ravanelli & Bengio 2018; used as fixed front end in RawNet2 / AASIST)
LCNN       - light CNN with Max-Feature-Map activations on LFCC (Wu et al. 2018; ASVspoof baseline style)
RawNet2    - raw waveform: SincConv -> residual blocks with filter-wise feature-map scaling -> GRU
             (Tak et al., ICASSP 2021)
AASIST     - SincConv -> 2-D residual encoder -> spectral & temporal graph attention ->
             heterogeneous stacking graph attention with master nodes -> readout
             (Jung et al., ICASSP 2022). AASIST-L is the 85k-parameter configuration.
All models output 2 logits: index 1 = bona fide, index 0 = spoof. Score = logit[1] - logit[0].
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .features import LFCC


# ----------------------------------------------------------------------------------- sinc front end
def _sinc(x: torch.Tensor) -> torch.Tensor:
    """Normalised sinc sin(pi x)/(pi x), written out (ONNX has no sinc op); identical to torch.sinc."""
    px = torch.pi * x
    safe = torch.where(x == 0, torch.ones_like(px), px)
    return torch.where(x == 0, torch.ones_like(px), torch.sin(safe) / safe)

class SincConv(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int, sr: int = 16000, learnable: bool = False):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels, self.kernel_size, self.sr = out_channels, kernel_size, sr
        to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)  # noqa: E731
        to_hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)  # noqa: E731
        mel = np.linspace(to_mel(0), to_mel(sr / 2), out_channels + 1)
        edges = torch.tensor(to_hz(mel), dtype=torch.float32)
        self.f1 = nn.Parameter(edges[:-1].clone(), requires_grad=learnable)
        self.band = nn.Parameter((edges[1:] - edges[:-1]).clone(), requires_grad=learnable)
        n = torch.arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1)
        self.register_buffer("n", n)
        self.register_buffer("window", torch.hamming_window(kernel_size, periodic=False))

    def filters(self) -> torch.Tensor:
        f1 = self.f1.abs()
        f2 = f1 + self.band.abs()
        t = self.n[None, :] / self.sr
        high = 2 * f2[:, None] / self.sr * _sinc(2 * f2[:, None] * t)
        low = 2 * f1[:, None] / self.sr * _sinc(2 * f1[:, None] * t)
        return ((high - low) * self.window[None, :]).unsqueeze(1)   # (C, 1, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:              # (B, 1, T) -> (B, C, T-K+1)
        return F.conv1d(x, self.filters())


# ----------------------------------------------------------------------------------- LFCC + LCNN
class MFM(nn.Module):
    """Max-Feature-Map: split channels in two halves and take the elementwise max (a learned competition)."""

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)


def mfm_conv(cin, cout, k, s=1, p=0):
    return nn.Sequential(nn.Conv2d(cin, 2 * cout, k, s, p), MFM())


class LCNN(nn.Module):
    def __init__(self, n_lfcc: int = 20, dropout: float = 0.5):
        super().__init__()
        self.lfcc = LFCC(n_coef=n_lfcc, n_filters=n_lfcc)
        self.net = nn.Sequential(
            mfm_conv(1, 32, 5, 1, 2), nn.MaxPool2d(2),
            mfm_conv(32, 32, 1), nn.BatchNorm2d(32), mfm_conv(32, 48, 3, 1, 1), nn.MaxPool2d(2), nn.BatchNorm2d(48),
            mfm_conv(48, 48, 1), nn.BatchNorm2d(48), mfm_conv(48, 64, 3, 1, 1), nn.MaxPool2d(2),
            mfm_conv(64, 64, 1), nn.BatchNorm2d(64), mfm_conv(64, 32, 3, 1, 1), nn.BatchNorm2d(32),
            mfm_conv(32, 32, 1), nn.BatchNorm2d(32), mfm_conv(32, 32, 3, 1, 1), nn.MaxPool2d(2),
        )
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(32 * 2, 2)

    def forward(self, wav):
        f = self.lfcc(wav).unsqueeze(1)                        # B, 1, 60, frames
        f = (f - f.mean(-1, keepdim=True)) / (f.std(-1, keepdim=True) + 1e-5)   # per-utterance CMVN
        h = self.net(f)                                         # B, 32, F', T'
        h = h.mean(2)                                           # pool frequency
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)              # mean + max over time
        return self.out(self.drop(h))


# ----------------------------------------------------------------------------------- RawNet2
class FMS(nn.Module):
    """Filter-wise feature map scaling: x * s + s, s = sigmoid(FC(avgpool(x)))."""

    def __init__(self, c):
        super().__init__()
        self.fc = nn.Linear(c, c)

    def forward(self, x):
        s = torch.sigmoid(self.fc(x.mean(-1))).unsqueeze(-1)
        return x * s + s


class ResBlock1d(nn.Module):
    def __init__(self, cin, cout, first=False):
        super().__init__()
        self.first = first
        self.bn1 = nn.BatchNorm1d(cin) if not first else None
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else None
        self.pool = nn.MaxPool1d(3)
        self.fms = FMS(cout)

    def forward(self, x):
        out = x if self.first else F.leaky_relu(self.bn1(x), 0.3)
        out = self.conv2(F.leaky_relu(self.bn2(self.conv1(out)), 0.3))
        out = out + (self.down(x) if self.down is not None else x)
        return self.fms(self.pool(out))


class RawNet2(nn.Module):
    def __init__(self, sinc_filters=20, sinc_kernel=1024, gru=1024, fc=1024):
        super().__init__()
        self.sinc = SincConv(sinc_filters, sinc_kernel)
        self.bn0 = nn.BatchNorm1d(sinc_filters)
        c = sinc_filters
        self.blocks = nn.Sequential(ResBlock1d(c, c, first=True), ResBlock1d(c, c),
                                    ResBlock1d(c, 128), ResBlock1d(128, 128), ResBlock1d(128, 128), ResBlock1d(128, 128))
        self.bn_gru = nn.BatchNorm1d(128)
        self.gru = nn.GRU(128, gru, num_layers=3, batch_first=True)
        self.fc = nn.Linear(gru, fc)
        self.out = nn.Linear(fc, 2)

    def forward(self, wav):
        x = self.sinc(wav.unsqueeze(1))
        x = F.leaky_relu(self.bn0(F.max_pool1d(x.abs(), 3)), 0.3)
        x = self.blocks(x)
        x = F.leaky_relu(self.bn_gru(x), 0.3).transpose(1, 2)    # B, T, 128
        h, _ = self.gru(x)
        return self.out(self.fc(h[:, -1]))


# ----------------------------------------------------------------------------------- AASIST
class ResBlock2d(nn.Module):
    def __init__(self, cin, cout, first=False, official=False):
        super().__init__()
        self.first, self.official = first, official
        self.bn1 = nn.BatchNorm2d(cin) if not first else None
        self.conv1 = nn.Conv2d(cin, cout, (2, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, (2, 3), padding=(0, 1))
        self.down = nn.Conv2d(cin, cout, (1, 3), padding=(0, 1)) if cin != cout else None
        self.pool = nn.MaxPool2d((1, 3))

    def forward(self, x):
        out = x if self.first else F.selu(self.bn1(x))
        if self.official:   # the reference code applies conv1 to x, discarding the bn1/SELU output
            out = x
        out = self.conv2(F.selu(self.bn2(self.conv1(out))))
        out = out + (self.down(x) if self.down is not None else x)
        return self.pool(out)


def _bn_nodes(bn, x):
    s = x.shape
    return bn(x.reshape(-1, s[-1])).reshape(s)


class GraphAttention(nn.Module):
    """Graph attention over fully connected nodes (B, N, D):
    a_ij = softmax_j( w^T tanh(W (h_i * h_j)) / T ),  h_i' = SELU(BN(W1 sum_j a_ij h_j + W2 h_i))."""

    def __init__(self, din, dout, temperature=1.0):
        super().__init__()
        self.att_proj = nn.Linear(din, dout)
        self.att_w = nn.Parameter(torch.empty(dout, 1)); nn.init.xavier_normal_(self.att_w)
        self.proj_att = nn.Linear(din, dout)
        self.proj_self = nn.Linear(din, dout)
        self.bn = nn.BatchNorm1d(dout)
        self.drop = nn.Dropout(0.2)
        self.temp = temperature

    def forward(self, x):
        x = self.drop(x)
        pair = x.unsqueeze(2) * x.unsqueeze(1)                              # B, N, N, D
        a = torch.softmax(torch.tanh(self.att_proj(pair)) @ self.att_w / self.temp, dim=-2).squeeze(-1)
        x = self.proj_att(a @ x) + self.proj_self(x)
        return F.selu(_bn_nodes(self.bn, x))


class HeteroGraphAttention(nn.Module):
    """Heterogeneous graph attention over two node types (temporal, spectral) plus a master node.
    Separate attention weights for type1-type1, type2-type2 and cross-type edges; the master node attends
    over all nodes and summarises the graph."""

    def __init__(self, din, dout, temperature=1.0):
        super().__init__()
        self.proj_t1, self.proj_t2 = nn.Linear(din, din), nn.Linear(din, din)
        self.att_proj, self.att_projM = nn.Linear(din, dout), nn.Linear(din, dout)
        self.w11, self.w22, self.w12, self.wM = (nn.Parameter(torch.empty(dout, 1)) for _ in range(4))
        for w in (self.w11, self.w22, self.w12, self.wM):
            nn.init.xavier_normal_(w)
        self.proj_att, self.proj_self = nn.Linear(din, dout), nn.Linear(din, dout)
        self.proj_attM, self.proj_selfM = nn.Linear(din, dout), nn.Linear(din, dout)
        self.bn = nn.BatchNorm1d(dout)
        self.drop = nn.Dropout(0.2)
        self.temp = temperature

    def forward(self, x1, x2, master=None):
        n1 = x1.shape[1]
        x = torch.cat([self.proj_t1(x1), self.proj_t2(x2)], 1)
        if master is None:
            master = x.mean(1, keepdim=True)
        x = self.drop(x)
        # master update: attention of the master over all nodes
        am = torch.softmax(torch.tanh(self.att_projM(x * master)) @ self.wM / self.temp, dim=-2)   # B, N, 1
        master = self.proj_attM(am.transpose(1, 2) @ x) + self.proj_selfM(master)
        # node update with type-specific edge weights
        e = torch.tanh(self.att_proj(x.unsqueeze(2) * x.unsqueeze(1)))                            # B, N, N, D
        logits = torch.cat([torch.cat([e[:, :n1, :n1] @ self.w11, e[:, :n1, n1:] @ self.w12], 2),
                            torch.cat([e[:, n1:, :n1] @ self.w12, e[:, n1:, n1:] @ self.w22], 2)], 1)
        a = torch.softmax(logits / self.temp, dim=-2).squeeze(-1)
        x = F.selu(_bn_nodes(self.bn, self.proj_att(a @ x) + self.proj_self(x)))
        return x[:, :n1], x[:, n1:], master


class GraphPool(nn.Module):
    """Keep the top-k fraction of nodes by a learned sigmoid score; kept nodes are scaled by their score."""

    def __init__(self, ratio, dim, p=0.3):
        super().__init__()
        self.ratio = ratio
        self.proj = nn.Linear(dim, 1)
        self.drop = nn.Dropout(p)

    def forward(self, h):
        s = torch.sigmoid(self.proj(self.drop(h)))                 # B, N, 1
        k = max(int(h.shape[1] * self.ratio), 1)
        idx = s.topk(k, dim=1).indices.expand(-1, -1, h.shape[-1])
        return torch.gather(h * s, 1, idx)


AASIST_CFG = {
    "aasist": dict(filts=[70, [1, 32], [32, 32], [32, 64], [64, 64]], gat=[64, 32]),
    "aasist_l": dict(filts=[70, [1, 32], [32, 32], [32, 24], [24, 24]], gat=[24, 32]),
}


class AASIST(nn.Module):
    def __init__(self, variant: str = "aasist", pool=(0.5, 0.7, 0.5, 0.5), temps=(2.0, 2.0, 100.0, 100.0), official: bool = False):
        super().__init__()
        cfg = AASIST_CFG[variant]
        f, (g0, g1) = cfg["filts"], cfg["gat"]
        c = f[-1][-1]
        self.sinc = SincConv(f[0], 128)
        self.bn0 = nn.BatchNorm2d(1)
        o = official
        self.encoder = nn.Sequential(ResBlock2d(*f[1], first=True, official=o), ResBlock2d(*f[2], official=o), ResBlock2d(*f[3], official=o),
                                     ResBlock2d(*f[4], official=o), ResBlock2d(*f[4], official=o), ResBlock2d(*f[4], official=o))
        self.pos_S = nn.Parameter(torch.randn(1, 23, c))
        self.master1 = nn.Parameter(torch.randn(1, 1, g0))
        self.master2 = nn.Parameter(torch.randn(1, 1, g0))
        self.gat_S = GraphAttention(c, g0, temps[0])
        self.gat_T = GraphAttention(c, g0, temps[1])
        self.pool_S = GraphPool(pool[0], g0)
        self.pool_T = GraphPool(pool[1], g0)
        self.hs11, self.hs12 = HeteroGraphAttention(g0, g1, temps[2]), HeteroGraphAttention(g1, g1, temps[2])
        self.hs21, self.hs22 = HeteroGraphAttention(g0, g1, temps[2]), HeteroGraphAttention(g1, g1, temps[2])
        self.pool_hS1, self.pool_hT1 = GraphPool(pool[2], g1), GraphPool(pool[2], g1)
        self.pool_hS2, self.pool_hT2 = GraphPool(pool[2], g1), GraphPool(pool[2], g1)
        self.drop_way = nn.Dropout(0.2)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(5 * g1, 2)

    def _branch(self, out_T, out_S, master, l1, l2, pS, pT):
        t, s, m = l1(out_T, out_S, master=master)
        s, t = pS(s), pT(t)
        ta, sa, ma = l2(t, s, master=m)
        return t + ta, s + sa, m + ma

    def embed(self, wav):
        x = self.sinc(wav.unsqueeze(1)).unsqueeze(1)                 # B, 1, 70, T
        x = F.selu(self.bn0(F.max_pool2d(x.abs(), (3, 3))))          # B, 1, 23, T/3
        e = self.encoder(x)                                          # B, C, 23, T'
        e_S = e.abs().amax(3).transpose(1, 2) + self.pos_S           # spectral nodes: B, 23, C
        e_T = e.abs().amax(2).transpose(1, 2)                        # temporal nodes: B, T', C
        out_S = self.pool_S(self.gat_S(e_S))
        out_T = self.pool_T(self.gat_T(e_T))
        b = wav.shape[0]
        t1, s1, m1 = self._branch(out_T, out_S, self.master1.expand(b, -1, -1), self.hs11, self.hs12, self.pool_hS1, self.pool_hT1)
        t2, s2, m2 = self._branch(out_T, out_S, self.master2.expand(b, -1, -1), self.hs21, self.hs22, self.pool_hS2, self.pool_hT2)
        t = torch.max(self.drop_way(t1), self.drop_way(t2))
        s = torch.max(self.drop_way(s1), self.drop_way(s2))
        m = torch.max(self.drop_way(m1), self.drop_way(m2))
        return torch.cat([t.abs().amax(1), t.mean(1), s.abs().amax(1), s.mean(1), m.squeeze(1)], 1)

    def forward(self, wav):
        return self.out(self.drop(self.embed(wav)))


def build(name: str) -> nn.Module:
    if name == "lcnn":
        return LCNN()
    if name == "rawnet2":
        return RawNet2()
    if name in AASIST_CFG:
        return AASIST(name)
    raise ValueError(name)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def load_official_aasist(path, variant: str = "aasist") -> AASIST:
    """Load the authors' released weights (github.com/clovaai/aasist) into this implementation.
    Every official tensor must map to exactly one of ours with the same shape (strict), except the fixed
    sinc filters, which the reference recomputes at init and does not store."""
    import re
    sd = torch.load(path, map_location="cpu", weights_only=True)
    ren = [(r"^first_bn\.", "bn0."), (r"^encoder\.(\d+)\.0\.", r"encoder.\g<1>."), (r"conv_downsample", "down"),
           (r"^GAT_layer_S\.", "gat_S."), (r"^GAT_layer_T\.", "gat_T."), (r"^HtrgGAT_layer_ST(\d)(\d)\.", r"hs\g<1>\g<2>."),
           (r"^out_layer\.", "out."), (r"proj_type(\d)", r"proj_t\g<1>"), (r"att_weight(11|22|12|M)$", r"w\g<1>"),
           (r"att_weight$", "att_w"), (r"proj_with_attM", "proj_attM"), (r"proj_without_attM", "proj_selfM"),
           (r"proj_with_att", "proj_att"), (r"proj_without_att", "proj_self")]
    mapped = {}
    for k, v in sd.items():
        for a, b in ren:
            k = re.sub(a, b, k)
        mapped[k] = v
    m = AASIST(variant, official=True)
    missing, unexpected = m.load_state_dict(mapped, strict=False)
    assert not unexpected, unexpected
    assert set(missing) <= {"sinc.f1", "sinc.band", "sinc.n", "sinc.window"}, missing
    return m.eval()
