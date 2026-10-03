"""Monochrome 3D renders for the deck (the reference style uses greyscale 3D abstract shapes).
Each render is a real speech spectrogram turned into a lit, glossy greyscale surface.

  python presentation/make_renders.py  ->  assets/figures/render_{wave,ribbon,orb}.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from voiceshield.data import load_wave  # noqa: E402

FIG = ROOT / "assets" / "figures"


def spectrogram(x, n=512, hop=128):
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
    s = np.abs(np.fft.rfft(frames, axis=1)).T
    return np.log1p(30 * s / s.max())


def surface(z, out, elev, azim, size, bg="#111111", cmap="Greys_r", zscale=1.0, crop=(0.06, 0.94)):
    z = gaussian_filter(z, 2.2)
    z = (z - z.min()) / (z.max() - z.min()) * zscale
    y, x = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    ls = LightSource(azdeg=315, altdeg=35)
    rgb = ls.shade(z, cmap=plt.get_cmap(cmap), vert_exag=40, blend_mode="soft")
    fig = plt.figure(figsize=size, dpi=200, facecolor=bg)
    ax = fig.add_subplot(projection="3d", facecolor=bg)
    ax.plot_surface(x, y, z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((z.shape[1] / z.shape[0], 1, 0.25))
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out, facecolor=bg)
    plt.close(fig)
    autocrop(out, bg)


def autocrop(path, bg, pad=0.04):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(int)
    ref = np.array([int(bg[i:i + 2], 16) for i in (1, 3, 5)])
    mask = np.abs(a - ref).sum(2) > 18
    ys, xs = np.where(mask)
    if len(xs):
        h, w = mask.shape
        py, px = int(h * pad), int(w * pad)
        im.crop((max(xs.min() - px, 0), max(ys.min() - py, 0), min(xs.max() + px, w), min(ys.max() + py, h))).save(path)


def blob(s, out, size=(8, 8), bg="#111111", light=(-0.5, 0.6, 0.7), bumps=0.22):
    """A sphere whose radius is displaced by the speech spectrogram, shaded like polished chrome."""
    u, v = np.mgrid[0:2 * np.pi:260j, 0:np.pi:180j]
    d = gaussian_filter(np.resize(s, u.shape), (3, 2))
    d = (d - d.min()) / (d.max() - d.min())
    r = 1 + bumps * d * np.sin(v) ** 2
    x, y, z = r * np.cos(u) * np.sin(v), r * np.sin(u) * np.sin(v), r * np.cos(v) * 0.9
    gx = np.stack(np.gradient(np.stack([x, y, z]), axis=1)); gy = np.stack(np.gradient(np.stack([x, y, z]), axis=2))
    n = np.cross(gx.reshape(3, -1).T, gy.reshape(3, -1).T); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    L = np.array(light) / np.linalg.norm(light)
    diff = np.clip(n @ L, 0, 1)
    spec = np.clip(n @ (L + np.array([0, 0, 1])) / np.linalg.norm(L + np.array([0, 0, 1])), 0, 1) ** 40
    shade = np.clip(0.08 + 0.75 * diff + 0.9 * spec, 0, 1).reshape(u.shape)
    fig = plt.figure(figsize=size, dpi=200, facecolor=bg)
    ax = fig.add_subplot(projection="3d", facecolor=bg)
    ax.plot_surface(x, y, z, facecolors=plt.cm.gray(shade), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.view_init(elev=18, azim=-60); ax.set_box_aspect((1, 1, 0.9)); ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1); fig.savefig(out, facecolor=bg); plt.close(fig)
    autocrop(out, bg)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    clips = sorted((ROOT / "data" / "LA" / "ASVspoof2019_LA_eval" / "flac").glob("*.flac"))
    x = np.concatenate([load_wave(c) for c in clips[:3]])[:16000 * 4]
    s = spectrogram(x)[:140]                 # low/mid frequencies carry the shape
    surface(s[:, ::2], FIG / "render_wave.png", elev=38, azim=-62, size=(10, 5))
    surface(s[::-1, ::3].T[:, :120], FIG / "render_ribbon.png", elev=48, azim=-35, size=(8, 8), bg="#e9e9e6", cmap="gray", zscale=1.6)
    print("renders:", sorted(p.name for p in FIG.glob("render_*.png")))


if __name__ == "__main__":
    main()
