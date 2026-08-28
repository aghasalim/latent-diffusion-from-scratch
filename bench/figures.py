"""Figures from the committed CSVs. Nothing here re-measures anything.

The one exception is the denoising animation. A trajectory is not in a table, so
that function retrains the f=4 pair with the settings recorded in
results/run-meta.json at seed 0 and samples with the repo's own DDIM loop. It
caches the frames and it never writes a CSV.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from PIL import Image

from bench.style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Red stays on the pixel baseline, which is the thing being beaten in every
# panel. The other three come from the shared palette in order of f.
COL = {"pixel DDPM": PALETTE[1], "LDM f=2": PALETTE[0],
       "LDM f=4": PALETTE[2], "LDM f=8": PALETTE[3]}


def fig_rate_distortion(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "stage1.csv")
    g = t.groupby("compression")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.7))

    for ax, col in ((a, "psnr"), (b, "rfid")):
        med, lo, hi = g[col].median(), g[col].min(), g[col].max()
        ax.vlines(med.index, lo.values, hi.values, color=PALETTE[0], alpha=0.3,
                  linewidth=7, zorder=1)
        ax.scatter(t["compression"], t[col], s=20, color=PALETTE[0], alpha=0.55,
                   zorder=2)
        ax.plot(med.index, med.values, marker="o", color=PALETTE[0], zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.6, 27)
        ax.set_xticks([1, 4, 16])
        ax.set_xticklabels(["1x\nf=2", "4x\nf=4", "16x\nf=8"])
        ax.set_xlabel("compression (pixels per latent element)")

    a.set_ylabel("reconstruction PSNR (dB)")
    titled(a, "Each step of compression costs 5 to 6 dB",
           "Bars span the 3 seeds. None sits more than 0.8 dB off its median.")

    f2 = t[t["f"] == 2]["rfid"]
    b.axhspan(f2.min(), f2.max(), color=PALETTE[5], alpha=0.16, zorder=0)
    b.text(25.5, (f2.min() * f2.max()) ** 0.5, "f=2 range, extended", ha="right",
           va="center", fontsize=9, color="#5a5a5a")
    b.set_yscale("log")
    b.set_yticks([0.02, 0.05, 0.1, 0.2])
    b.set_yticklabels(["0.02", "0.05", "0.10", "0.20"])
    b.set_ylim(0.014, 0.30)
    b.minorticks_off()
    b.set_ylabel("rFID (lower is better)")
    titled(b, "The floor only clearly rises at f=8",
           "The f=4 range straddles f=2, so those two are not separated.")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_quality_vs_cost(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "stage2.csv")
    at50 = t[t["nfe"] == t["nfe"].max()]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.2, 4.9))

    for model, colour in COL.items():
        s = t[t["model"] == model]
        med = s.groupby("nfe")["cfid"].median()
        a.scatter(s["nfe"], s["cfid"], s=15, color=colour, alpha=0.3, zorder=2)
        a.plot(med.index, med.values, marker="o", color=colour, zorder=3)
        a.annotate(model, (med.index[-1], med.values[-1]), xytext=(9, 0),
                   textcoords="offset points", va="center", color=colour,
                   fontsize=9.5, fontweight="semibold")

        p = at50[at50["model"] == model]
        x, y = p["train_s"].median(), p["cfid"].median()
        b.scatter(p["train_s"], p["cfid"], s=15, color=colour, alpha=0.3, zorder=2)
        b.scatter([x], [y], s=95, color=colour, zorder=3)
        right = x > 300
        b.annotate(model, (x, y), xytext=(-10 if right else 10, 9),
                   textcoords="offset points", ha="right" if right else "left",
                   color=colour, fontsize=9.5, fontweight="semibold")

    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xlim(8.5, 135)
    a.set_xticks([10, 25, 50])
    a.set_xticklabels(["10", "25", "50"])
    a.set_yticks([10, 20, 50, 100, 200])
    a.set_yticklabels(["10", "20", "50", "100", "200"])
    a.minorticks_off()
    a.set_xlabel("NFE (model calls per sample)")
    a.set_ylabel("cFID (lower is better)")
    titled(a, "Latent diffusion wins at every step budget",
           "Same UNet, schedule and sampler everywhere. Dots are single seeds.")

    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(30, 1300)
    b.set_ylim(8, 260)
    b.set_xticks([50, 100, 200, 500, 1000])
    b.set_xticklabels(["50", "100", "200", "500", "1000"])
    b.set_yticks([10, 20, 50, 100, 200])
    b.set_yticklabels(["10", "20", "50", "100", "200"])
    b.minorticks_off()
    b.set_xlabel("training wall clock (seconds, CPU)")
    b.set_ylabel("cFID at 50 NFE (lower is better)")
    titled(b, "f=4 buys the most quality per second spent",
           "Median of 3 seeds at 50 NFE. Down and to the left is better.")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_cost_breakdown(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "stage2.csv")
    s = t[t["nfe"] == t["nfe"].max()]
    models = [m for m in COL if m in set(s["model"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.7))

    for ax, col in ((a, "train_s"), (b, "latent_elems")):
        vals = [s[s["model"] == m][col].median() for m in models]
        ax.bar(range(len(models)), vals, width=0.62,
               color=[COL[m] for m in models])
        for i, v in enumerate(vals):
            ratio = vals[0] / v
            label = f"{v:.0f}\n{ratio:.1f}x less" if ratio > 1.05 else f"{v:.0f}"
            ax.annotate(label, (i, v), xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9.2)
        ax.set_ylim(top=max(vals) * 1.28)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.grid(axis="x", visible=False)

    a.set_ylabel("training wall clock (seconds, CPU)")
    titled(a, "Spatial size sets the cost, not element count",
           "f=2 holds 1024 elements like pixels and still trains 2.9x faster")

    b.set_ylabel("elements per sample (count)")
    titled(b, "Only f=4 and f=8 shrink the problem",
           "What the UNet operates on. f=2 is the same size as pixel space.")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# --- the animation ----------------------------------------------------------
GIF_F, GIF_SEED, GIF_NFE, GIF_N, GIF_FPS = 4, 0, 50, 16, 15
GIF_DPI, GIF_GREYS, GIF_HOLD = 96, 32, 15
CACHE = RESULTS / "denoise-frames.npz"


def _denoise_frames() -> np.ndarray:
    """Decoded DDIM trajectory for the f=4 model, shape (steps + 1, N, 32, 32).

    Retraining is about six minutes on a laptop CPU, so the frames are cached.
    The cache is gitignored because it is derived, not measured. Settings come
    from results/run-meta.json so they cannot drift from the documented run.
    """
    if CACHE.exists():
        return np.load(CACHE)["frames"]

    import torch

    from experiments.main import train_autoencoder, train_diffusion
    from ldm.data import load

    meta = json.loads((RESULTS / "run-meta.json").read_text())
    args = SimpleNamespace(**meta)
    train, val = load(meta["n"])
    ae, _, _ = train_autoencoder(GIF_F, train, val, args, GIF_SEED)
    net, diff, scale, _, shape = train_diffusion("latent", ae, train, args, GIF_SEED)

    @torch.no_grad()
    def decode(z):
        img = ae.decode(z / scale).clamp(-1, 1)
        return ((img[:, 0] + 1) * 127.5).round().to(torch.uint8).numpy()

    # The same seed is drawn twice on purpose: once to show the noise the
    # sampler starts from, once inside the sampler itself.
    noise = torch.randn((GIF_N, *shape),
                        generator=torch.Generator().manual_seed(GIF_SEED + 9))
    frames = [decode(noise)]
    diff.ddim_sample(net, (GIF_N, *shape), nfe=GIF_NFE,
                     generator=torch.Generator().manual_seed(GIF_SEED + 9),
                     callback=lambda i, t, x0, x: frames.append(decode(x)))

    out = np.stack(frames)
    np.savez_compressed(CACHE, frames=out)
    return out


def _tile(frame: np.ndarray, gap: int = 2) -> np.ndarray:
    """N square images into one square grid with white gutters."""
    side = round(len(frame) ** 0.5)
    p = np.pad(frame, ((0, 0), (gap, gap), (gap, gap)), constant_values=255)
    return np.vstack([np.hstack(list(p[r * side:(r + 1) * side]))
                      for r in range(side)])


def _gif_palette() -> Image.Image:
    """One fixed palette for every frame: a grey ramp plus the progress bar.

    Nothing in the figure is coloured except that bar, so 32 grey levels hold
    it. The mean error against the full colour render is under 1 of 255, which
    is not visible, and the short palette is most of the size saving. Sharing
    one palette across frames also lets the writer store only what changed.
    """
    green = np.array(to_rgb(PALETTE[2])) * 255
    ramp = [[v] * 3 for v in np.linspace(0, 255, GIF_GREYS).round()]
    # the bar is drawn on white, so its antialiased edge needs the blend too
    blend = [(255 + (green - 255) * t).round() for t in np.linspace(0, 1, 8)[1:]]
    pal = Image.new("P", (1, 1))
    pal.putpalette(np.concatenate(ramp + blend).astype(np.uint8).tobytes())
    return pal


def fig_denoising(out: Path) -> Path:
    tiles = [_tile(f) for f in _denoise_frames()]

    fig, (ax, axp) = plt.subplots(
        2, 1, figsize=(4.5, 5.05), dpi=GIF_DPI,
        gridspec_kw={"height_ratios": [1, 0.05], "hspace": 0.16})
    im = ax.imshow(tiles[0], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    titled(ax, "From noise to digits in 50 steps",
           "LDM f=4, seed 0, latent decoded to pixels each step")

    axp.set_xlim(0, GIF_NFE)
    axp.set_ylim(0, 1)
    axp.set_yticks([])
    axp.set_xticks([0, 10, 20, 30, 40, 50])
    axp.grid(False)
    axp.set_xlabel("DDIM step")
    bar = axp.barh(0.5, 0, height=1.0, color=PALETTE[2], align="center")[0]

    pal = _gif_palette()
    frames = []
    for i in range(len(tiles)):
        im.set_data(tiles[i])
        bar.set_width(i)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(Image.fromarray(rgb).quantize(palette=pal,
                                                    dither=Image.Dither.NONE))
    plt.close(fig)

    # Writing the frames here rather than through matplotlib's PillowWriter is
    # the point of the exercise. That writer picks a palette per frame and
    # dithers it, which puts fresh noise in every pixel of every frame and
    # left nothing for the compressor to find. It made a 1.8 MB file.
    step = round(1000 / GIF_FPS)
    frames[0].save(out, save_all=True, append_images=frames[1:], loop=0,
                   duration=[step] * (len(frames) - 1) + [step * GIF_HOLD],
                   optimize=True)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for fn, name in ((fig_rate_distortion, "rate-distortion.png"),
                     (fig_quality_vs_cost, "quality-vs-cost.png"),
                     (fig_cost_breakdown, "cost-breakdown.png"),
                     (fig_denoising, "denoising.gif")):
        p = fn(RESULTS / name)
        print(f"-> {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
