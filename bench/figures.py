"""Figures from committed CSVs. Nothing is re-run or re-measured here."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
COL = {"pixel DDPM": "#b2182b", "LDM f=2": "#92c5de",
       "LDM f=4": "#2166ac", "LDM f=8": "#1a9850"}


def fig_rate_distortion(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "stage1.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    g = t.groupby("compression")
    for ax, col, ylabel, title in (
            (a, "psnr", "reconstruction PSNR (dB)", "Distortion against rate"),
            (b, "rfid", "rFID (the floor stage two cannot beat)", "Reconstruction cFID")):
        med, lo, hi = g[col].median(), g[col].min(), g[col].max()
        ax.plot(med.index, med.values, marker="o", color="#762a83", linewidth=2)
        ax.fill_between(med.index, lo.values, hi.values, color="#762a83", alpha=0.18, linewidth=0)
        for x, y, f in zip(med.index, med.values, g["f"].median().values):
            ax.annotate(f"f={int(f)}", (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("compression ratio (pixels per latent element)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Stage one: what the autoencoder costs you, median and range over 3 seeds",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_quality_vs_cost(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "stage2.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for model, colour in COL.items():
        s = t[t["model"] == model]
        if s.empty:
            continue
        g = s.groupby("nfe")
        med, lo, hi = g["cfid"].median(), g["cfid"].min(), g["cfid"].max()
        a.plot(med.index, med.values, marker="o", color=colour, label=model, linewidth=1.9)
        a.fill_between(med.index, lo.values, hi.values, color=colour, alpha=0.15, linewidth=0)
        at50 = s[s["nfe"] == s["nfe"].max()]
        b.scatter(at50["train_s"], at50["cfid"], color=colour, s=95, label=model, zorder=3)
    a.set_xscale("log", base=2); a.set_yscale("log")
    a.set_xlabel("NFE (denoising steps)"); a.set_ylabel("cFID (lower is better)")
    a.set_title("Quality against sampling budget")
    a.grid(alpha=0.3, which="both"); a.legend(frameon=False, fontsize=9)

    b.set_xlabel("training wall clock (s, CPU)"); b.set_ylabel("cFID at the largest NFE")
    b.set_yscale("log")
    b.set_title("Quality against training cost\ndown and to the left is better")
    b.grid(alpha=0.3, which="both"); b.legend(frameon=False, fontsize=9)
    fig.suptitle("Stage two: identical UNet, schedule and sampler in every space", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_cost_breakdown(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "stage2.csv")
    s = t[t["nfe"] == t["nfe"].max()]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    models = [m for m in COL if m in set(s["model"])]
    for ax, col, ylabel, title in (
            (a, "train_s", "seconds", "Training time, matched step count"),
            (b, "latent_elems", "elements per sample",
             "What the diffusion model actually operates on")):
        vals = [s[s["model"] == m][col].median() for m in models]
        bars = ax.bar(range(len(models)), vals, color=[COL[m] for m in models])
        base = vals[0]
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v * 1.03,
                    f"{v:.0f}\n{base / v:.1f}x" if v else "0", ha="center", fontsize=8.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.set_yscale("log")
        ax.grid(alpha=0.3, axis="y", which="both")
    fig.suptitle("Where the saving comes from", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for p in (fig_rate_distortion(RESULTS / "rate-distortion.png"),
              fig_quality_vs_cost(RESULTS / "quality-vs-cost.png"),
              fig_cost_breakdown(RESULTS / "cost-breakdown.png")):
        print(f"-> {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
