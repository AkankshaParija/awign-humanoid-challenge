"""README figures from outputs/retarget_summary.csv (63 trials).

  docs/figures/finger_error.png     per-finger fingertip error, mean +- std
  docs/figures/error_vs_length.png  trial length vs mean fingertip error,
                                    least-squares fit line and Pearson r

Prints every number it plots, so the README sentences can quote them.

    python scripts/plot_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "outputs/retarget_summary.csv"
OUT = ROOT / "docs/figures"

FINGERS = ["thumb", "index", "middle", "ring", "little"]

# one series per chart, so one hue; recessive ink for axes and text
BLUE = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    rows = list(csv.DictReader(CSV.open()))
    OUT.mkdir(parents=True, exist_ok=True)

    # --- 1. per-finger error, mean +- std across trials ---
    err = {f: np.array([float(r[f"{f}_mm"]) for r in rows]) for f in FINGERS}
    means = [err[f].mean() for f in FINGERS]
    stds = [err[f].std(ddof=1) for f in FINGERS]       # sample std across trials

    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    style(ax)
    ax.bar(FINGERS, means, width=0.6, color=BLUE,
           yerr=stds, capsize=4, error_kw={"ecolor": INK_2, "elinewidth": 1.2})
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 1.5, f"{m:.1f}", ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_ylabel("fingertip error (mm)", color=INK)
    ax.set_title(f"Fingertip error per finger, mean ± std over {len(rows)} trials",
                 color=INK, fontsize=11, loc="left")
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.15)
    fig.tight_layout()
    fig.savefig(OUT / "finger_error.png", dpi=150)
    plt.close(fig)

    # --- 2. trial length vs mean error ---
    length = np.array([int(r["frames"]) for r in rows])
    mean_err = np.array([float(r["mean_mm"]) for r in rows])
    slope, intercept = np.polyfit(length, mean_err, 1)
    r = np.corrcoef(length, mean_err)[0, 1]

    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.scatter(length, mean_err, s=36, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    xs = np.array([length.min(), length.max()])
    ax.plot(xs, slope * xs + intercept, color=INK_2, linewidth=2, zorder=2)
    ax.text(0.02, 0.96, f"r = {r:.2f}   fit: {slope:.2f} mm per frame",
            transform=ax.transAxes, va="top", color=INK, fontsize=10)
    ax.set_xlabel("trial length (frames)", color=INK)
    ax.set_ylabel("mean fingertip error (mm)", color=INK)
    ax.set_title(f"Longer trials have larger error ({len(rows)} trials)",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "error_vs_length.png", dpi=150)
    plt.close(fig)

    print(f"{len(rows)} trials")
    for f, m, s in zip(FINGERS, means, stds):
        print(f"  {f:7s} mean {m:5.1f} mm  std {s:5.1f} mm")
    print(f"  other four fingers, mean of means {np.mean(means[1:]):.1f} mm")
    print(f"length {length.min()}-{length.max()} frames, r = {r:.2f}, "
          f"slope {slope:.3f} mm/frame (x 30 frames = {slope * 30:.1f} mm per second), "
          f"intercept {intercept:.1f} mm")
    print(f"wrote {OUT / 'finger_error.png'} and {OUT / 'error_vs_length.png'}")


if __name__ == "__main__":
    main()
