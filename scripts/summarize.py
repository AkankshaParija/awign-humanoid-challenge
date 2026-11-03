"""Summarise the retargeting run: per-trial quality, and which trials to use.

Retargeting error varies a lot between trials, mostly with how well the hand
was tracked in that clip. Rather than treat all trials as equal, rank them and
take the best ones for the headline results, saying so in the README.

    python scripts/summarize.py outputs/qpos
    python scripts/summarize.py outputs/qpos --top 30 --write outputs/best_trials.txt
    python scripts/summarize.py outputs/qpos --csv outputs/retarget_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "little"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("qpos_dir", nargs="?", default="outputs/qpos")
    ap.add_argument("--top", type=int, default=0,
                    help="list only the N best trials")
    ap.add_argument("--write", help="write the chosen trial names to a file")
    ap.add_argument("--csv", help="write the full per-trial table")
    args = ap.parse_args()

    files = sorted(Path(args.qpos_dir).glob("*.json"))
    if not files:
        raise SystemExit(f"no JSON in {args.qpos_dir}")

    rows = []
    for f in files:
        d = json.loads(f.read_text())
        err = np.array(d["tip_error_m"])
        pf = d.get("per_finger_error_m", {})
        rows.append({
            "trial": f.stem,
            "frames": len(err),
            "mean_mm": float(err.mean() * 1000),
            "median_mm": float(np.median(err) * 1000),
            "p90_mm": float(np.percentile(err, 90) * 1000),
            "palm_resid_mm": float(d.get("palm_fit_residual_m", 0) * 1000),
            **{f"{k}_mm": float(v * 1000) for k, v in pf.items()},
        })

    rows.sort(key=lambda r: r["mean_mm"])
    allm = np.array([r["mean_mm"] for r in rows])

    print(f"{len(rows)} trials, {sum(r['frames'] for r in rows)} frames total\n")
    print("retargeting error (mm):")
    print(f"  best {allm.min():.1f}   median {np.median(allm):.1f}   "
          f"mean {allm.mean():.1f}   worst {allm.max():.1f}")
    for thr in (25, 35, 50):
        n = int((allm <= thr).sum())
        print(f"  <= {thr}mm : {n:3d} trials ({n / len(allm) * 100:.0f}%)")

    pf_cols = [f"{f}_mm" for f in FINGERS if f"{f}_mm" in rows[0]]
    if pf_cols:
        print("\nmean error by finger (mm):")
        for c in pf_cols:
            vals = np.array([r[c] for r in rows])
            print(f"  {c[:-3]:<8} {vals.mean():6.1f}")

    show = rows[:args.top] if args.top else rows
    print(f"\n{'trial':<22}{'frames':>7}{'mean':>8}{'p90':>8}{'palm':>8}")
    for r in show:
        print(f"{r['trial']:<22}{r['frames']:>7}{r['mean_mm']:>8.1f}"
              f"{r['p90_mm']:>8.1f}{r['palm_resid_mm']:>8.1f}")

    if args.write:
        p = Path(args.write)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(r["trial"] for r in show) + "\n")
        print(f"\nwrote {len(show)} trial names to {p}")

    if args.csv:
        p = Path(args.csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
