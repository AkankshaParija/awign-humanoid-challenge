"""Temporally smooth the human hand landmarks before retargeting.

Job: the replayed hand motion is jittery. Tracking dropouts are already
linearly interpolated upstream (extract_hands.py, before episodes are even
split), so the remaining jitter is frame-to-frame MediaPipe noise on frames
that *were* detected. This smooths the 'world' landmark sequence per trial
with a Savitzky-Golay filter (per-coordinate, across time) and writes a
copy of each trial to a new folder, untouched otherwise: grasp_frame,
release_frame, phases and annotation are never modified, so nothing here
can leak into or change the hand-marked ground truth.

    python scripts/smooth_landmarks.py data/trials data/trials_smoothed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


def smooth_world(world: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    n = world.shape[0]
    if n < window:
        return world  # too short to filter meaningfully, leave as-is
    w = window if window % 2 == 1 else window + 1
    w = min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if w <= polyorder:
        return world
    return savgol_filter(world, window_length=w, polyorder=polyorder, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--polyorder", type=int, default=2)
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    files = sorted(args.src.glob("*.json"))
    if not files:
        raise SystemExit(f"no JSON in {args.src}")

    for f in files:
        d = json.loads(f.read_text())
        world = np.array(d["world"])
        d["world"] = smooth_world(world, args.window, args.polyorder).tolist()
        (args.dst / f.name).write_text(json.dumps(d))

    print(f"smoothed {len(files)} trials -> {args.dst} "
          f"(savgol window={args.window} polyorder={args.polyorder})")


if __name__ == "__main__":
    main()
