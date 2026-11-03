"""Measure jitter in retargeted qpos trajectories: mean frame-to-frame joint
velocity and acceleration, in radians/frame and radians/frame^2.

    python scripts/jitter.py outputs/qpos
    python scripts/jitter.py outputs/retarget_smoothed/qpos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("qpos_dir", type=Path)
    args = ap.parse_args()

    files = sorted(args.qpos_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no JSON in {args.qpos_dir}")

    vel_means, acc_means = [], []
    for f in files:
        q = np.array(json.loads(f.read_text())["qpos"])  # (n_frames, n_joints)
        if len(q) < 3:
            continue
        vel = np.diff(q, axis=0)
        acc = np.diff(vel, axis=0)
        vel_means.append(np.mean(np.abs(vel)))
        acc_means.append(np.mean(np.abs(acc)))

    print(f"{len(files)} trials")
    print(f"mean |joint velocity|     {np.mean(vel_means):.5f} rad/frame")
    print(f"mean |joint acceleration| {np.mean(acc_means):.5f} rad/frame^2")


if __name__ == "__main__":
    main()
