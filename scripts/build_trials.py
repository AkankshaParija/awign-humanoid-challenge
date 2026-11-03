"""Turn hand-marked grasp/release frames into per-trial datasets.

Reads the CSV written by annotate.py and the episode keypoints, and emits one
JSON per trial with its keypoints and a per-frame phase label.

    python scripts/build_trials.py data/annotations.csv data/episodes data/trials

Phases, derived from the two marked frames:

    reach      start of the trial window up to the grasp
    grasp      a short window at the grasp frame
    transport  between grasp and release
    release    a short window at the release frame
    retreat    after the release

A trial window starts a little before the grasp and ends a little after the
release; --lead and --tail control how much.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def phases_for(n: int, grasp: int, release: int, fps: float) -> list[str]:
    edge = max(1, int(0.15 * fps))
    out = ["reach"] * n
    for i in range(max(0, grasp - edge // 2), min(n, grasp + edge)):
        out[i] = "grasp"
    for i in range(min(n, grasp + edge), max(0, release - edge)):
        out[i] = "transport"
    for i in range(max(0, release - edge), min(n, release + edge)):
        out[i] = "release"
    for i in range(min(n, release + edge), n):
        out[i] = "retreat"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", nargs="?", default="data/annotations.csv")
    ap.add_argument("episodes_dir", nargs="?", default="data/episodes")
    ap.add_argument("out_dir", nargs="?", default="data/trials")
    ap.add_argument("--lead", type=float, default=0.6,
                    help="seconds kept before the grasp")
    ap.add_argument("--tail", type=float, default=0.4,
                    help="seconds kept after the release")
    ap.add_argument("--min-frames", type=int, default=20)
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.csv_path).open()))
    if not rows:
        raise SystemExit(f"{args.csv_path} is empty")

    by_clip: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append(
            (int(r["grasp_frame"]), int(r["release_frame"])))

    eps_dir = Path(args.episodes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, skipped = 0, []
    all_phases: Counter = Counter()
    directions: Counter = Counter()

    for clip_name in sorted(by_clip):
        kp_path = eps_dir / (Path(clip_name).stem + ".json")
        if not kp_path.exists():
            skipped.append(f"{clip_name}: no keypoints at {kp_path.name}")
            continue

        ep = json.loads(kp_path.read_text())
        world = np.array(ep["world"])
        image = np.array(ep["image"])
        fps = ep.get("fps", 30.0)
        lead, tail = int(args.lead * fps), int(args.tail * fps)

        for t, (g, r) in enumerate(sorted(by_clip[clip_name]), 1):
            s = max(0, g - lead)
            e = min(len(world), r + tail)
            if e - s < args.min_frames:
                skipped.append(f"{clip_name} trial {t}: only {e - s} frames")
                continue
            if r >= len(world):
                skipped.append(f"{clip_name} trial {t}: release {r} past end "
                               f"({len(world)} frames)")
                continue

            ph = phases_for(e - s, g - s, r - s, fps)
            all_phases.update(ph)

            # Which way did the cup travel? The image-space wrist x position
            # at grasp vs at release says so, and the camera never moved, so
            # the sign is consistent across the whole session.
            dx = float(image[r, 0, 0] - image[g, 0, 0])
            direction = ("to_target" if dx > 0 else "to_start") \
                if abs(dx) > 0.03 else "unclear"
            directions.update([direction])

            trial = {
                "source_episode": kp_path.name,
                "source_clip": clip_name,
                "trial": t,
                "fps": fps,
                "n_frames": int(e - s),
                "grasp_frame": int(g - s),
                "release_frame": int(r - s),
                "world": world[s:e].tolist(),
                "image": image[s:e].tolist(),
                "phases": ph,
                "annotation": "manual",
                "direction": direction,
                "wrist_dx_image": dx,
            }
            name = f"{Path(clip_name).stem}_t{t:02d}.json"
            (out_dir / name).write_text(json.dumps(trial))
            written += 1

    print(f"wrote {written} trials to {out_dir}")
    if all_phases:
        total = sum(all_phases.values())
        print("\nframes per phase:")
        for ph in ["reach", "grasp", "transport", "release", "retreat"]:
            c = all_phases.get(ph, 0)
            print(f"  {ph:<10} {c:6d}  ({c / total * 100:4.1f}%)")
    if directions:
        print("\ntrials by direction (from wrist travel in the image):")
        for d, c in directions.most_common():
            print(f"  {d:<10} {c:4d}")
        if directions.get("unclear"):
            print("  'unclear' means the wrist barely moved sideways; check those")

    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for s in skipped[:12]:
            print(f"  {s}")


if __name__ == "__main__":
    main()
