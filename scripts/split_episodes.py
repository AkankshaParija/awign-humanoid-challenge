"""Split one long continuous recording into per-trial episodes.

Assumes you filmed all repetitions in a single take, taking your hand fully
out of frame for a beat between trials. Those hand-free gaps are the episode
boundaries.

Run extract_hands.py on the long video first, then:

    python scripts/split_episodes.py data/keypoints/session_01.json data/episodes
    python scripts/split_episodes.py data/keypoints/session_01.json data/episodes \
        --video data/raw/session_01.mp4 --write-clips

Tune --min-gap / --min-episode if the split looks wrong; --dry-run just prints.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs for each run of True in mask."""
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("keypoints", help="JSON from extract_hands.py for the long take")
    ap.add_argument("out_dir", nargs="?", default="data/episodes")
    ap.add_argument("--min-gap", type=float, default=1.0,
                    help="seconds of no-hand that counts as a boundary")
    ap.add_argument("--min-episode", type=float, default=1.5,
                    help="discard episodes shorter than this many seconds")
    ap.add_argument("--pad", type=float, default=0.2,
                    help="seconds of context to keep either side")
    ap.add_argument("--video", help="the long video, needed for --write-clips")
    ap.add_argument("--write-clips", action="store_true",
                    help="also cut mp4 files with ffmpeg")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.keypoints).read_text())
    fps = data["fps"]
    detected = np.array(data["detected"], dtype=bool)
    world = np.array(data["world"])
    image = np.array(data["image"])

    min_gap_f = int(args.min_gap * fps)
    min_ep_f = int(args.min_episode * fps)
    pad_f = int(args.pad * fps)

    # Close short dropouts so a single missed frame does not split an episode.
    filled = detected.copy()
    for s, e in find_runs(~detected):
        if (e - s) < min_gap_f:
            filled[s:e] = True

    episodes = [(s, e) for s, e in find_runs(filled) if (e - s) >= min_ep_f]

    total_s = len(detected) / fps
    print(f"{Path(args.keypoints).name}: {len(detected)} frames, {total_s:.1f}s @ {fps:.0f}fps")
    print(f"hand present in {detected.mean() * 100:.0f}% of frames")
    print(f"found {len(episodes)} episodes\n")

    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, (s, e) in enumerate(episodes, start=1):
        s_pad = max(0, s - pad_f)
        e_pad = min(len(detected), e + pad_f)
        dur = (e_pad - s_pad) / fps
        t0 = s_pad / fps
        print(f"  ep_{i:03d}  t={t0:6.2f}s  dur={dur:5.2f}s  frames={e_pad - s_pad}")

        if args.dry_run:
            continue

        ep = {
            "episode": i,
            "source": data["clip"],
            "fps": fps,
            "start_frame": int(s_pad),
            "end_frame": int(e_pad),
            "start_time": float(t0),
            "n_frames": int(e_pad - s_pad),
            "world": world[s_pad:e_pad].tolist(),
            "image": image[s_pad:e_pad].tolist(),
            "detected": detected[s_pad:e_pad].tolist(),
        }
        (out_dir / f"ep_{i:03d}.json").write_text(json.dumps(ep))

        if args.write_clips:
            if not args.video:
                raise SystemExit("--write-clips needs --video")
            dst = out_dir / f"ep_{i:03d}.mp4"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}",
                 "-i", args.video, "-t", f"{dur:.3f}",
                 "-c:v", "libx264", "-preset", "veryfast", "-an", str(dst)],
                check=True,
            )

    if args.dry_run:
        print("\n(dry run, nothing written)")
    else:
        print(f"\nwrote {len(episodes)} episodes to {out_dir}")
        if len(episodes) < 20:
            print("fewer episodes than expected? try lowering --min-gap")


if __name__ == "__main__":
    main()
