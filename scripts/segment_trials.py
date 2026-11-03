"""Split episodes into individual trials and label their phases.

Your hand stays in frame across several pick-and-places, so the hand-present
gaps used by split_episodes.py cannot separate them. What does separate them
is the grasp itself: the distance from the thumb tip to the finger tips is
small while you hold the cup and large when your hand is open, so every trial
shows up as a dip in that signal.

The dips give you the trial boundaries and the phase labels at the same time:

    reach      hand approaching, still open
    grasp      aperture closing onto the cup
    transport  holding, moving to the target
    release    aperture opening again
    retreat    hand moving away, open

These labels are a starting point, not ground truth. Check them against the
overlay or in Label Studio and fix what is wrong, the organisers care about
annotation quality.

    # look at the detected signal first
    python scripts/segment_trials.py data/episodes --plot outputs/aperture.png --dry-run

    # write trials and labels
    python scripts/segment_trials.py data/episodes data/trials
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WRIST = 0
THUMB_TIP = 4
GRASP_TIPS = [8, 12]      # index, middle: the fingers that oppose the thumb
INDEX_MCP = 5


def aperture_signal(world: np.ndarray) -> np.ndarray:
    """Thumb-to-fingertip distance, normalised by palm length.

    Dividing by palm length makes the signal independent of how far the hand
    was from the camera and of whose hand it is.
    """
    palm = np.linalg.norm(world[:, INDEX_MCP] - world[:, WRIST], axis=1)
    palm = np.where(palm < 1e-6, np.nan, palm)
    d = np.mean([np.linalg.norm(world[:, t] - world[:, THUMB_TIP], axis=1)
                 for t in GRASP_TIPS], axis=0)
    sig = d / palm
    # fill any nan from degenerate frames
    if np.isnan(sig).any():
        idx = np.arange(len(sig))
        good = ~np.isnan(sig)
        if good.any():
            sig = np.interp(idx, idx[good], sig[good])
        else:
            sig = np.zeros_like(sig)
    return sig


def smooth(x: np.ndarray, fps: float, cutoff_hz: float = 3.0) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    if len(x) < 12:
        return x
    wn = min(cutoff_hz / (fps / 2.0), 0.99)
    b, a = butter(2, wn, btype="low")
    return filtfilt(b, a, x)


def find_grasps(sig: np.ndarray, fps: float, min_hold_s: float = 0.4,
                lo_q: float = 25.0, hi_q: float = 75.0) -> list[tuple[int, int]]:
    """Closed-hand intervals, found with hysteresis on the aperture signal."""
    lo, hi = np.percentile(sig, lo_q), np.percentile(sig, hi_q)
    if hi - lo < 1e-6:
        return []
    close_t = lo + 0.35 * (hi - lo)
    open_t = lo + 0.65 * (hi - lo)

    closed, out, start = False, [], 0
    for i, v in enumerate(sig):
        if not closed and v < close_t:
            closed, start = True, i
        elif closed and v > open_t:
            closed = False
            out.append((start, i))
    if closed:
        out.append((start, len(sig)))

    min_f = int(min_hold_s * fps)
    return [(s, e) for s, e in out if (e - s) >= min_f]


def label_phases(n: int, grasps: list[tuple[int, int]], fps: float) -> list[str]:
    """One phase string per frame."""
    edge = max(1, int(0.15 * fps))
    phases = ["idle"] * n
    for k, (g0, g1) in enumerate(grasps):
        prev_end = grasps[k - 1][1] if k else 0
        for i in range(prev_end, min(g0, n)):
            phases[i] = "reach"
        for i in range(g0, min(g0 + edge, n)):
            phases[i] = "grasp"
        for i in range(min(g0 + edge, n), max(g0 + edge, g1 - edge)):
            if i < n:
                phases[i] = "transport"
        for i in range(max(0, g1 - edge), min(g1 + edge, n)):
            phases[i] = "release"
    if grasps:
        for i in range(min(grasps[-1][1] + edge, n), n):
            phases[i] = "retreat"
    return phases


def process(ep_path: Path, out_dir: Path | None, min_hold_s: float,
            write: bool) -> dict:
    ep = json.loads(ep_path.read_text())
    world = np.array(ep["world"])
    fps = ep.get("fps", 30.0)

    raw = aperture_signal(world)
    sig = smooth(raw, fps)
    grasps = find_grasps(sig, fps, min_hold_s=min_hold_s)
    phases = label_phases(len(sig), grasps, fps)

    print(f"  {ep_path.name}: {len(sig)} frames, {len(grasps)} grasp(s)")
    for k, (g0, g1) in enumerate(grasps, 1):
        print(f"      trial {k}: frames {g0}-{g1} "
              f"({(g1 - g0) / fps:.1f}s hold)")

    result = {
        "source_episode": ep_path.name,
        "fps": fps,
        "n_frames": len(sig),
        "aperture": sig.tolist(),
        "grasps": [[int(a), int(b)] for a, b in grasps],
        "phases": phases,
    }

    if write and out_dir is not None:
        (out_dir / f"{ep_path.stem}_phases.json").write_text(json.dumps(result))
        # one trial file per grasp, ready for retargeting
        for k, (g0, g1) in enumerate(grasps, 1):
            prev_end = grasps[k - 2][1] if k > 1 else 0
            s = prev_end
            e = min(len(sig), g1 + int(0.5 * fps))
            trial = {
                "source_episode": ep_path.name,
                "trial": k,
                "fps": fps,
                "n_frames": int(e - s),
                "grasp_frame": int(g0 - s),
                "release_frame": int(g1 - s),
                "world": world[s:e].tolist(),
                "image": np.array(ep["image"])[s:e].tolist(),
                "phases": phases[s:e],
            }
            name = f"{ep_path.stem}_t{k:02d}.json"
            (out_dir / name).write_text(json.dumps(trial))

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", help="episode JSON or folder")
    ap.add_argument("out_dir", nargs="?", default="data/trials")
    ap.add_argument("--min-hold", type=float, default=0.4,
                    help="shortest believable grasp, seconds")
    ap.add_argument("--plot", help="write a PNG of the aperture signals")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.episodes)
    paths = sorted(src.glob("ep_*.json")) if src.is_dir() else [src]
    paths = [p for p in paths if not p.stem.endswith("_phases")]
    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"segmenting {len(paths)} episode(s)")
    results = [process(p, out_dir, args.min_hold, not args.dry_run) for p in paths]

    total = sum(len(r["grasps"]) for r in results)
    per_ep = [len(r["grasps"]) for r in results]
    print(f"\ntotal trials found: {total}")
    print(f"per episode: min={min(per_ep)} max={max(per_ep)} "
          f"mean={np.mean(per_ep):.1f}")
    if total == 0:
        print("no grasps detected; check the aperture plot, the signal may be flat")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = min(6, len(results))
        fig, axes = plt.subplots(n, 1, figsize=(11, 2.0 * n), sharex=False)
        if n == 1:
            axes = [axes]
        for ax, r in zip(axes, results[:n]):
            sig = np.array(r["aperture"])
            t = np.arange(len(sig)) / r["fps"]
            ax.plot(t, sig, lw=1.2)
            for g0, g1 in r["grasps"]:
                ax.axvspan(g0 / r["fps"], g1 / r["fps"], alpha=0.25, color="tab:red")
            ax.set_title(f"{r['source_episode']}  ({len(r['grasps'])} grasps)",
                         fontsize=9)
            ax.set_ylabel("aperture")
        axes[-1].set_xlabel("seconds")
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=120)
        print(f"wrote {args.plot}  (red bands are detected grasps)")

    if args.dry_run:
        print("(dry run, nothing written)")
    else:
        print(f"wrote trials to {out_dir}")


if __name__ == "__main__":
    main()
