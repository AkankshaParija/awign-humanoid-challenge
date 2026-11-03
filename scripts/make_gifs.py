"""Convert rendered replay mp4s into small GIFs for the README.

No ffmpeg available, so frames are pulled with OpenCV and written out with
Pillow (both already in requirements.txt / the venv).

    python scripts/make_gifs.py outputs/videos/ep_007_t01.mp4 docs/gifs/ep_007_t01.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image


def mp4_to_gif(src: Path, dst: Path, stride: int = 2, max_width: int = 480,
              fps: float = 15.0) -> None:
    cap = cv2.VideoCapture(str(src))
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / w
                frame = cv2.resize(frame, (max_width, int(h * scale)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        i += 1
    cap.release()

    if not frames:
        raise SystemExit(f"no frames read from {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps) * stride
    frames[0].save(dst, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    print(f"wrote {dst} ({len(frames)} frames, {dst.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-width", type=int, default=480)
    args = ap.parse_args()
    mp4_to_gif(args.src, args.dst, stride=args.stride, max_width=args.max_width)


if __name__ == "__main__":
    main()
