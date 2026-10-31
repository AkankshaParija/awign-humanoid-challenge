"""Overlay extracted keypoints back onto a clip, so you can eyeball the quality.

Usage:
    python scripts/preview_keypoints.py data/raw/clip_001.mp4 data/keypoints/clip_001.json out.mp4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# MediaPipe hand topology
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
FINGERTIPS = [4, 8, 12, 16, 20]


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    clip, kp_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    data = json.loads(Path(kp_path).read_text())
    image = np.array(data["image"])
    detected = data["detected"]

    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    i = 0
    while i < len(image):
        ok, frame = cap.read()
        if not ok:
            break
        pts = [(int(x * w), int(y * h)) for x, y, _ in image[i]]
        for a, b in EDGES:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for j, p in enumerate(pts):
            colour = (0, 0, 255) if j in FINGERTIPS else (255, 200, 0)
            cv2.circle(frame, p, 5, colour, -1)
        if not detected[i]:
            cv2.putText(frame, "INTERPOLATED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        writer.write(frame)
        i += 1

    cap.release()
    writer.release()
    found = sum(detected) / len(detected) * 100
    print(f"wrote {out_path}  (hand detected in {found:.0f}% of frames)")


if __name__ == "__main__":
    main()
