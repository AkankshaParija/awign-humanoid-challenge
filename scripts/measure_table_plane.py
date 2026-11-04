"""Measure the phone's tilt from the sticky note (a known 7.5 cm square on
the table), for wrist_traj.CAM_TO_SIM.

The camera never moves during the session. In several clips, the pink note
is found by colour, its 4 corners by polygon fit, and cv2.solvePnP
(IPPE_SQUARE, same assumed focal length as everywhere else) gives the
table plane's normal in camera axes. Clips where the note outline isn't a
clean quadrilateral (occluded) are skipped.

    python scripts/measure_table_plane.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import wrist_traj as W  # noqa: E402
from config import STICKY_NOTE_SIDE  # noqa: E402

CLIPS = ["ep_047.mp4", "ep_008.mp4", "ep_010.mp4", "ep_011.mp4", "ep_013.mp4", "ep_021.mp4", "ep_034.mp4"]
EPISODES = Path(__file__).resolve().parent.parent / "data/episodes"


def table_normal(frame: np.ndarray) -> tuple[np.ndarray, float] | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (150, 40, 120), (179, 255, 255)) | cv2.inRange(hsv, (0, 40, 120), (8, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    quad = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True).reshape(-1, 2).astype(np.float64)
    if len(quad) != 4:
        return None
    centre = quad.mean(axis=0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0]))]
    K = np.array([[W.FOCAL_PX, 0, W.IMAGE_W / 2], [0, W.FOCAL_PX, W.IMAGE_H / 2], [0, 0, 1.0]])
    h = STICKY_NOTE_SIDE / 2
    best = None
    # try both windings and all 4 starting corners; keep the best reprojection
    for square in (np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]]),
                   np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])):
        for k in range(4):
            obj = np.roll(square, k, axis=0)
            _, rv, tv = cv2.solvePnP(obj, quad, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
            err = float(np.linalg.norm(proj[:, 0] - quad, axis=1).mean())
            if tv[2, 0] > 0 and (best is None or err < best[0]):
                best = (err, rv)
    err, rv = best
    n = cv2.Rodrigues(rv)[0][:, 2]
    return (-n if n[1] > 0 else n), err               # normal points up (image -y)


def main() -> None:
    normals = []
    for clip in CLIPS:
        cap = cv2.VideoCapture(str(EPISODES / clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
        ok, frame = cap.read()
        out = table_normal(frame) if ok else None
        if out is None:
            print(f"{clip}: skipped (note not a clean quadrilateral)")
            continue
        n, err = out
        normals.append(n)
        print(f"{clip}: table normal {n.round(3)}, reprojection {err:.1f} px")
    n = np.mean(normals, axis=0)
    n /= np.linalg.norm(n)
    print(f"mean table normal in camera axes {n.round(4)}; camera looks "
          f"{np.degrees(np.arcsin(-n[2])):.1f} deg below horizontal")


if __name__ == "__main__":
    main()
