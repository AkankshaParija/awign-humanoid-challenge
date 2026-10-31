"""Extract 3D hand keypoints from every clip in a folder using MediaPipe.

Writes one JSON per clip into the output folder:
    {
      "clip": "clip_001.mp4",
      "fps": 30.0,
      "n_frames": 180,
      "world": [[[x,y,z] * 21] * n_frames],   # metric, wrist-centred
      "image": [[[x,y,z] * 21] * n_frames],   # normalised image coords
      "detected": [true, false, ...]
    }

"world" is what you retarget from (it is in metres, origin at the wrist).
"image" is useful for overlaying and for locating the hand in the frame.

Usage:
    python scripts/extract_hands.py --check
    python scripts/extract_hands.py data/raw data/keypoints
    python scripts/extract_hands.py data/raw data/keypoints --no-filter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

N_LANDMARKS = 21
TASK_MODEL = "models/hand_landmarker.task"
TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def detect_api() -> str:
    """Return 'tasks' or 'legacy' depending on what this mediapipe exposes."""
    try:
        from mediapipe.tasks.python import vision  # noqa: F401

        return "tasks"
    except Exception:
        pass
    try:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            return "legacy"
    except Exception:
        pass
    return "none"


class TasksExtractor:
    def __init__(self, model_path: str) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"missing {model_path}\n"
                f"download it with:\n  mkdir -p models && curl -L -o {model_path} {TASK_MODEL_URL}"
            )
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def __call__(self, frame_bgr, timestamp_ms: int):
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect_for_video(image, timestamp_ms)
        if not res.hand_landmarks:
            return None, None
        img = [[p.x, p.y, p.z] for p in res.hand_landmarks[0]]
        world = [[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]]
        return np.array(img), np.array(world)

    def close(self) -> None:
        self.landmarker.close()


class LegacyExtractor:
    def __init__(self) -> None:
        import mediapipe as mp

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, frame_bgr, timestamp_ms: int):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None, None
        img = [[p.x, p.y, p.z] for p in res.multi_hand_landmarks[0].landmark]
        world = [[p.x, p.y, p.z] for p in res.multi_hand_world_landmarks[0].landmark]
        return np.array(img), np.array(world)

    def close(self) -> None:
        self.hands.close()


def fill_gaps(seq: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Linearly interpolate frames where no hand was detected."""
    if detected.all():
        return seq
    if not detected.any():
        return seq
    idx = np.arange(len(seq))
    good = idx[detected]
    out = seq.copy()
    for j in range(seq.shape[1]):
        for k in range(seq.shape[2]):
            out[:, j, k] = np.interp(idx, good, seq[good, j, k])
    return out


def lowpass(seq: np.ndarray, fps: float, cutoff_hz: float = 6.0) -> np.ndarray:
    """Zero-phase Butterworth low-pass, the smoothing step DexMV describes."""
    from scipy.signal import butter, filtfilt

    nyq = fps / 2.0
    wn = min(cutoff_hz / nyq, 0.99)
    if len(seq) < 12:
        return seq
    b, a = butter(2, wn, btype="low")
    flat = seq.reshape(len(seq), -1)
    smoothed = filtfilt(b, a, flat, axis=0)
    return smoothed.reshape(seq.shape)


def process_clip(path: Path, extractor, apply_filter: bool, max_width: int = 1280) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Downscaling costs no accuracy (MediaPipe resizes internally anyway) but
    # saves a lot of time on 4K footage. Image landmarks stay normalised, so
    # the output format is unchanged.
    scale = min(1.0, max_width / src_w) if (max_width and src_w) else 1.0

    img_seq, world_seq, detected = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        if total and i % 300 == 0:
            print(f"    {i}/{total} frames ({i / total * 100:.0f}%)", flush=True)
        ts = int(i * 1000.0 / fps)
        img, world = extractor(frame, ts)
        if img is None:
            img = np.zeros((N_LANDMARKS, 3))
            world = np.zeros((N_LANDMARKS, 3))
            detected.append(False)
        else:
            detected.append(True)
        img_seq.append(img)
        world_seq.append(world)
        i += 1
    cap.release()

    if not img_seq:
        raise RuntimeError(f"no frames read from {path}")

    img_arr = np.array(img_seq)
    world_arr = np.array(world_seq)
    det = np.array(detected)

    img_arr = fill_gaps(img_arr, det)
    world_arr = fill_gaps(world_arr, det)

    if apply_filter:
        img_arr = lowpass(img_arr, fps)
        world_arr = lowpass(world_arr, fps)

    rate = det.mean() * 100
    print(f"  {path.name}: {len(det)} frames, hand found in {rate:.0f}%")

    return {
        "clip": path.name,
        "fps": float(fps),
        "n_frames": int(len(det)),
        "world": world_arr.tolist(),
        "image": img_arr.tolist(),
        "detected": det.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", nargs="?", default="data/raw")
    ap.add_argument("out_dir", nargs="?", default="data/keypoints")
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--max-width", type=int, default=1280,
                    help="downscale frames to this width before detection (0 disables)")
    ap.add_argument("--check", action="store_true", help="report which MediaPipe API is available")
    args = ap.parse_args()

    api = detect_api()
    if args.check:
        print(f"mediapipe API available: {api}")
        if api == "tasks":
            have = Path(TASK_MODEL).exists()
            print(f"task model present: {have}")
            if not have:
                print(f"download:\n  mkdir -p models && curl -L -o {TASK_MODEL} {TASK_MODEL_URL}")
        return

    if api == "none":
        sys.exit("could not find a usable MediaPipe hand API. run with --check")

    extractor = TasksExtractor(TASK_MODEL) if api == "tasks" else LegacyExtractor()
    print(f"using MediaPipe {api} API")

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(
        p for p in raw_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".m4v", ".avi"}
    )
    if not clips:
        sys.exit(f"no video files in {raw_dir}")

    print(f"found {len(clips)} clips in {raw_dir}")
    for clip in clips:
        result = process_clip(clip, extractor, apply_filter=not args.no_filter,
                              max_width=args.max_width)
        out_path = out_dir / f"{clip.stem}.json"
        out_path.write_text(json.dumps(result))
    extractor.close()
    print(f"\nwrote {len(clips)} files to {out_dir}")


if __name__ == "__main__":
    main()
