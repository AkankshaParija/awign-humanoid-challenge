#!/usr/bin/env bash
# One-time setup: folders, extra deps, MediaPipe model.
# Run from the repo root with your venv active.
set -euo pipefail

echo "==> creating folders"
mkdir -p data/raw data/keypoints data/episodes data/sample
mkdir -p scripts sim models outputs

echo "==> installing python deps"
pip install --quiet --upgrade scipy

echo "==> fetching MediaPipe hand landmarker model"
if [ ! -f models/hand_landmarker.task ]; then
  curl -L -o models/hand_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
  echo "    downloaded models/hand_landmarker.task"
else
  echo "    already present, skipping"
fi

echo "==> checking mediapipe"
python scripts/extract_hands.py --check

echo
echo "setup done. next:"
echo "  1. copy your long recording to data/raw/session_01.mp4"
echo "  2. python scripts/extract_hands.py data/raw data/keypoints"
echo "  3. python scripts/split_episodes.py data/keypoints/session_01.json data/episodes --dry-run"
