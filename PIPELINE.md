# Pipeline, in run order

Everything below runs from the repo root with the venv active.

## 0. One-time setup

```bash
bash setup_project.sh
```

Creates the folders, installs scipy, downloads the MediaPipe hand model.

## 1. Get the footage in

Copy the **original** file off your phone (AirDrop or cable, not WhatsApp,
WhatsApp recompresses to about 850x480 and you lose detection quality).

```bash
cp ~/Desktop/your_recording.mov data/raw/session_01.mp4
```

One long continuous take is fine and preferred, the camera never moves.

## 2. Extract hand keypoints

```bash
python scripts/extract_hands.py data/raw data/keypoints
```

Prints a detection rate per clip. You want 90% or better. Below about 75%,
fix the lighting or the angle and refilm rather than pushing on.

## 3. Eyeball the tracking

```bash
python scripts/preview_keypoints.py \
    data/raw/session_01.mp4 data/keypoints/session_01.json outputs/overlay.mp4
open outputs/overlay.mp4
```

Red dots are fingertips. If the skeleton looks sane through the grasp, good.
Frames marked INTERPOLATED had no detection and were filled in.

## 4. Split the long take into episodes

```bash
# look first
python scripts/split_episodes.py data/keypoints/session_01.json data/episodes --dry-run

# then write them, with video clips alongside
python scripts/split_episodes.py data/keypoints/session_01.json data/episodes \
    --video data/raw/session_01.mp4 --write-clips
```

Expect one episode per trial. Too few means your between-trial gaps were
short, so lower `--min-gap`. Too many means a trial got cut in half, so
raise it.

## 5. Annotate phases

Label each episode with: reach, grasp, lift, place, release.
Label Studio works, or a CSV of frame numbers if you want it done faster.
The organisers said annotation quality matters, so do this properly.

## 6. Build the sim scene

```bash
python scripts/build_scene.py            # copies sim/scene_cup.xml next to the hand
python -m mujoco.viewer --mjcf=$PWD/mujoco_menagerie/shadow_hand/scene_cup.xml
```

Then tune the cup size, coaster position and table height in `sim/scene_cup.xml`
to match what you measured in `data/setup.md`, and re-run `build_scene.py`.

## 7. Inspect the hand model

```bash
python scripts/inspect_model.py mujoco_menagerie/shadow_hand/right_hand.xml
```

You need the joint names and ranges for the retargeter. Save the output.

## Still to come

- `retarget.py`: fit Shadow joint angles so its fingertips track yours
- `replay.py`: drive the sim, contact-based grasp check, record GIFs
