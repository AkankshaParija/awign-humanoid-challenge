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

## 5. Mark the trials by hand

An episode can hold several pick-and-places: the hand releases the cup but
stays in frame, so the hand-present gaps cannot separate them.

Automatic segmentation from grasp aperture was tried and **does not work for
this task**. A wrap-around cup grasp keeps the thumb roughly a cup-diameter
from the fingers throughout, so the signal barely changes between an open hand
near the cup and a hand holding it, and the thumb is usually occluded by the
cup during the grasp anyway. `segment_trials.py` is kept in the repo for the
record; `outputs/aperture.png` shows why it was abandoned. That is worth a
line in the README under what did not work.

So the events get marked directly, which is one pass and two keys per trial:

```bash
python scripts/annotate.py data/episodes data/annotations.csv
```

| key | action |
|---|---|
| space | play / pause |
| `.` `,` | step one frame |
| `>` `<` | jump ten frames |
| `g` | mark grasp at this frame |
| `r` | mark release, closing the trial |
| `u` | undo the last mark |
| `n` / `p` | next / previous clip |
| `q` | save and quit |

Mark the frame where the hand first closes on the cup (`g`) and the frame
where it lets go (`r`). The CSV is written after every clip, so quitting and
resuming is safe, and already-annotated clips are skipped on the next run.

Then build the per-trial datasets:

```bash
python scripts/build_trials.py data/annotations.csv data/episodes data/trials
```

Each trial gets its keypoints, its grasp and release frames, and a per-frame
phase label: reach, grasp, transport, release, retreat. The script prints the
phase distribution, which is a good sanity check and a nice table for the
README.

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

## 7. Retarget onto the Shadow hand

Check what the script auto-detected before running it for real:

```bash
python scripts/retarget.py --model mujoco_menagerie/shadow_hand/right_hand.xml --probe
```

It prints the palm body, the fingertip body it picked per finger, and the
joints it will optimise. If a finger is missing or the tip body looks wrong,
say so before going further, the rest of the pipeline depends on this mapping.

Then run it:

```bash
# one episode first
python scripts/retarget.py data/trials/ep_001_t01.json outputs/qpos/ep_001_t01.json \
    --model mujoco_menagerie/shadow_hand/right_hand.xml

# then the rest
python scripts/retarget.py data/trials outputs/qpos \
    --model mujoco_menagerie/shadow_hand/right_hand.xml
```

Mean tip error is printed per episode. Single-digit mm is good; above about
20mm something is off with the mapping or the keypoints.

How it works: both hands are put into their own palm frame (wrist, index MCP,
little MCP) so camera orientation and model conventions cancel out, leaving
only pose shape. A bounded least-squares solve then finds the finger joints
that place the robot fingertips on the scaled human targets, warm started
from the previous frame for smoothness. The first frame gets a dedicated
multi-restart solve with no smoothing term, which is the initialisation point
DexMV's supplementary material raises.

Wrist pose is not optimised, it comes straight from MediaPipe and is saved as
`wrist_pos` / `wrist_quat` for the replay step to drive the free joint.

## Still to come

- `replay.py`: drive the sim scene, fingertip-contact grasp check, record GIFs
