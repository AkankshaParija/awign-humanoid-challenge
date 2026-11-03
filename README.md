# Cup pick-and-place → Shadow Hand, from my own phone footage

**Akanksha Parija** — Awign x Armature AI Labs Humanoid Challenge submission.

Phone video of me
picking up a cup and placing it on a sticky note (and back) drives a
[Shadow Hand](https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand)
(floating wrist) in MuJoCo, via hand-pose extraction → manual event
annotation → kinematic retargeting → physics replay with contact-based
grasp detection.

All data is mine: one continuous phone recording of my own hand, hand-split
into 63 trials and hand-annotated frame-by-frame.

## Honest assessment

The hand motion is still not great. Mean fingertip error is 42mm, and for
four of the five fingers it is 44-51mm, only the thumb is good (17.7mm).
That is a real gap, not a rounding error, and I don't want to bury it.

The main cause is camera occlusion, not a bug in the retargeting solve. One
static phone camera means the cup blocks the view of the fingers wrapped
around it during transport, and tracking quality drops right when it
matters most. The thumb usually stays on the visible side of the grip in
this camera angle, which is why it alone tracks well.

I tried to fix this with temporal smoothing (Savitzky-Golay on the
landmarks, then re-run through the same retargeting code). Jitter went
down: mean joint velocity dropped about 6% and acceleration about 12%. But
fingertip error stayed exactly the same, 42.0mm before and after, because
smoothing cleans up noise, not a camera-angle bias. It also slightly hurt
the contact-based grasp count (55/63 down to 51/63 kinematic, 31/63 down
to 26/63 physics), because smoothing blunts the peak finger closure the
contact check looks for. So I kept the original, unsmoothed data as the
one used everywhere in this repo. Full numbers in
[outputs/retarget_smoothed/NOTES.md](outputs/retarget_smoothed/NOTES.md).

On the 55/63 grasp count: that number means the cup is seeded right at the
hand's own fingertip position at the annotated grasp frame, then contact
is tracked from there. It tells you the hand closes into a believable
grip around a cup-sized object, it does not tell you the hand found the
cup on its own in open space, because there's no camera-to-table
calibration to check that against.

Physics mode is weaker than kinematic mode, 31/63 grasps instead of 55/63,
and the cup often gets knocked away instead of lifted. That's expected
given a 42mm average grip error meeting real contact dynamics, and it's
the more honest number of the two.

## Results

| Trial | Grasp registered | Max fingertips touching | Cup moved |
|---|---|---|---|
| `ep_035_t01` (good) | 66/140 frames, ≥2 fingertips | 3 | 2.0 cm |
| `ep_021_t01` (good) | 66/136 frames, ≥2 fingertips | 3 | 6.5 cm |
| `ep_013_t01` (good) | 60/193 frames, ≥2 fingertips | 3 | 1.8 cm |
| `ep_011_t02` (failure) | 0/176 frames | 0 | 0 cm |

The 3 good trials are the top 3 by frames with ≥2 fingertips touching the cup
(ranked from `outputs/replay_summary.csv`, tie-broken by how many fingertips
touch at once — 3-way contact beats a borderline 2). `ep_011_t02` is kept as
the failure case because it is the one trial where fingertip contact never
happens even once (max contacts = 0), the cleanest example of retargeting
failing outright.

Each GIF is phone footage with the MediaPipe landmarks drawn on it (left)
synced frame-by-frame against the MuJoCo replay (right), trimmed to the
annotated grasp-to-release window, with a live overlay of fingertip contact
count and grasp status (`scripts/make_result_gifs.py`). The table and sticky
note in the MuJoCo panel are staged under wherever that trial's hand actually
moves — there's no camera-to-table calibration (see
[Honest assessment](#honest-assessment)), so this is a staging choice to make
the shot read as "hand over a table," not a claim about the real cup location:

![good trial, grasp succeeds, cup moves](docs/gifs/ep_035_t01.gif)
![good trial, sustained grasp](docs/gifs/ep_021_t01.gif)
![good trial, longest sustained contact](docs/gifs/ep_013_t01.gif)
![failure case, fingertips never close on the cup](docs/gifs/ep_011_t02.gif)

Across all 63 trials, replayed in the MuJoCo scene with real fingertip-contact
detection (no scripted "holding" flag):

- **55/63** trials register a grasp (≥2 fingertips touching the cup at some
  point). The cup is initialised at each trial's own annotated grasp frame,
  positioned at that frame's fingertip centroid — not at a real,
  independently-measured location — so this number validates hand shape and
  finger closure around a cup-sized object, not object localisation.
- **7/63** end with the cup within one sticky-note-width of the target
  location (heuristic — see [Limitations](#limitations); there is no
  camera-to-table calibration, so placement accuracy itself is not measured).
- The 8 trials with zero grasp frames are exactly the trials with the worst
  retargeting error (mean 35–76mm vs. the 42mm overall average) — the
  fingertips simply never get close enough to the cup for the
  ≥2-fingertip-contact threshold to fire. This is the retargeting error
  surfacing downstream, not a separate replay bug.

### Retargeting accuracy (63 trials, all episodes)

| Metric | Value |
|---|---|
| Mean fingertip error | 42.0 mm |
| Median fingertip error | 41.9 mm |
| Thumb error | 17.7 mm |
| Other 4 fingers, mean | 48.1 mm |
| Error vs. trial length, correlation (r) | 0.64 |

The thumb tracks far better than the other fingers (17.7mm vs. 44-51mm) even
though it has the least room for error during a wrap grip — see
[Occlusion analysis](#occlusion-analysis).

## Pipeline

```
phone video (one continuous take, camera static)
     │
     ▼
MediaPipe hand-pose extraction          scripts/extract_hands.py
     │  (4K downscaled to 1280px for detection)
     ▼
split into episodes (hand leaves frame) scripts/split_episodes.py
     │
     ▼
manual grasp/release annotation         scripts/annotate.py
     │  (63 trials: 32 to_target, 30 to_start; OpenCV keyboard tool,
     │   2 keystrokes per trial — g=grasp, r=release)
     ▼
build per-trial datasets + phase labels  scripts/build_trials.py
     │  (reach / grasp / transport / release / retreat)
     ▼
DexPilot/AnyTeleop-style retargeting     scripts/retarget.py
     │  (palm-frame alignment + bounded least-squares per frame,
     │   warm-started from the previous frame)
     ▼
MuJoCo scene: Shadow Hand + tapered cup  scripts/gen_scene.py, build_scene.py
     + sticky-note target on a table
     ▼
replay + fingertip-contact grasp check   scripts/replay.py
     │
     ▼
GIFs of good trials + a failure case     scripts/make_result_gifs.py
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

`mujoco_menagerie/` and the raw video/keypoint intermediates are not
committed (see `.gitignore`) — `data/trials/` (the per-trial datasets) and
`outputs/qpos/` (retargeted joint angles, already computed) are small
enough to keep and are committed, so replay works right after cloning
without re-running the retargeter.

No ffmpeg is used anywhere; video I/O goes through OpenCV, GIFs through
Pillow.

## How to run

```bash
# 1. build the MuJoCo scene (tapered cup + sticky note, from config.py)
python scripts/gen_scene.py
python scripts/build_scene.py

# 2. retarget every trial onto the Shadow Hand (already run; outputs/qpos/ has the results)
python scripts/retarget.py data/trials outputs/qpos \
    --model mujoco_menagerie/shadow_hand/right_hand.xml

# 3. replay every trial with fingertip-contact grasp detection
python scripts/replay.py outputs/qpos \
    --scene mujoco_menagerie/shadow_hand/scene_cup.xml \
    --summary outputs/replay_summary.csv

# 4. render a specific trial to video, then to GIF (generic, any trial)
python scripts/replay.py outputs/qpos/ep_007_t01.json \
    --scene mujoco_menagerie/shadow_hand/scene_cup.xml \
    --out outputs/videos/ep_007_t01.mp4
python scripts/make_gifs.py outputs/videos/ep_007_t01.mp4 docs/gifs/ep_007_t01.gif

# 5. regenerate the 4 README gifs specifically (close camera, trimmed, overlay)
python scripts/make_result_gifs.py
```

`PIPELINE.md` has the full step-by-step from raw phone footage onward
(extraction, splitting, annotation, trial-building) with the exact commands
and what to look for at each step.

## Occlusion analysis

Fingertip error correlates with trial length (r = 0.64) and the thumb tracks
far better than the other four fingers (17.7mm vs. 44–51mm, per-finger means:
index 44.4, middle 47.4, ring 50.5, little 50.2). Both point at
the same cause: during transport, the cup itself occludes the camera's view
of the fingers wrapped around it, and MediaPipe's detection confidence (and
accuracy) drops. The thumb is usually on the near/visible side of the grip in
this camera angle, which is why it survives occlusion better than the index
through little fingers, which end up on the far side, against or wrapped
around the cup. Longer trials simply spend more frames in that occluded
transport phase, so their error accumulates. This is a camera-angle and
occlusion limitation of a single static phone camera, not a bug in the
retargeting solve itself — the first-frame (fully visible, open-hand)
retargeting error is consistently sub-10mm; it's the mid-trial frames that
the correlation is driven by.

### What didn't work: aperture-based trial segmentation

`scripts/segment_trials.py` tries to find trial boundaries (and
reach/grasp/transport/release/retreat phases) automatically from the
thumb-to-fingertip distance ("aperture"). It does not work for this task: a
wrap-around cup grasp keeps the thumb roughly a cup-diameter from the other
fingertips throughout, so the aperture signal barely changes between "open
hand near the cup" and "hand holding the cup" — and the thumb is usually
occluded by the cup during the grasp anyway, which is the same occlusion
problem as above. `outputs/aperture.png` shows the flat, indistinct signal
that led to abandoning this approach in favor of manual annotation
(`scripts/annotate.py`), which the organizers specifically said they value
for quality.

## Limitations

- **"On sticky note" is a heuristic, not ground truth.** `wrist_pos` in the
  retargeted data comes straight from MediaPipe in the phone camera's own
  coordinate frame; there was no camera-to-table calibration captured during
  filming, so the scene's cup and sticky-note positions are placeholders,
  not measured locations. `replay.py` works around this by starting each
  trial's cup near the hand's own fingertips at the annotated grasp frame
  (so contact detection has something real to report), but "cup ends up near
  the sticky note" in the replay says more about whether the kinematic
  attach-on-contact logic tracked the hand correctly than about real-world
  placement accuracy. The headline metric that *is* trustworthy is the
  fingertip-contact grasp count, since that only depends on the relative
  geometry of the retargeted hand and the cup it's reaching for.
- **Kinematic mode doesn't test real contact dynamics.** The default replay
  mode teleports the cup to the fingertip centroid once ≥2 fingertips touch
  it; `--mode physics` leaves the cup to MuJoCo's contact solver instead.
  Run both and compare: physics mode drops to **31/63** trials registering
  a grasp (vs. 55/63 kinematic), and in many of those the cup is knocked
  tens of centimetres away rather than lifted, because the retargeted grip
  itself has ~40mm average fingertip error and an imprecise grip plus real
  contact dynamics mostly means a glancing hit, not a stable hold. Both
  modes are implemented (`--mode kinematic` / `--mode physics`); the
  physics numbers are the honest ones.
- **Occlusion, not solver error, is the main source of inaccuracy** — see
  above. A second camera angle or markers on the fingers would likely fix
  this more than any change to the retargeting math.
- **Temporal smoothing reduces jitter but not error.** See
  [Honest assessment](#honest-assessment) and
  [outputs/retarget_smoothed/NOTES.md](outputs/retarget_smoothed/NOTES.md)
  for the full before/after numbers — the attempt is kept in the repo for
  the record, but `outputs/qpos` (unsmoothed) is what everything else here
  uses.

## What I'd do next

- A second, overhead or side, camera to resolve the occlusion during
  transport — this is the single biggest lever on the 42mm error.
- Calibrate the phone camera against the real table/cup/sticky-note layout
  so "on sticky note" becomes a real spatial check instead of a heuristic.
- Policy training on top of the retargeted demonstrations (not attempted
  here, per the challenge's own phasing — this submission is the data
  pipeline and simulated replay, not an RL policy).
