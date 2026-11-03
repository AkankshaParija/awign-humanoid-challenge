# Filming checklist

This was the pre-filming plan, written before recording. The ArUco marker
and `data/setup.md` calibration below were **not** actually captured during
filming — see the README's Limitations section for what that means for the
replay results (no camera-to-table calibration, so object placement in the
sim scene is not independently measured). Kept here as the original plan,
not as a description of what the committed data actually contains.

Target: ~30 clips, 5 to 10 seconds each. About 40 minutes of work.

## Fixed setup (do not move anything once you start)

- Phone on a tripod or a stack of books, locked in place.
- Plain table, plain background, no clutter.
- ArUco marker (or any flat object of known size) taped to the table. This is
  your world origin, it must stay put and stay visible in every frame.
- Good daylight, no harsh shadows across the hand.
- Lock the phone's exposure and focus so they do not drift between clips.
- Record at 1080p/30fps. Higher fps is fine, higher resolution is not needed.

## Measure once, write it down

Record these in `data/setup.md`:

- Table height, marker position relative to the table edge.
- Coaster position relative to the marker (x, y in cm).
- Cup diameter and height.
- Camera height and roughly how far from the table edge.

The sim scene gets built from these numbers, so be reasonably careful.

## The task

Right hand picks up the cup and places it on the coaster.

- Same grasp style every clip (fingers around the body, not the handle).
- Start with your hand out of frame or resting flat, then reach.
- Move at a normal pace, not fast. Motion blur ruins pose extraction.
- Keep the hand fully in frame the entire time.
- Pause for half a second after placing, before pulling the hand away.

## Coverage

Use 5 or 6 cup start positions, about 5 clips each:

| Position | Where |
|---|---|
| A | near centre |
| B | left of centre |
| C | right of centre |
| D | far from hand |
| E | close to hand |
| F | (optional) rotated/offset |

Coaster stays in one place for all clips. Vary only the cup.

## Naming

Save as `data/raw/clip_001.mp4`, `clip_002.mp4`, ... in order.
Note which position each clip used in `data/setup.md`.

## Before you stop filming

Run the extractor on 2 or 3 clips and look at the overlay:

```bash
python scripts/extract_hands.py data/raw data/keypoints
python scripts/preview_keypoints.py data/raw/clip_001.mp4 \
    data/keypoints/clip_001.json /tmp/check.mp4
```

If the hand is detected in well under 90% of frames, fix the lighting or the
angle and refilm now, while everything is still set up.
