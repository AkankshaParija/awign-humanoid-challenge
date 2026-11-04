# Smoothing experiment (Job 1) — results, not used as the canonical pipeline

Savitzky-Golay smoothing (window=7, polyorder=2) on the MediaPipe 'world'
landmarks, re-run through the unmodified retarget.py:

| metric | original | smoothed landmarks |
|---|---|---|
| mean fingertip error | 42.0 mm | 42.0 mm |
| mean \|joint velocity\| | 0.02756 rad/frame | 0.02587 rad/frame (-6%) |
| mean \|joint acceleration\| | 0.03273 rad/frame^2 | 0.02892 rad/frame^2 (-12%) |
| kinematic grasp count | 63/63 | 63/63 |
| physics grasp count | 59/63 | 60/63 |

(Grasp counts regenerated after fixing the replay joint-mapping bug, see
the README's "Bug found and fixed", and again with the approximate wrist
trajectory (replay.py --wrist image, depth clamp) and the penetration-
minimising kinematic attach. The earlier counts, computed with the
scrambled joints, were 55/63 -> 51/63 kinematic and 31/63 -> 26/63 physics.)

Jitter goes down, fingertip accuracy is unchanged, and the grasp counts
barely move (kinematic unchanged, physics +1 trial). Net: not a clear win. Kept outputs/qpos (unsmoothed) as
the canonical data used everywhere else in this repo (README, GIFs).

Smoothing weight sweep (retarget.py --smooth, on a 16-trial subset, same
smoothed landmarks): 0.005->57.7mm, 0.01 (default)->57.7mm, 0.03->58.2mm,
0.08->59.8mm, 0.2->65.8mm. 0.0 (no regularization) fails to converge on
some trials and hangs. The existing default (0.01) is already at or near
the optimum; no change made.

Tracking dropouts: already linearly interpolated upstream of this pipeline
(scripts/extract_hands.py, before episodes are split) -- e.g. ep_004 had
16/232 frames without a detection, already filled in. No further change
needed here.
