"""Render the README's result GIFs: phone footage (with MediaPipe landmarks
drawn on top, cropped around the hand) side by side with the MuJoCo replay,
frame-synced, with a text overlay of trial id / running grasp-frame count /
live fingertip contact count.

The cup is STATIC in these renders: it is placed once at the hand's own
fingertip position at the annotated grasp frame and does not move again.
There is no scripted attach-and-carry logic here -- this script only shows
the retargeted hand moving past a fixed cup and counts fingertip contact,
it does not simulate or fake picking the cup up. (replay.py's kinematic
mode, used for the actual numbers in outputs/replay_summary.csv, does
attach the cup once contact holds; that is a separate, labeled replay mode
and is not used for these illustration GIFs.)

MuJoCo side: camera held tight and close on the hand/cup so fingers are
clearly visible closing (or not) around the cup. Forearm/wrist (the
oversized grey cylinder from the original Shadow Hand model) is hidden; the
visible hand is tinted a dark steel blue for contrast against the light
table. Table and sticky note are staged under wherever this trial's hand
actually operates -- there is no camera-to-table calibration (see README
Limitations), so this is a staging choice for a readable shot, not a claim
about the real cup location.

    python scripts/make_result_gifs.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import Replayer, fingertip_contacts  # noqa: E402
from make_gifs import mp4_to_gif  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CUP_HEIGHT, TABLE_HEIGHT  # noqa: E402

SCENE = ROOT / "mujoco_menagerie/shadow_hand/scene_cup.xml"
QPOS_DIR = ROOT / "outputs/qpos"
TRIALS_DIR = ROOT / "data/trials"
EPISODES_DIR = ROOT / "data/episodes"
VIDEO_DIR = ROOT / "outputs/videos_v4"
GIF_DIR = ROOT / "docs/gifs"

HAND_RGBA = [0.22, 0.32, 0.52, 1.0]   # dark steel blue, not white-on-tan
MJ_W, MJ_H = 640, 480                 # mujoco panel: >=640px wide
PHONE_SIDE = 480                      # phone panel: square crop, same height

# Picked by re-measuring fingertip contact with the cup STATIC (not
# attached/carried) over every trial's grasp-to-release window (77/117 for
# ep_015_t01, 57/127 for ep_031_t01, the top 2 by that count, both with
# 3-way simultaneous contact). The GIFs themselves render a shorter window
# centred on the grasp (see render_trial), so the numbers shown in the
# overlay/README (44/55 and 24/55) are measured over that shorter span, not
# the full window used to rank trials here -- both are real, just over
# different spans. These static counts are lower than
# outputs/replay_summary.csv's "frames_with_grasp" everywhere, because that
# pipeline lets the cup follow the fingertips once contact first holds (see
# replay.py's kinematic mode), sustaining contact long after the real,
# noisy fingertips would have drifted off a fixed cup. ep_011_t02 is kept
# as the failure case: one of the two trials where contact never happens
# even once (max contacts=0).
TRIALS = {
    "ep_015_t01": {"failure": False},
    "ep_031_t01": {"failure": False},
    "ep_011_t02": {"failure": True,
                   "why": "no contact (76mm retarget error)"},
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17), (0, 17),
]


def recolor_and_hide(model) -> None:
    for i in range(model.ngeom):
        b = model.geom_bodyid[i]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name in ("rh_forearm", "rh_wrist"):
            model.geom_rgba[i] = [0, 0, 0, 0]  # hidden: the oversized "cylinder"
        elif name.startswith("rh_"):
            model.geom_rgba[i] = HAND_RGBA


def find_episode_offset(trial_image: np.ndarray, ep_image: np.ndarray) -> int:
    """trial['image'] is a verbatim slice of the episode's image array
    (data/trials/*.json is built by build_trials.py as world[s:e]/image[s:e]);
    recover s by matching, since the trial json doesn't store it directly."""
    probe = trial_image[0]
    for s in range(len(ep_image) - len(trial_image) + 1):
        if np.allclose(ep_image[s], probe, atol=1e-6):
            return s
    raise SystemExit("could not align trial frames back to the source episode")


def crop_box(image_lm: np.ndarray, frame_w: int, frame_h: int,
            margin: float = 0.45) -> tuple[int, int, int, int]:
    """Union bounding box of the hand across every frame in the clip,
    expanded, so the crop stays fixed (no jitter) instead of re-centring
    every frame. Anchored at the bottom so the face (well above the hand
    in this camera setup) is cropped out."""
    xs = image_lm[:, :, 0].flatten() * frame_w
    ys = image_lm[:, :, 1].flatten() * frame_h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    side = max(w, h) * (1 + margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = cx - side / 2, cx + side / 2
    y1 = cy + side / 2
    y0 = y1 - side
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(frame_w, int(x1)), min(frame_h, int(y1))
    return x0, y0, x1, y1


def draw_landmarks(frame: np.ndarray, lm: np.ndarray, frame_w: int, frame_h: int) -> None:
    pts = [(int(x * frame_w), int(y * frame_h)) for x, y, _ in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 220, 80), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 4, (40, 220, 255), -1, cv2.LINE_AA)


def render_trial(rep: Replayer, trial: str, table_body: int, sticky_body: int,
                 sticky_offset: np.ndarray, lead: int = 10, after: int = 45) -> Path:
    rec = json.loads((QPOS_DIR / f"{trial}.json").read_text())
    trial_meta = json.loads((TRIALS_DIR / f"{trial}.json").read_text())
    grasp_f, release_f = trial_meta["grasp_frame"], trial_meta["release_frame"]
    n_total = len(rec["qpos"])
    # a window centred on the grasp, not the full grasp-to-release span:
    # the cup is static at the grasp-frame hand position (see module
    # docstring), and by the time a trial reaches transport the retargeted
    # wrist has often drifted well away from that point (the same jitter
    # documented in the README's occlusion analysis) -- a fixed close
    # camera loses the hand entirely past that point, so this only shows
    # the grasp-centric window the camera and caption both describe
    lo = max(0, grasp_f - lead)
    hi = min(n_total, release_f, grasp_f + after)
    window_n = hi - lo

    # --- phone footage side: align trial frames back to the source clip ---
    ep_path = EPISODES_DIR / trial_meta["source_episode"]
    ep = json.loads(ep_path.read_text())
    ep_image = np.array(ep["image"])
    trial_image = np.array(trial_meta["image"])
    offset = find_episode_offset(trial_image, ep_image)

    clip_path = EPISODES_DIR / trial_meta["source_clip"]
    cap = cv2.VideoCapture(str(clip_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, y0, x1, y1 = crop_box(trial_image[lo:hi], frame_w, frame_h)

    phone_frames = []
    for t in range(lo, hi):
        cap.set(cv2.CAP_PROP_POS_FRAMES, offset + t)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((frame_h, frame_w, 3), np.uint8)
        draw_landmarks(frame, trial_image[t], frame_w, frame_h)
        crop = frame[y0:y1, x0:x1]
        crop = cv2.resize(crop, (PHONE_SIDE, PHONE_SIDE))
        phone_frames.append(crop)
    cap.release()

    # --- mujoco side: cup is STATIC, placed once, never moved again ---
    cup_start = rep.tip_centroid_at_frame(rec, grasp_f)

    qpos = np.array(rec["qpos"])
    qadr = np.array(rec["qpos_adr"])
    wrist_pos = np.array(rec["wrist_pos"])
    wrist_quat = np.array(rec["wrist_quat"])
    fps = rec.get("fps", 30.0)
    hand_free_adr = rep._hand_free_adr()

    # stage the table/sticky-note under this trial's own hand trajectory
    # (see module docstring: illustrative placement, not a calibrated one)
    table_top_z = cup_start[2] - CUP_HEIGHT / 2 - 0.005
    rep.model.body_pos[table_body] = [cup_start[0], cup_start[1],
                                      table_top_z - TABLE_HEIGHT / 2]
    rep.model.body_pos[sticky_body] = [cup_start[0] + sticky_offset[0],
                                       cup_start[1] + sticky_offset[1],
                                       table_top_z + sticky_offset[2]]

    mujoco.mj_resetData(rep.model, rep.data)
    rep.data.qpos[rep.cup_qadr:rep.cup_qadr + 3] = cup_start
    rep.data.qpos[rep.cup_qadr + 3:rep.cup_qadr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(rep.model, rep.data)

    # tight, close camera: fingers and cup should clearly fill the frame.
    # The cup is a SOLID stack of cylinders (not a hollow drinking-cup
    # mesh), so a steep top-down angle just shows a solid disc blocking
    # the whole grip. Worse, the hand's wrist orientation changes a lot
    # over a clip (this is the same noisy tracking documented in the
    # README's occlusion analysis), so even a good azimuth picked from one
    # test frame goes back to hiding the hand a few frames later. Instead,
    # re-pick the azimuth EVERY frame: render a ring of candidates and keep
    # whichever one shows the most hand pixels in the center of frame.
    rep.cam.lookat[:] = [cup_start[0], cup_start[1], cup_start[2]]
    rep.cam.distance = 0.4
    rep.cam.elevation = -22

    def best_azimuth(candidates) -> int:
        best_az, best_score = candidates[0], -1
        for az in candidates:
            rep.cam.azimuth = az
            rep.renderer.update_scene(rep.data, rep.cam)
            full = rep.renderer.render().astype(int)
            hh, ww = full.shape[:2]
            # score only the central region: the floor/skybox checker
            # pattern is also blue and would otherwise swamp the count
            region = full[hh // 4:3 * hh // 4, ww // 4:3 * ww // 4]
            r, b = region[..., 0], region[..., 2]
            # the hand is the only blue-dominant object in the scene (cup
            # is red, table/sticky-note/floor are tan/yellow/dark blue but
            # not blue > red the way the hand tint is); robust to shading,
            # unlike matching the raw unshaded rgba value
            score = int(np.sum((b > r + 15) & (b > 60)))
            if score > best_score:
                best_score, best_az = score, az
        return best_az

    mj_frames, contact_counts = [], []
    for t in range(lo, hi):
        if hand_free_adr is not None:
            rep.data.qpos[hand_free_adr:hand_free_adr + 3] = wrist_pos[t]
            rep.data.qpos[hand_free_adr + 3:hand_free_adr + 7] = wrist_quat[t]
        rep.data.qpos[qadr] = qpos[t]
        # cup qpos is set once above and never touched again in this loop
        mujoco.mj_forward(rep.model, rep.data)

        ncon = fingertip_contacts(rep.model, rep.data, rep.tips, rep.cup_body)
        contact_counts.append(ncon)

        rep.cam.azimuth = best_azimuth(list(range(0, 360, 30)))
        rep.renderer.update_scene(rep.data, rep.cam)
        frame = cv2.cvtColor(rep.renderer.render().copy(), cv2.COLOR_RGB2BGR)
        mj_frames.append(frame)

    # --- compose side by side + overlay ---
    meta = TRIALS[trial]
    out_frames = []
    grasp_so_far = 0
    for i, (pf, mf) in enumerate(zip(phone_frames, mj_frames)):
        canvas = np.hstack([pf, mf])
        ncon = contact_counts[i]
        is_grasp = ncon >= 2
        if is_grasp:
            grasp_so_far += 1

        lines = [
            trial,
            f"fingertips in contact: {ncon}",
            f"grasp frames: {grasp_so_far} / {window_n}",
        ]
        if meta["failure"]:
            lines.append(f"FAILED: {meta['why']}")
        y = 26
        for line in lines:
            is_fail_line = meta["failure"] and line.startswith("FAILED")
            if is_fail_line:
                color = (80, 80, 255)
            elif is_grasp:
                color = (0, 200, 0)
            else:
                color = (235, 235, 235)
            scale = 0.6
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                       (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                       color, 2, cv2.LINE_AA)
            y += 26
        cv2.putText(canvas, "phone + MediaPipe", (10, PHONE_SIDE - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, "phone + MediaPipe", (10, PHONE_SIDE - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "MuJoCo replay (cup static)", (PHONE_SIDE + 10, PHONE_SIDE - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, "MuJoCo replay (cup static)", (PHONE_SIDE + 10, PHONE_SIDE - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out_frames.append(canvas)

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_DIR / f"{trial}.mp4"
    h, w = out_frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in out_frames:
        vw.write(f)
    vw.release()
    print(f"  {trial}: frames {lo}-{hi} ({window_n} frames), canvas {w}x{h}, "
          f"grasp {grasp_so_far}/{window_n} -> {out_path}")
    return out_path


def main() -> None:
    rep = Replayer(str(SCENE), mode="kinematic", width=MJ_W, height=MJ_H)
    recolor_and_hide(rep.model)

    table_body = mujoco.mj_name2id(rep.model, mujoco.mjtObj.mjOBJ_BODY, "table")
    sticky_body = mujoco.mj_name2id(rep.model, mujoco.mjtObj.mjOBJ_BODY, "sticky_note")
    # preserve the original scene's cup-to-sticky-note offset, just
    # re-centred on wherever this trial's hand actually operates
    orig_table_xy = rep.model.body_pos[table_body][:2].copy()
    orig_sticky = rep.model.body_pos[sticky_body].copy()
    orig_table_top_z = rep.model.body_pos[table_body][2] + TABLE_HEIGHT / 2
    sticky_offset = np.array([
        orig_sticky[0] - orig_table_xy[0],
        orig_sticky[1] - orig_table_xy[1],
        orig_sticky[2] - orig_table_top_z,
    ])

    GIF_DIR.mkdir(parents=True, exist_ok=True)
    for trial in TRIALS:
        mp4 = render_trial(rep, trial, table_body, sticky_body, sticky_offset)
        gif = GIF_DIR / f"{trial}.gif"
        cap = cv2.VideoCapture(str(mp4))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        canvas_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        # target ~15fps playback, capped frame count to keep file size down
        stride = max(1, round(src_fps / 15))
        stride = max(stride, math.ceil(n_frames / 32))
        mp4_to_gif(mp4, gif, stride=stride, max_width=canvas_w, fps=src_fps)


if __name__ == "__main__":
    main()
