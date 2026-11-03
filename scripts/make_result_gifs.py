"""Render the README's result GIFs: phone footage (with MediaPipe landmarks
drawn on top, cropped around the hand) side by side with the MuJoCo replay,
frame-synced, with a text overlay of trial id / fingertip contact count /
grasp yes-no.

MuJoCo side: camera pulled back to show the whole hand, cup and a tabletop;
since there is no camera-to-table calibration (see README Limitations), the
table and sticky note are repositioned under wherever this trial's hand
actually moves, purely so the shot reads as "hand over a table" instead of
floating in empty space -- it is a staging choice, not a claim about the
real cup location. The forearm/wrist (the oversized grey cylinder from the
original Shadow Hand model) is hidden; only the functional hand is shown.

    python scripts/make_result_gifs.py
"""

from __future__ import annotations

import json
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
VIDEO_DIR = ROOT / "outputs/videos_v3"
GIF_DIR = ROOT / "docs/gifs"

LIGHT_HAND_RGBA = [0.82, 0.82, 0.86, 1.0]
PANEL_H = 220

# picked from outputs/replay_summary.csv: ranked by frames_with_grasp (>=2
# fingertips touching the cup), tie-broken by max_fingertip_contacts
# (how closed the grip gets, not just whether it clears the threshold).
# ep_035_t01 and ep_021_t01 tie for the most grasp-frames (66) with 3-way
# contact; ep_013_t01 is next (60 frames, also 3-way contact). ep_011_t02
# is kept as the failure case: it is the one trial where fingertip contact
# never happens even once (max_fingertip_contacts=0), the cleanest example
# of retargeting failing outright rather than partially.
TRIALS = {
    "ep_035_t01": {"failure": False},
    "ep_021_t01": {"failure": False},
    "ep_013_t01": {"failure": False},
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


def lighten_and_hide(model) -> None:
    for i in range(model.ngeom):
        b = model.geom_bodyid[i]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name in ("rh_forearm", "rh_wrist"):
            model.geom_rgba[i] = [0, 0, 0, 0]  # hidden: the oversized "cylinder"
        elif name.startswith("rh_"):
            model.geom_rgba[i] = LIGHT_HAND_RGBA


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
    every frame. Biased to crop tighter above the hand than below/around
    it, since the face sits well above the hand in this camera setup and
    this is meant to be a hand close-up, not a face shot."""
    xs = image_lm[:, :, 0].flatten() * frame_w
    ys = image_lm[:, :, 1].flatten() * frame_h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    side = max(w, h) * (1 + margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = cx - side / 2, cx + side / 2
    y1 = cy + side / 2
    y0 = y1 - side  # keep the full square, but anchor it at the bottom
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
                 sticky_offset: np.ndarray, lead: int = 8, tail: int = 8) -> Path:
    rec = json.loads((QPOS_DIR / f"{trial}.json").read_text())
    trial_meta = json.loads((TRIALS_DIR / f"{trial}.json").read_text())
    grasp_f, release_f = trial_meta["grasp_frame"], trial_meta["release_frame"]
    n_total = len(rec["qpos"])
    lo = max(0, grasp_f - lead)
    hi = min(n_total, release_f + tail)

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
        crop = cv2.resize(crop, (PANEL_H, PANEL_H))
        phone_frames.append(crop)
    cap.release()

    # --- mujoco side ---
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

    rep.cam.lookat[:] = [cup_start[0], cup_start[1], cup_start[2] + 0.02]
    rep.cam.distance = 0.5
    rep.cam.azimuth = 140
    rep.cam.elevation = -18

    held, grip_offset = False, None
    mj_frames, contact_counts = [], []
    for t in range(lo, hi):
        if hand_free_adr is not None:
            rep.data.qpos[hand_free_adr:hand_free_adr + 3] = wrist_pos[t]
            rep.data.qpos[hand_free_adr + 3:hand_free_adr + 7] = wrist_quat[t]
        rep.data.qpos[qadr] = qpos[t]
        mujoco.mj_forward(rep.model, rep.data)

        ncon = fingertip_contacts(rep.model, rep.data, rep.tips, rep.cup_body)
        if ncon >= 2 and not held:
            held = True
            tip_mean = np.mean([rep.data.xpos[i] for i in sorted(rep.tips)], axis=0)
            grip_offset = rep.data.qpos[rep.cup_qadr:rep.cup_qadr + 3] - tip_mean
        elif ncon < 1 and held:
            held = False
        if held and grip_offset is not None:
            tip_mean = np.mean([rep.data.xpos[i] for i in sorted(rep.tips)], axis=0)
            rep.data.qpos[rep.cup_qadr:rep.cup_qadr + 3] = tip_mean + grip_offset
            mujoco.mj_forward(rep.model, rep.data)

        rep.renderer.update_scene(rep.data, rep.cam)
        frame = cv2.cvtColor(rep.renderer.render().copy(), cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (int(PANEL_H * 4 / 3), PANEL_H))
        mj_frames.append(frame)
        contact_counts.append(ncon)

    # --- compose side by side + overlay ---
    meta = TRIALS[trial]
    out_frames = []
    for i, (pf, mf) in enumerate(zip(phone_frames, mj_frames)):
        canvas = np.hstack([pf, mf])
        ncon = contact_counts[i]
        grasp_txt = "grasp: yes (>=2)" if ncon >= 2 else "grasp: no"
        lines = [trial, f"fingertips in contact: {ncon}", grasp_txt]
        if meta["failure"]:
            lines.append(f"FAILED: {meta['why']}")
        y = 18
        for line in lines:
            is_fail_line = meta["failure"] and line.startswith("FAILED")
            color = (80, 80, 255) if is_fail_line else (30, 30, 30)
            scale = 0.42
            cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                       (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                       color, 1, cv2.LINE_AA)
            y += 17
        cv2.putText(canvas, "phone + MediaPipe", (8, PANEL_H - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "MuJoCo replay", (PANEL_H + 8, PANEL_H - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        out_frames.append(canvas)

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_DIR / f"{trial}.mp4"
    h, w = out_frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in out_frames:
        vw.write(f)
    vw.release()
    print(f"  {trial}: frames {lo}-{hi} ({hi - lo} frames), canvas {w}x{h} -> {out_path}")
    return out_path


def main() -> None:
    rep = Replayer(str(SCENE), mode="kinematic", width=640, height=PANEL_H)
    lighten_and_hide(rep.model)

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
        import math
        stride = max(1, round(src_fps / 15))
        stride = max(stride, math.ceil(n_frames / 45))
        mp4_to_gif(mp4, gif, stride=stride, max_width=canvas_w, fps=src_fps)


if __name__ == "__main__":
    main()
