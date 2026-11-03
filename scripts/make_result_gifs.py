"""Render the README's result GIFs: one close, wide camera on hand+cup+sticky
note, idle approach/retreat frames trimmed to the annotated grasp/release
window, a lighter hand colour for contrast, and a text overlay with trial id,
grasp yes/no and live fingertip-contact count.

Reuses Replayer/fingertip_contacts from replay.py unchanged -- this only
adds trimming, recolouring and the overlay on top, it does not touch how
contact or the kinematic attach logic works.

    python scripts/make_result_gifs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import Replayer, fingertip_contacts  # noqa: E402
from make_gifs import mp4_to_gif  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "mujoco_menagerie/shadow_hand/scene_cup.xml"
QPOS_DIR = ROOT / "outputs/qpos"
TRIALS_DIR = ROOT / "data/trials"
VIDEO_DIR = ROOT / "outputs/videos_v2"
GIF_DIR = ROOT / "docs/gifs"

LIGHT_HAND_RGBA = [0.82, 0.82, 0.86, 1.0]

TRIALS = {
    "ep_007_t01": {"failure": False},
    "ep_019_t01": {"failure": False},
    "ep_033_t01": {"failure": False},
    "ep_011_t02": {"failure": True,
                   "why": "fingertips never close on the cup (retarget error 76mm)"},
}


def lighten_hand(model) -> None:
    for i in range(model.ngeom):
        b = model.geom_bodyid[i]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name.startswith("rh_"):
            model.geom_rgba[i] = LIGHT_HAND_RGBA


def render_trial(rep: Replayer, trial: str, lead: int = 8, tail: int = 8) -> Path:
    rec = json.loads((QPOS_DIR / f"{trial}.json").read_text())
    trial_meta = json.loads((TRIALS_DIR / f"{trial}.json").read_text())
    grasp_f, release_f = trial_meta["grasp_frame"], trial_meta["release_frame"]
    n_total = len(rec["qpos"])
    lo = max(0, grasp_f - lead)
    hi = min(n_total, release_f + tail)

    cup_start = rep.tip_centroid_at_frame(rec, grasp_f)

    qpos = np.array(rec["qpos"])
    qadr = np.array(rec["qpos_adr"])
    wrist_pos = np.array(rec["wrist_pos"])
    wrist_quat = np.array(rec["wrist_quat"])
    fps = rec.get("fps", 30.0)
    hand_free_adr = rep._hand_free_adr()

    # frame the camera on where the hand actually moves in this clip, not
    # the scene's sticky-note placeholder -- there is no calibration tying
    # the two together (see README Limitations), so centring on the
    # placeholder can point the camera at empty space
    palm_body = mujoco.mj_name2id(rep.model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")
    palm_track, tip_track = [], []
    for t in range(lo, hi):
        rep.data.qpos[hand_free_adr:hand_free_adr + 3] = wrist_pos[t]
        rep.data.qpos[hand_free_adr + 3:hand_free_adr + 7] = wrist_quat[t]
        rep.data.qpos[qadr] = qpos[t]
        mujoco.mj_forward(rep.model, rep.data)
        palm_track.append(rep.data.xpos[palm_body].copy())
        tip_track.append(np.mean([rep.data.xpos[i] for i in sorted(rep.tips)], axis=0).copy())
    palm_track, tip_track = np.array(palm_track), np.array(tip_track)
    # frame on the palm, not the fingertip centroid: fingertip position
    # depends on the solved joint angles, which can be degenerate on a
    # badly-retargeted trial, while the palm only depends on wrist_pos/quat
    # (straight from MediaPipe) and always stays near the actual hand
    center = np.median(palm_track, axis=0)
    spread = float(np.percentile(
        np.linalg.norm(np.vstack([palm_track, tip_track]) - center, axis=1), 90))

    mujoco.mj_resetData(rep.model, rep.data)
    rep.data.qpos[rep.cup_qadr:rep.cup_qadr + 3] = cup_start
    rep.data.qpos[rep.cup_qadr + 3:rep.cup_qadr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(rep.model, rep.data)

    rep.cam.lookat[:] = center
    # wide enough to keep a jittery/degenerate trajectory in frame -- a
    # failure case that swings the hand all over is exactly what should be
    # visible, not cropped out
    rep.cam.distance = min(1.1, max(0.4, 1.4 * spread + 0.22))
    rep.cam.azimuth = 150
    # a shallower, more head-on angle: a badly-retargeted trial can swing
    # the hand well above or below the table plane, and a steep top-down
    # angle loses it behind the floor/table geometry
    rep.cam.elevation = -12

    held, grip_offset = False, None
    frames = []
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
        frame = rep.renderer.render().copy()

        meta = TRIALS[trial]
        grasp_txt = "grasp: yes" if held or ncon >= 2 else "grasp: no"
        lines = [f"{trial}", grasp_txt, f"fingertips touching: {ncon}"]
        if meta["failure"]:
            lines.append(f"FAILED: {meta['why']}")
        y = 24
        for line in lines:
            color = (255, 80, 80) if meta["failure"] and line.startswith("FAILED") else (20, 20, 20)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                       (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                       color, 1, cv2.LINE_AA)
            y += 22
        frames.append(frame)

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_DIR / f"{trial}.mp4"
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"  {trial}: frames {lo}-{hi} ({hi - lo} frames) -> {out_path}")
    return out_path


def main() -> None:
    rep = Replayer(str(SCENE), mode="kinematic", width=960, height=720)
    lighten_hand(rep.model)
    rep.sticky_note_body = mujoco.mj_name2id(rep.model, mujoco.mjtObj.mjOBJ_BODY, "sticky_note")

    GIF_DIR.mkdir(parents=True, exist_ok=True)
    for trial in TRIALS:
        mp4 = render_trial(rep, trial)
        gif = GIF_DIR / f"{trial}.gif"
        # keep gifs under ~5MB: stride up for longer clips rather than
        # shrinking below the 640px-wide floor
        n_frames = int(cv2.VideoCapture(str(mp4)).get(cv2.CAP_PROP_FRAME_COUNT))
        stride = 1 if n_frames <= 70 else (2 if n_frames <= 140 else 3)
        mp4_to_gif(mp4, gif, stride=stride, max_width=640)


if __name__ == "__main__":
    main()
