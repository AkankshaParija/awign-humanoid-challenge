"""Finger-articulation GIFs: phone footage with MediaPipe landmarks (left)
next to the Shadow Hand (right) driven ONLY by the retargeted finger joint
angles for that trial.

  * wrist fixed: the hand's root pose is one constant canonical pose for
    every frame and every clip (fingers up, palm towards the camera, hand
    centred); no wrist translation, no wrist rotation, no cup.
  * same camera for every clip, light flat background (everything that is
    not hand is painted over using MuJoCo's segmentation render), the same
    high-contrast orange hand as the other GIFs.
  * checks, printed per clip: sim visibility (palm + 5 fingertips projected
    inside the image, and >=2% of pixels change when the hand is hidden;
    every shown frame must pass), and the phone face-line rule (crop never
    above y=200 px, every hand landmark below the line and inside the crop).
  * only frames where MediaPipe actually detected the hand are shown (the
    longest contiguous run of detected frames that also passes the checks),
    so no interpolated tracking appears in the clip.

The replay logic and every reported number are untouched: this only poses
the hand from outputs/qpos and renders it.

    python scripts/make_finger_gifs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_result_gifs as R  # noqa: E402  (shared panels, checks, GIF writer)
from replay import Replayer  # noqa: E402

ROOT = R.ROOT
GIF_DIR = ROOT / "docs/gifs"

# Picked from all 63 trials by finger flexion range (mean non-thumb J1-J3
# flexion, fraction of joint range, 5th-95th percentile over the trial),
# preferring continuous motion over trials whose flexion joints sit pinned
# at a joint limit and jump in steps, and high tracking confidence (share of
# frames MediaPipe actually detected). See the README "Finger retargeting
# replay" section for the numbers.
TRIALS = ["ep_028_t01", "ep_029_t01"]

# canonical hand pose: with the root at identity the fingers point +z and
# the palm faces +x, so a camera at azimuth 180 (looking along -x) sees the palm
ROOT_POS = np.array([0.0, 0.0, 1.0])
ROOT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
CAM_AZIMUTH = 180
CAM_ELEVATION = -25                # slightly above: curled fingers read better than head-on
CAM_DISTANCE = 0.34                # hand fills the panel; open hand still fits (checked)
BACKGROUND_BGR = (236, 240, 242)   # light warm grey
OVERLAY_MAX_LINES = 2


def hide_scene_props(model) -> None:
    """Move the cup, table and sticky note far away: this view has no cup."""
    for name in ("cup", "table", "sticky_note"):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b >= 0:
            model.body_pos[b] = [50.0, 50.0, -50.0]


def flat_background(rgb: np.ndarray, seg: np.ndarray, hand_geoms: set[int]) -> np.ndarray:
    """Keep hand pixels, paint everything else BACKGROUND_BGR (BGR out)."""
    ids, types = seg[..., 0], seg[..., 1]
    is_hand = (types == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(ids, list(hand_geoms))
    out = np.empty_like(rgb)
    out[:] = BACKGROUND_BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out[is_hand] = bgr[is_hand]
    return out


def detected_frames(trial_meta: dict) -> np.ndarray:
    """Per trial frame: did MediaPipe actually detect the hand (not interpolated)?"""
    ep = json.loads((R.EPISODES_DIR / trial_meta["source_episode"]).read_text())
    trial_image = np.array(trial_meta["image"])
    off = R.find_episode_offset(trial_image, np.array(ep["image"]))
    return np.array(ep["detected"][off:off + len(trial_image)], dtype=bool)


def longest_run(ok: list[bool]) -> tuple[int, int]:
    """[start, end) of the longest run of True."""
    best, start = (0, 0), None
    for i, v in enumerate(ok + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def render_trial(rep: Replayer, seg_renderer, trial: str, hand_geoms: list[int]) -> dict:
    meta_t = json.loads((R.TRIALS_DIR / f"{trial}.json").read_text())
    rec = json.loads((R.QPOS_DIR / f"{trial}.json").read_text())
    qpos = np.array(rec["qpos"])
    qadr = rep.scene_qpos_adr(rec)                 # by joint name, like replay.py
    ha = rep._hand_free_adr()
    palm_body = mujoco.mj_name2id(rep.model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")
    tips = sorted(rep.tips)
    n = len(qpos)

    # fixed camera, centred on the hand in its canonical open pose
    mujoco.mj_resetData(rep.model, rep.data)
    rep.data.qpos[ha:ha + 3] = ROOT_POS
    rep.data.qpos[ha + 3:ha + 7] = ROOT_QUAT
    mujoco.mj_forward(rep.model, rep.data)
    hand_bodies = {int(rep.model.geom_bodyid[g]) for g in hand_geoms}
    centre = np.mean([rep.data.xpos[b] for b in hand_bodies], axis=0)
    rep.cam.lookat[:] = centre
    rep.cam.distance = CAM_DISTANCE
    rep.cam.azimuth = CAM_AZIMUTH
    rep.cam.elevation = CAM_ELEVATION

    frames = []
    for t in range(n):
        rep.data.qpos[ha:ha + 3] = ROOT_POS        # wrist never moves
        rep.data.qpos[ha + 3:ha + 7] = ROOT_QUAT
        rep.data.qpos[qadr] = qpos[t]              # finger joints from retargeting
        mujoco.mj_forward(rep.model, rep.data)
        rep.renderer.update_scene(rep.data, rep.cam)
        rgb = rep.renderer.render().copy()
        inside = R.points_inside(rep, [rep.data.xpos[palm_body]] + [rep.data.xpos[i] for i in tips])
        pixels = R.hand_pixel_frac(rep, rgb, hand_geoms) >= R.MIN_HAND_PIXEL_FRAC
        seg_renderer.update_scene(rep.data, rep.cam)
        seg = seg_renderer.render()
        frames.append({"t": t, "img": flat_background(rgb, seg, set(hand_geoms)),
                       "ok_sim": inside and pixels})

    # show only frames where tracking is real and every check passes
    detected = detected_frames(meta_t)
    phone_ok = R.phone_landmarks_ok(meta_t)
    ok = [bool(detected[f["t"]] and phone_ok[f["t"]] and f["ok_sim"]) for f in frames]
    a, b = longest_run(ok)
    frames = frames[a:b]
    lo, hi = frames[0]["t"], frames[-1]["t"] + 1

    phones, checks = R.phone_frames(meta_t, lo, hi)
    out = []
    for pf, f in zip(phones, frames):
        canvas = np.hstack([pf, f["img"]])
        lines = [(f"{trial}  frame {f['t']}", (240, 240, 240)),
                 ("finger retargeting only (wrist fixed)", (240, 240, 240))]
        assert len(lines) <= OVERLAY_MAX_LINES
        R.overlay(canvas, lines)
        out.append(canvas)

    flex = R_flexion(rep.model, rec)[lo:hi]
    return {"frames": out, "lo": lo, "hi": hi, "n_total": n,
            "sim_pass": all(f["ok_sim"] for f in frames),
            "detected_shown": float(detected[lo:hi].mean()),
            "flex_min": float(flex.min()), "flex_max": float(flex.max()),
            **checks, "fps": rec.get("fps", 30.0)}


def R_flexion(model, rec: dict) -> np.ndarray:
    """Mean non-thumb flexion (J1-J3) per frame, as a fraction of joint range."""
    import re
    idx, lo, hi = [], [], []
    for i, name in enumerate(rec["joint_names"]):
        if re.match(r"rh_(FF|MF|RF|LF)J[123]$", name):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            idx.append(i)
            lo.append(model.jnt_range[j][0])
            hi.append(model.jnt_range[j][1])
    q = np.array(rec["qpos"])[:, idx]
    return ((q - np.array(lo)) / (np.array(hi) - np.array(lo))).mean(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", nargs="*", default=TRIALS)
    ap.add_argument("--gif-dir", type=Path, default=GIF_DIR)
    ap.add_argument("--stills", type=Path, default=ROOT / "outputs/gif_stills")
    args = ap.parse_args()

    rep = Replayer(str(R.SCENE), mode="kinematic", width=R.MJ_W, height=R.MJ_H)
    hand_geoms = R.style_scene(rep.model)
    hide_scene_props(rep.model)
    seg_renderer = mujoco.Renderer(rep.model, height=R.MJ_H, width=R.MJ_W)
    seg_renderer.enable_segmentation_rendering()

    for trial in args.trials:
        res = render_trial(rep, seg_renderer, trial, hand_geoms)
        gif = args.gif_dir / f"{trial}_fingers.gif"
        R.save_gif(res["frames"], gif, res["fps"])
        args.stills.mkdir(parents=True, exist_ok=True)
        for name, i in (("start", 0), ("mid", len(res["frames"]) // 2), ("end", -1)):
            cv2.imwrite(str(args.stills / f"{trial}_fingers_{name}.png"), res["frames"][i])
        size_mb = gif.stat().st_size / 1e6
        good = (res["sim_pass"] and res["crop_below_face_line"] and res["hand_in_crop"]
                and res["detected_shown"] == 1.0 and size_mb < 5)
        print(f"{trial}: {'OK' if good else 'FAIL'}  frames {res['lo']}-{res['hi'] - 1} "
              f"({res['hi'] - res['lo']} of {res['n_total']}), sim visibility all frames "
              f"{res['sim_pass']}, detected in shown frames {res['detected_shown'] * 100:.0f}%, "
              f"crop top y={res['crop'][1]} (>= {R.FACE_LINE_Y}: {res['crop_below_face_line']}), "
              f"hand in crop {res['hand_in_crop']}, flexion shown {res['flex_min']:.2f}-"
              f"{res['flex_max']:.2f}, {gif.name} {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
