"""orient-fix STEP 2: check the data-driven hand orientation against the phone.

A virtual pinhole camera is put exactly where the phone is in sim
coordinates (wrist_traj.to_sim_palm's phone_cam_pos_sim), looking along sim
+y with z up, with the same assumed focal length (924 px at 1280x720). If
the orientation is right, the sim hand should land on top of my real hand.

Reports, over every frame of every trial:
  * angle between the sim palm normal (from the model's wrist and knuckles
    after forward kinematics) and the palm normal from the landmarks;
  * pixel error between the projected sim wrist/knuckles and MediaPipe's
    image landmarks (wrist, index/middle/ring/pinky MCP).
Saves side-by-side stills (real | sim | overlay), cropped below the face
line, for the trials given.

    python scripts/orient_check.py --stills <dir> --trials ep_047_t01 ep_026_t01
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_result_gifs as R  # noqa: E402
import make_finger_gifs as FG  # noqa: E402
import wrist_traj as W  # noqa: E402
from replay import Replayer  # noqa: E402

RW, RH = 640, 360                        # render at half the phone resolution (same aspect)
FOVY_DEG = float(np.degrees(2 * np.arctan((W.IMAGE_H / 2) / W.FOCAL_PX)))
SIM_POINTS = ["rh_wrist", "rh_ffknuckle", "rh_mfknuckle", "rh_rfknuckle", "rh_lfknuckle"]
MP_POINTS = [0, 5, 9, 13, 17]


def project_phone(p_sim: np.ndarray, cam_pos: np.ndarray) -> np.ndarray:
    """Sim point -> pixel in the 1280x720 phone image (pinhole, assumed f)."""
    p_cam = W.CAM_TO_SIM.T @ (p_sim - cam_pos)          # back to OpenCV camera axes
    return np.array([W.FOCAL_PX * p_cam[0] / p_cam[2] + W.IMAGE_W / 2,
                     W.FOCAL_PX * p_cam[1] / p_cam[2] + W.IMAGE_H / 2])


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def pose(rep, rec, qadr, ha, t) -> None:
    rep.data.qpos[ha:ha + 3] = rec["wrist_pos"][t]
    rep.data.qpos[ha + 3:ha + 7] = rec["wrist_quat"][t]
    rep.data.qpos[qadr] = rec["qpos"][t]
    mujoco.mj_forward(rep.model, rep.data)


def measure(rep, trial: str) -> dict:
    meta = json.loads((R.TRIALS_DIR / f"{trial}.json").read_text())
    raw = json.loads((R.QPOS_DIR / f"{trial}.json").read_text())
    rec = W.to_sim_palm(rep, raw, meta)
    _, mp_frames = W.palm_orientation_sim(rep, np.array(meta["image"]))
    cam = np.array(rec["phone_cam_pos_sim"])
    qadr, ha = rep.scene_qpos_adr(rec), rep._hand_free_adr()
    body = lambda n: mujoco.mj_name2id(rep.model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    px_mp = np.array(meta["image"])[:, :, :2] * [W.IMAGE_W, W.IMAGE_H]
    ang, pix = [], []
    for t in range(len(raw["qpos"])):
        pose(rep, rec, qadr, ha, t)
        x = {n: rep.data.xpos[body(n)].copy() for n in SIM_POINTS}
        P_sim = W.palm_frame(x["rh_wrist"], x["rh_ffknuckle"], x["rh_mfknuckle"], x["rh_lfknuckle"])
        ang.append(angle_deg(P_sim[:, 2], mp_frames[t][:, 2]))
        pix.append(np.mean([np.linalg.norm(project_phone(x[n], cam) - px_mp[t, k])
                            for n, k in zip(SIM_POINTS, MP_POINTS)]))
    return {"rec": rec, "meta": meta, "cam": cam, "ang": np.array(ang), "pix": np.array(pix)}


def render_phone_view(rep, renderer, seg, hand_geoms, res: dict, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Sim hand seen from the phone (light background) and its mask, 640x360."""
    rec, cam = res["rec"], res["cam"]
    pose(rep, rec, rep.scene_qpos_adr(rec), rep._hand_free_adr(), t)
    vcam = mujoco.MjvCamera()
    vcam.type = mujoco.mjtCamera.mjCAMERA_FREE
    vcam.lookat[:] = cam + np.array([0.0, 1.0, 0.0])    # 1 m straight ahead (+y)
    vcam.distance, vcam.azimuth, vcam.elevation = 1.0, 90.0, 0.0
    renderer.update_scene(rep.data, vcam)
    rgb = renderer.render().copy()
    seg.update_scene(rep.data, vcam)
    s = seg.render()
    mask = (s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(s[..., 0], list(hand_geoms))
    return FG.flat_background(rgb, s, set(hand_geoms)), mask


def phone_frame(meta: dict, t: int) -> np.ndarray:
    ep = json.loads((R.EPISODES_DIR / meta["source_episode"]).read_text())
    off = R.find_episode_offset(np.array(meta["image"]), np.array(ep["image"]))
    cap = cv2.VideoCapture(str(R.EPISODES_DIR / meta["source_clip"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, off + t)
    ok, frame = cap.read()
    cap.release()
    R.draw_landmarks(frame, np.array(meta["image"])[t], W.IMAGE_W, W.IMAGE_H)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", type=Path, required=True)
    ap.add_argument("--trials", nargs="*", default=["ep_047_t01", "ep_026_t01"])
    args = ap.parse_args()

    rep = Replayer(str(R.SCENE), width=RW, height=RH)
    hand_geoms = R.style_scene(rep.model)
    FG.hide_scene_props(rep.model)
    rep.model.vis.global_.fovy = FOVY_DEG
    renderer = mujoco.Renderer(rep.model, height=RH, width=RW)
    seg = mujoco.Renderer(rep.model, height=RH, width=RW)
    seg.enable_segmentation_rendering()

    # all 63 trials: angular and pixel error
    all_ang, all_pix = [], []
    for f in sorted(glob.glob(str(R.QPOS_DIR / "*.json"))):
        res = measure(rep, Path(f).stem)
        all_ang.append(res["ang"])
        all_pix.append(res["pix"])
    a, p = np.concatenate(all_ang), np.concatenate(all_pix)
    print(f"all 63 trials, {len(a)} frames: palm-normal angle sim vs landmarks mean {a.mean():.1f} deg, "
          f"median {np.median(a):.1f}, p95 {np.percentile(a, 95):.1f}, max {a.max():.1f}; "
          f"wrist+knuckle reprojection mean {p.mean():.1f} px, median {np.median(p):.1f}, "
          f"p95 {np.percentile(p, 95):.1f}, max {p.max():.1f} (of 1280x720)")

    args.stills.mkdir(parents=True, exist_ok=True)
    for trial in args.trials:
        res = measure(rep, trial)
        meta = res["meta"]
        g, r = meta["grasp_frame"], meta["release_frame"]
        frames = np.linspace(max(0, g - 10), min(len(res["ang"]) - 1, r + 10), 6).astype(int)
        print(f"{trial}: angle mean {res['ang'].mean():.1f} max {res['ang'].max():.1f} deg, "
              f"reprojection mean {res['pix'].mean():.1f} max {res['pix'].max():.1f} px; stills at {list(frames)}")
        # one fixed crop per trial around the hand, never above the face line
        lm = np.array(meta["image"])[frames][:, :, :2] * [W.IMAGE_W, W.IMAGE_H]
        x0, x1 = int(max(0, lm[..., 0].min() - 120)), int(min(W.IMAGE_W, lm[..., 0].max() + 120))
        y0, y1 = int(max(R.FACE_LINE_Y, lm[..., 1].min() - 80)), int(min(W.IMAGE_H, lm[..., 1].max() + 120))
        for t in frames:
            real = phone_frame(meta, t)
            sim, mask = render_phone_view(rep, renderer, seg, hand_geoms, res, t)
            sim_full = cv2.resize(sim, (W.IMAGE_W, W.IMAGE_H), interpolation=cv2.INTER_NEAREST)
            mask_full = cv2.resize(mask.astype(np.uint8), (W.IMAGE_W, W.IMAGE_H),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            over = real.copy()
            over[mask_full] = (0.45 * over[mask_full] + 0.55 * np.array([0, 90, 255])).astype(np.uint8)
            tiles = [im[y0:y1, x0:x1] for im in (real, sim_full, over)]
            sheet = np.hstack(tiles)
            for txt, x in (("real + MediaPipe", 8), ("sim, phone-view camera", tiles[0].shape[1] + 8),
                           ("overlay", 2 * tiles[0].shape[1] + 8)):
                cv2.putText(sheet, txt, (x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(sheet, txt, (x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(sheet, f"{trial} f{t}  palm angle {res['ang'][t]:.0f} deg, knuckles {res['pix'][t]:.0f} px",
                        (8, sheet.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imwrite(str(args.stills / f"{trial}_f{t:03d}.png"), sheet)


if __name__ == "__main__":
    main()
