"""Approximate global wrist trajectory from the image-space landmarks, and
the sim-frame hand pose built from it.

Why: retarget.py's wrist_pos comes from MediaPipe *world* landmarks, which
are centred on the hand, so the hand's real movement across the table is
lost and the sim hand only rotates in place. Here the wrist is put back in
the phone camera's 3D frame with a pinhole model:

  depth  Z = min over two hand bones of  f * L / d
           d = the bone's length in the image, in pixels
           L = its assumed typical real length:
               wrist -> middle-finger MCP   9.5 cm
               index MCP -> little MCP      7.5 cm (knuckle width)
         A bone tilted towards the camera only ever looks SHORTER than it
         is, so each bone gives an upper bound on depth and the less
         foreshortened one (the smaller bound) is the better estimate.
         The two bones are roughly perpendicular, so they rarely both
         point at the camera. Tried and dropped: wrist->MCP alone (far too
         noisy: per-trial depth range median 80 cm, 95th-percentile depth
         1.67 m, because in this camera angle that bone often points
         nearly at the lens), and a foreshortening correction from
         MediaPipe's world landmarks (those turned out not to be aligned
         with the camera here, giving a 0.21 m median depth).
  X = (u - cx) * Z / f,  Y = (v - cy) * Z / f

f is NOT calibrated: it is an assumed typical phone main camera, 26 mm
35mm-equivalent focal length, i.e. f = 1280 * 26/36 = 924 px at the 1280x720
video resolution used for detection (~69 deg horizontal field of view).
The accuracy of this trajectory has not been measured.

The trajectory is then lightly smoothed and, for the sim, rotated once per
trial so that the little->index knuckle line at the annotated grasp frame
points up (same rotation the GIFs used before). The hand's free joint sits at
the forearm origin, ~21 cm behind the wrist, so the root position is set so
that the model's wrist body (rh_wrist) lands on the estimated wrist.

Only the hand's global pose changes. Joint angles (qpos) and the 42 mm
retargeting error are untouched.
"""

from __future__ import annotations

import mujoco
import numpy as np

# --- camera / hand assumptions (stated in the README) ---
IMAGE_W, IMAGE_H = 1280, 720      # resolution the landmarks were detected at
FOCAL_EQUIV_MM = 26.0             # typical phone main camera, 35mm-equivalent
FOCAL_PX = IMAGE_W * FOCAL_EQUIV_MM / 36.0   # 924 px; 36 mm = full-frame width
WRIST_TO_MCP_M = 0.095            # typical adult wrist -> middle MCP, 9-10 cm
KNUCKLE_WIDTH_M = 0.075           # typical adult index MCP -> little MCP
# where the wrist sits at the annotated grasp frame, in the sim. The camera
# frame puts the hand ~0.7 m from the lens, which after rotating lands
# under the floor, so every trial is shifted by one constant offset (a rigid
# move, contact is unaffected). 0.6 m keeps it clear of the floor and of the
# scene's placeholder table (top at 0.30 m).
GRASP_WRIST_ANCHOR = np.array([0.0, 0.0, 0.6])
SMOOTH_WINDOW = 9                 # frames (~0.3 s) for x, y
DEPTH_SMOOTH_WINDOW = 15          # frames (~0.5 s): depth is the noisiest part
# Depth wobble clamp: within one trial the hand stays roughly the same
# distance from the phone (the cup moves sideways across the table, ~20 cm),
# so depth is clipped to the trial's median depth +-8 cm before smoothing.
# This is an assumption about the task, not a measurement.
DEPTH_CLAMP_M = 0.08

WRIST, INDEX_MCP, MIDDLE_MCP, LITTLE_MCP = 0, 5, 9, 17   # MediaPipe landmark indices


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average along axis 0, edges padded with the end values."""
    if window <= 1:
        return x.copy()
    pad = window // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)])
    kernel = np.ones(window) / window
    return np.stack([np.convolve(xp[:, k], kernel, mode="valid")
                     for k in range(x.shape[1])], axis=1)


def wrist_in_camera(image_lm: np.ndarray) -> np.ndarray:
    """(T,3) wrist position in the phone camera frame, metres
    (x right, y down, z away from the camera), smoothed."""
    px = image_lm[:, :, :2] * [IMAGE_W, IMAGE_H]          # normalised -> pixels
    def bone_px(a, b):
        return np.maximum(np.linalg.norm(px[:, a] - px[:, b], axis=1), 1.0)

    z_bone = FOCAL_PX * WRIST_TO_MCP_M / bone_px(WRIST, MIDDLE_MCP)
    z_knuckles = FOCAL_PX * KNUCKLE_WIDTH_M / bone_px(INDEX_MCP, LITTLE_MCP)
    z = np.minimum(z_bone, z_knuckles)
    z_med = np.median(z)
    z = np.clip(z, z_med - DEPTH_CLAMP_M, z_med + DEPTH_CLAMP_M)
    z = moving_average(z[:, None], DEPTH_SMOOTH_WINDOW)[:, 0]
    u, v = px[:, WRIST, 0], px[:, WRIST, 1]
    x = (u - IMAGE_W / 2) * z / FOCAL_PX
    y = (v - IMAGE_H / 2) * z / FOCAL_PX
    return moving_average(np.stack([x, y, z], axis=1), SMOOTH_WINDOW)


def knuckle_up_rotation(rep, rec: dict, grasp_f: int) -> np.ndarray:
    """3x3 rotation (camera axes -> sim axes) that makes the little->index
    knuckle line at the grasp frame point straight up (+z). In a side wrap
    grasp the cup's axis runs along that line. Image "right" stays sim x."""
    rep.tip_centroid_at_frame(rec, grasp_f)               # poses the hand at grasp_f
    body = lambda n: mujoco.mj_name2id(rep.model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    up = rep.data.xpos[body("rh_ffknuckle")] - rep.data.xpos[body("rh_lfknuckle")]
    up = up / np.linalg.norm(up)
    ex = np.array([1.0, 0.0, 0.0])
    ex = ex - up * (ex @ up)
    ex /= np.linalg.norm(ex)
    return np.stack([ex, np.cross(up, ex), up])           # rows = sim axes in camera coords


def to_sim(rep, rec: dict, trial_meta: dict, translate: bool = True) -> dict:
    """Copy of a retargeted record whose wrist_pos/wrist_quat are the hand
    ROOT pose in the sim frame, ready for Replayer.run / tip_centroid_at_frame.

    translate=False keeps the old behaviour (hand-centred wrist_pos, so the
    hand only rotates in place) but in the same rotated frame, for
    before/after comparisons."""
    R = knuckle_up_rotation(rep, rec, trial_meta["grasp_frame"])
    q_R = np.zeros(4)
    mujoco.mju_mat2Quat(q_R, R.flatten())

    quats = []
    for q in rec["wrist_quat"]:
        q_new = np.zeros(4)
        mujoco.mju_mulQuat(q_new, q_R, np.array(q))       # rotate in the world frame
        quats.append(q_new)
    quats = np.array(quats)

    out = dict(rec)
    out["wrist_quat"] = quats.tolist()
    g = trial_meta["grasp_frame"]
    if not translate:
        pos = np.array(rec["wrist_pos"]) @ R.T
        out["wrist_pos"] = (pos - pos[g] + GRASP_WRIST_ANCHOR).tolist()
        return out

    wrist_cam = wrist_in_camera(np.array(trial_meta["image"]))
    wrist_sim = wrist_cam @ R.T
    wrist_sim = wrist_sim - wrist_sim[g] + GRASP_WRIST_ANCHOR
    # put rh_wrist (not the forearm origin, 21 cm back) on the estimated wrist
    wrist_body = mujoco.mj_name2id(rep.model, mujoco.mjtObj.mjOBJ_BODY, "rh_wrist")
    offset = rep.model.body_pos[wrist_body]               # wrist in the forearm (root) frame
    root = np.empty_like(wrist_sim)
    for t, q in enumerate(quats):
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, q)
        root[t] = wrist_sim[t] - Rq.reshape(3, 3) @ offset
    out["wrist_pos"] = root.tolist()
    return out
