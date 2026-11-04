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


# ---------------------------------------------------------------------------
# Orientation from data (orient-fix): the palm frame from MediaPipe's world
# landmarks, one fixed camera->sim rotation for all clips, one fixed
# MediaPipe-palm -> Shadow-palm offset. No per-clip heuristic.
# ---------------------------------------------------------------------------

# OpenCV camera axes (x right, y down, z forward) -> sim axes (x right,
# y forward, z up), ONE fixed rotation for every clip. Sim "up" is the real
# table normal, measured from the sticky note (a known 7.5 cm square) by
# scripts/measure_table_plane.py: PnP in 6 clips, 0.6-2.1 px reprojection,
# the phone looks 17.5 deg below horizontal. (A level-phone mapping was tried
# first; with it the replayed hand drifts down through a level sim table.)
TABLE_NORMAL_CAM = np.array([0.0548, -0.9519, -0.3015])


def _cam_to_sim(up_cam: np.ndarray) -> np.ndarray:
    z = up_cam / np.linalg.norm(up_cam)
    x = np.array([1.0, 0.0, 0.0])
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])           # rows = sim axes in camera coords


CAM_TO_SIM = _cam_to_sim(TABLE_NORMAL_CAM)


def phone_view_angles() -> tuple[float, float]:
    """MuJoCo free-camera azimuth/elevation (deg) that look the way the phone does."""
    f = CAM_TO_SIM @ np.array([0.0, 0.0, 1.0])        # camera forward, in sim axes
    return float(np.degrees(np.arctan2(f[1], f[0]))), float(np.degrees(np.arcsin(f[2])))
INDEX_MCP_, PINKY_MCP_ = 5, 17
TIPS_ = [8, 12, 16, 20]
ORIENT_SMOOTH_ALPHA = 0.5        # slerp weight per frame, run forwards then backwards


def palm_frame(wrist, index_mcp, middle_mcp, pinky_mcp) -> np.ndarray:
    """3x3, columns = palm axes: y = wrist -> middle MCP, x = pinky MCP ->
    index MCP made orthogonal to y, z = x cross y."""
    y = middle_mcp - wrist
    y = y / np.linalg.norm(y)
    x = index_mcp - pinky_mcp
    x = x - y * (x @ y)
    x = x / np.linalg.norm(x)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def model_palm_frame(rep) -> tuple[np.ndarray, np.ndarray]:
    """The Shadow Hand's palm frame built the same way from its own wrist
    and knuckles, with the root at identity and all joints at 0. Returns
    (M in root coordinates, the palm-side direction in root coordinates:
    the way the middle fingertip moves when the finger flexes)."""
    m, d = rep.model, rep.data
    body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    ha = rep._hand_free_adr()
    mujoco.mj_resetData(m, d)
    d.qpos[ha:ha + 3] = 0
    d.qpos[ha + 3:ha + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    M = palm_frame(d.xpos[body("rh_wrist")], d.xpos[body("rh_ffknuckle")],
                   d.xpos[body("rh_mfknuckle")], d.xpos[body("rh_lfknuckle")])
    tip0 = d.xpos[body("rh_mfdistal")].copy()
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "rh_MFJ3")
    d.qpos[m.jnt_qposadr[j]] = 1.2
    mujoco.mj_forward(m, d)
    palm_side = d.xpos[body("rh_mfdistal")] - tip0
    mujoco.mj_resetData(m, d)
    return M, palm_side / np.linalg.norm(palm_side)


def handedness_check(rep, world_trials: list[np.ndarray]) -> dict:
    """Is MediaPipe's palm frame the same handedness as the Shadow Hand's?
    In both, z = x cross y; check that it points to the palm side (where the
    fingers curl) in both. For MediaPipe, the palm side is where the
    fingertips sit relative to the palm plane, averaged over all frames."""
    M, palm_side = model_palm_frame(rep)
    model_sign = float(np.sign(M[:, 2] @ palm_side))
    dots = []
    for w in world_trials:
        for t in range(len(w)):
            P = palm_frame(w[t, WRIST], w[t, INDEX_MCP_], w[t, MIDDLE_MCP], w[t, PINKY_MCP_])
            centre = w[t, [WRIST, INDEX_MCP_, MIDDLE_MCP, PINKY_MCP_]].mean(axis=0)
            dots.append(float((w[t, TIPS_].mean(axis=0) - centre) @ P[:, 2]))
    dots = np.array(dots)
    mp_sign = float(np.sign(np.median(dots)))
    # thumb side: the thumb tip should sit on the index side (+x) in both
    thumb_x = [float((w[t, 4] - w[t, WRIST]) @ palm_frame(w[t, WRIST], w[t, INDEX_MCP_],
               w[t, MIDDLE_MCP], w[t, PINKY_MCP_])[:, 0]) for w in world_trials for t in range(len(w))]
    return {"model_z_toward_palm": model_sign, "mediapipe_z_toward_palm": mp_sign,
            "mediapipe_frames_palm_side_agree": float(np.mean(np.sign(dots) == mp_sign)),
            "thumb_on_index_side_frac": float(np.mean(np.array(thumb_x) > 0)),
            "mirrored": model_sign != mp_sign}


def slerp_smooth(quats: np.ndarray, alpha: float = ORIENT_SMOOTH_ALPHA) -> np.ndarray:
    """Light zero-lag smoothing on the rotation itself: exponential slerp
    forwards, then backwards over the result."""
    def one_pass(q):
        out = q.copy()
        for t in range(1, len(q)):
            a, b = out[t - 1], q[t].copy()
            if a @ b < 0:
                b = -b                                    # same hemisphere
            res = np.zeros(4)
            # slerp(a, b, alpha) = a * (a^-1 b)^alpha
            inv = np.zeros(4)
            mujoco.mju_negQuat(inv, a)
            dq = np.zeros(4)
            mujoco.mju_mulQuat(dq, inv, b)
            vel = np.zeros(3)
            mujoco.mju_quat2Vel(vel, dq, 1.0)
            step = np.zeros(4)
            mujoco.mju_axisAngle2Quat(step, vel / (np.linalg.norm(vel) + 1e-12),
                                      np.linalg.norm(vel) * alpha)
            mujoco.mju_mulQuat(res, a, step)
            out[t] = res / np.linalg.norm(res)
        return out
    return one_pass(one_pass(quats)[::-1])[::-1]


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).flatten())
    return q


def landmarks_in_camera(image_lm: np.ndarray) -> np.ndarray:
    """(T,21,3) hand landmarks in the phone camera frame, metres, from the
    IMAGE landmarks: pixel position back-projected at the wrist depth, plus
    MediaPipe's per-landmark relative depth z (relative to the wrist, same
    scale as normalised x). The world landmarks were tried first and are
    not usable here: their palm normal agrees with the image on which way
    the palm faces in only 56% of frames, and a per-frame PnP fit of their
    3D shape to the image does no better (48%, 40 px reprojection error)."""
    z_wrist = wrist_in_camera(image_lm)[:, 2]
    u = image_lm[:, :, 0] * IMAGE_W
    v = image_lm[:, :, 1] * IMAGE_H
    Z = z_wrist[:, None] + image_lm[:, :, 2] * IMAGE_W * z_wrist[:, None] / FOCAL_PX
    return np.stack([(u - IMAGE_W / 2) * Z / FOCAL_PX, (v - IMAGE_H / 2) * Z / FOCAL_PX, Z], axis=2)


def palm_orientation_sim(rep, image_lm: np.ndarray, mirrored: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: (root quaternion in sim axes, raw palm frame in sim axes).
    R_root = CAM_TO_SIM . P_cam(t) . M^T, then slerp-smoothed."""
    M, _ = model_palm_frame(rep)
    flip = np.diag([-1.0, 1.0, 1.0]) if mirrored else np.eye(3)
    pts = landmarks_in_camera(image_lm)
    quats, frames = [], []
    for t in range(len(pts)):
        w = pts[t] @ flip.T                               # global mirror, if needed
        P = CAM_TO_SIM @ palm_frame(w[WRIST], w[INDEX_MCP_], w[MIDDLE_MCP], w[PINKY_MCP_])
        frames.append(P)
        quats.append(mat_to_quat(P @ M.T))
    return slerp_smooth(np.array(quats)), np.array(frames)


def to_sim_palm(rep, rec: dict, trial_meta: dict, mirrored: bool = False) -> dict:
    """Like to_sim, but the hand orientation comes from the palm frame in the
    world landmarks (no knuckle-up heuristic), and translation uses the same
    image-space wrist trajectory rotated by the fixed CAM_TO_SIM."""
    quats, _ = palm_orientation_sim(rep, np.array(trial_meta["image"]), mirrored)
    out = dict(rec)
    out["wrist_quat"] = quats.tolist()
    g = trial_meta["grasp_frame"]
    wrist_sim = wrist_in_camera(np.array(trial_meta["image"])) @ CAM_TO_SIM.T
    shift = GRASP_WRIST_ANCHOR - wrist_sim[g]
    wrist_sim = wrist_sim + shift
    wrist_body = mujoco.mj_name2id(rep.model, mujoco.mjtObj.mjOBJ_BODY, "rh_wrist")
    offset = rep.model.body_pos[wrist_body]
    root = np.empty_like(wrist_sim)
    for t, q in enumerate(quats):
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, q)
        root[t] = wrist_sim[t] - Rq.reshape(3, 3) @ offset
    out["wrist_pos"] = root.tolist()
    out["phone_cam_pos_sim"] = shift.tolist()             # the phone sits at sim origin + shift
    return out
