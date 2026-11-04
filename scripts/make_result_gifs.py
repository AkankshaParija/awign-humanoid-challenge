"""Render the README's result GIFs: phone footage (with MediaPipe landmarks
drawn on top, cropped around the hand) side by side with the MuJoCo replay,
frame-synced, with a text overlay.

What the MuJoCo side shows (kinematic attach, NOT physics-based carrying):
  1. The cup starts resting on the table, at the hand's own fingertip
     position at the annotated grasp frame (same seeding as replay.py).
  2. Once >=2 fingertips touch it, the cup is ATTACHED: it is held fixed in
     the palm frame, at the offset chosen by attach_fit.fit_offset to
     minimise hand-cup penetration over the attached frames (same as
     replay.py's kinematic mode). Kinematic attach does not resolve
     collisions; the remaining penetration is measured and printed.
  3. At the annotated release frame it is detached, falls straight down
     (simple free fall, drawn kinematically) and rests upright on the table.
     No reported numbers are computed from these clips except the
     per-clip penetration and visibility checks printed below.

Scene staging: there is no camera-to-table calibration, so the table and
sticky note are placed for a readable shot, not at measured positions. The
table top is put under the cup's starting position; the sticky note is drawn
where the cup is released. The clip does not show the cup landing on the
real sticky note location.

Camera: one fixed set of camera settings for every clip (distance,
azimuth, elevation). Only the look-at point moves: it follows the palm
(lightly smoothed) and is pulled halfway towards the cup.

Automatic checks, printed per clip:
  * sim visibility: palm + 5 fingertips projected inside the image, and
    >=2% of pixels change when the hand is hidden (must pass >=95%);
  * trimming: the clip is cut to the longest run of frames around the
    attached span where the projected hand points stay inside the sim
    image AND all 21 hand landmarks stay inside the phone crop, so the hand
    never leaves either panel. A trial whose attached span itself fails is
    rejected;
  * no face: the phone camera is static, and my chin sits at y ~110-140 px
    (of 720) in every frame, read off the footage by hand. The phone crop
    never starts above FACE_LINE_Y = 200 px, and frames where any hand
    landmark goes above that line are trimmed. (OpenCV's Haar face
    detectors were tried as an extra check but can't find my partly
    cropped face even in the top band of the frame where it is, so they
    prove nothing and aren't used.);
  * the carried cup never sinks more than 1 cm below the staged table top.

Saves stills (grasp frame, attach frame, mid-carry, release) to --stills
(default outputs/gif_stills, gitignored).

    python scripts/make_result_gifs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import Replayer, fingertip_contacts  # noqa: E402
import wrist_traj  # noqa: E402
import attach_fit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CUP_HEIGHT, STICKY_NOTE_THICKNESS, TABLE_HEIGHT  # noqa: E402

SCENE = ROOT / "mujoco_menagerie/shadow_hand/scene_cup.xml"
QPOS_DIR = ROOT / "outputs/qpos"
TRIALS_DIR = ROOT / "data/trials"
EPISODES_DIR = ROOT / "data/episodes"
GIF_DIR = ROOT / "docs/gifs"

# --- one look for every clip ---
HAND_RGBA = [0.95, 0.33, 0.0, 1.0]    # saturated orange: high contrast on light tan
CUP_RGBA = [0.15, 0.35, 0.85, 1.0]    # blue cup, so it doesn't blend into an orange hand
TABLE_RGBA = [0.76, 0.63, 0.47, 1.0]  # tan
TABLE_HALF_XY = (0.60, 0.60)          # bigger table top so a carried cup stays over it
MJ_W, MJ_H = 560, 420                 # mujoco panel
PHONE_SIDE = 420                      # phone panel: square crop, same height
CAM_DISTANCE = 0.55                   # metres from the look-at point
CAM_AZIMUTH = 240                     # degrees
CAM_ELEVATION = -60                   # degrees. Chosen from grasp-frame stills of
                                      # side (-10), three-quarter (-30) and above (-60)
                                      # views: with the knuckle line vertical, fingers
                                      # bend in the horizontal plane, so the curl shows
                                      # from above; az 150 looked along the palm's edge
LOOKAT_SMOOTH = 0.3                   # EMA weight on the new palm position per frame

# Hand root pose: scripts/wrist_traj.py (approximate wrist trajectory from
# the image landmarks, rotated so the grasp's knuckle line is vertical).
# Set by --wrist; "image" is the default, "rotate-only" is the old
# rotate-in-place behaviour, for comparison.
WRIST_MODE = "image"
# Hand orientation: "knuckle" = per-trial knuckle-line-vertical rotation
# (main); "palm" = palm frame from the image landmarks with one fixed
# camera->sim rotation (orient-fix), viewed with PHONE_VIEW_* below.
ORIENT_MODE = "knuckle"
# phone-view camera for --orient palm: the phone's own viewing direction
# (from the measured CAM_TO_SIM), following the palm at a fixed distance so
# the hand is readable
PHONE_VIEW_AZIMUTH, PHONE_VIEW_ELEVATION = wrist_traj.phone_view_angles()
PHONE_VIEW_DISTANCE = 0.5

LEAD = 15        # frames of approach shown before the annotated grasp
AFTER = 15       # frames shown after the annotated release
GIF_STRIDE = 2   # 30 fps source -> ~15 fps GIF
GIF_SCALE = 0.8  # shrink the side-by-side canvas to keep each GIF under 5 MB
GIF_COLORS = 128

# visibility check thresholds
EDGE_MARGIN_PX = 4          # projected points must be at least this far inside
DIFF_LEVEL = 25             # a pixel "differs" if any channel changes by more than this
MIN_HAND_PIXEL_FRAC = 0.02  # >=2% of the panel must change when the hand is hidden
PASS_RATE_REQUIRED = 0.95
FACE_LINE_Y = 200           # px (of 720): nothing above this line is ever shown
MAX_PEN_MM_GIF = 5.0        # clips must have max hand-cup penetration below this

# Chosen by rendering all 32 to_target trials (python scripts/make_result_gifs.py
# --trials ...) with every check below, plus the mean finger flexion at the
# annotated grasp frame (non-thumb J1-J3, fraction of joint range).
# ep_047_t01 is the only trial passing every check including max
# penetration < 5 mm; it is a side pinch, not a full wrap (flexion 0.29:
# index and middle nearly straight). ep_026_t01 is the typical case: the
# only clearly curled trial (flexion 0.48, the to_target median) that
# passes every check except penetration (mean 9.3, max 15.4 mm), so the
# 5 mm limit is switched off for it. Clean/typical labels are in the README captions.
TRIALS = {
    "ep_047_t01": {"failure": False, "pen_limit": True},    # clean case, side pinch
    "ep_026_t01": {"failure": False, "pen_limit": False},   # typical case, residual overlap
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17), (0, 17),
]


# ---------------------------------------------------------------- scene look

def style_scene(model) -> list[int]:
    """Recolour hand and cup, hide the forearm, enlarge the table, boost the
    headlight. Returns the geom ids of the visible hand (used to hide it for
    the visibility check)."""
    hand_geoms = []
    cup_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    for i in range(model.ngeom):
        b = model.geom_bodyid[i]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name in ("rh_forearm", "rh_wrist"):
            model.geom_rgba[i] = [0, 0, 0, 0]  # hidden: the oversized "cylinder"
        elif name.startswith("rh_"):
            model.geom_rgba[i] = HAND_RGBA
            # the Shadow Hand geoms use materials; rgba only wins without one
            model.geom_matid[i] = -1
            hand_geoms.append(i)
        elif b == cup_body:
            model.geom_rgba[i] = CUP_RGBA
            model.geom_matid[i] = -1
    table_top = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    model.geom_size[table_top][:2] = TABLE_HALF_XY
    model.vis.headlight.ambient[:] = [0.35, 0.35, 0.35]
    model.vis.headlight.diffuse[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.specular[:] = [0.1, 0.1, 0.1]
    # explicit tan so the table doesn't wash out to white under the lights
    table_mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "table_mat")
    model.mat_rgba[table_mat] = TABLE_RGBA
    return hand_geoms


# ------------------------------------------------------------ phone side

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
             margin: float = 0.25) -> tuple[int, int, int, int]:
    """Union bounding box of the hand across every frame in the clip, plus a
    margin, so the crop stays fixed (no jitter) and shows the cup moving.
    Not forced square: a square crop of a wide transport grows upwards into
    the face, so the crop is letterboxed into the panel instead (see
    fit_panel)."""
    xs = image_lm[:, :, 0].flatten() * frame_w
    ys = image_lm[:, :, 1].flatten() * frame_h
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, x1 = max(0, int(x0 - mx)), min(frame_w, int(x1 + mx))
    y0, y1 = max(0, int(y0 - my)), min(frame_h, int(y1 + 2 * my))  # more room below: the cup/table
    y0 = max(y0, FACE_LINE_Y)       # never show anything above the face line
    return x0, y0, x1, y1


def fit_panel(img: np.ndarray, side: int) -> np.ndarray:
    """Scale img to fit a side x side panel, padding the rest dark grey."""
    h, w = img.shape[:2]
    k = side / max(h, w)
    img = cv2.resize(img, (int(w * k), int(h * k)))
    out = np.full((side, side, 3), 30, np.uint8)
    y, x = (side - img.shape[0]) // 2, (side - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    return out


def draw_landmarks(frame: np.ndarray, lm: np.ndarray, frame_w: int, frame_h: int) -> None:
    pts = [(int(x * frame_w), int(y * frame_h)) for x, y, _ in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 220, 80), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 4, (40, 220, 255), -1, cv2.LINE_AA)


def phone_landmarks_ok(trial_meta: dict) -> np.ndarray:
    """Per trial frame: every hand landmark is inside the video frame and
    below the face line (so the crop can show the whole hand, no face)."""
    lm = np.array(trial_meta["image"])
    x = lm[:, :, 0] * wrist_traj.IMAGE_W
    y = lm[:, :, 1] * wrist_traj.IMAGE_H
    return ((x.min(1) >= 0) & (x.max(1) < wrist_traj.IMAGE_W)
            & (y.min(1) >= FACE_LINE_Y + 10) & (y.max(1) < wrist_traj.IMAGE_H))


def phone_frames(trial_meta: dict, lo: int, hi: int) -> tuple[list[np.ndarray], dict]:
    ep = json.loads((EPISODES_DIR / trial_meta["source_episode"]).read_text())
    trial_image = np.array(trial_meta["image"])
    offset = find_episode_offset(trial_image, np.array(ep["image"]))
    cap = cv2.VideoCapture(str(EPISODES_DIR / trial_meta["source_clip"]))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, y0, x1, y1 = crop_box(trial_image[lo:hi], frame_w, frame_h)
    out, hand_in_crop = [], True
    for t in range(lo, hi):
        cap.set(cv2.CAP_PROP_POS_FRAMES, offset + t)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((frame_h, frame_w, 3), np.uint8)
        px = trial_image[t, :, :2] * [frame_w, frame_h]
        hand_in_crop &= bool(np.all((px[:, 0] >= x0) & (px[:, 0] < x1)
                                    & (px[:, 1] >= y0) & (px[:, 1] < y1)))
        draw_landmarks(frame, trial_image[t], frame_w, frame_h)
        out.append(fit_panel(frame[y0:y1, x0:x1], PHONE_SIDE))
    cap.release()
    checks = {"crop": (x0, y0, x1, y1),
              "crop_below_face_line": y0 >= FACE_LINE_Y, "hand_in_crop": hand_in_crop}
    return out, checks


# ------------------------------------------------------ visibility check

def project(scene, point: np.ndarray, w: int, h: int) -> tuple[float, float]:
    """World point -> pixel (u, v) using the camera MuJoCo just rendered
    with (renderer.scene.camera[0]: position, forward/up axes, frustum)."""
    cam = scene.camera[0]
    # MuJoCo keeps a left- and right-eye camera (offset by the stereo IPD);
    # a mono render uses their average, so average the positions too
    pos = (np.array(scene.camera[0].pos) + np.array(scene.camera[1].pos)) / 2
    fwd, up = np.array(cam.forward), np.array(cam.up)
    right = np.cross(fwd, up)
    rel = point - pos
    z = rel @ fwd                       # depth along the view direction
    if z <= 1e-6:
        return -1.0, -1.0               # behind the camera: counts as outside
    # point on the near plane, in the same units as the frustum extents
    x = (rel @ right) * cam.frustum_near / z
    y = (rel @ up) * cam.frustum_near / z
    # frustum_width is left at 0 by mjv_updateScene (it means "derive from
    # the image aspect ratio"), so compute the half-width the same way
    half_w = cam.frustum_width or (cam.frustum_top - cam.frustum_bottom) / 2 * w / h
    u = (x - cam.frustum_center + half_w) / (2 * half_w) * w
    v = (cam.frustum_top - y) / (cam.frustum_top - cam.frustum_bottom) * h
    return u, v


def points_inside(rep: Replayer, pts: list[np.ndarray]) -> bool:
    for p in pts:
        u, v = project(rep.renderer.scene, p, MJ_W, MJ_H)
        if not (EDGE_MARGIN_PX <= u <= MJ_W - EDGE_MARGIN_PX
                and EDGE_MARGIN_PX <= v <= MJ_H - EDGE_MARGIN_PX):
            return False
    return True


def hand_pixel_frac(rep: Replayer, with_hand: np.ndarray, hand_geoms: list[int]) -> float:
    """Fraction of pixels that change when the hand is made invisible."""
    alpha = rep.model.geom_rgba[hand_geoms, 3].copy()
    rep.model.geom_rgba[hand_geoms, 3] = 0
    rep.renderer.update_scene(rep.data, rep.cam)
    without = rep.renderer.render().astype(int)
    rep.model.geom_rgba[hand_geoms, 3] = alpha
    diff = np.abs(with_hand.astype(int) - without).max(axis=2) > DIFF_LEVEL
    return float(diff.mean())


# ------------------------------------------------------------- rendering

OVERLAY_SCALE = 0.85        # large enough to read once the GIF is scaled down
OVERLAY_MAX_LINES = 3


def overlay(canvas: np.ndarray, lines: list[tuple[str, tuple]]) -> None:
    """At most 3 short lines, top-left, dark outline under light text."""
    assert len(lines) <= OVERLAY_MAX_LINES, f"overlay has {len(lines)} lines"
    y = 34
    for text, color in lines:
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_SCALE,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_SCALE,
                    color, 2, cv2.LINE_AA)
        y += 34


def render_trial(rep: Replayer, trial: str, hand_geoms: list[int],
                 table_body: int, sticky_body: int, palm_body: int) -> dict:
    meta_t = json.loads((TRIALS_DIR / f"{trial}.json").read_text())
    grasp_f, release_f = meta_t["grasp_frame"], meta_t["release_frame"]
    raw = json.loads((QPOS_DIR / f"{trial}.json").read_text())
    if ORIENT_MODE == "palm":
        rec = wrist_traj.to_sim_palm(rep, raw, meta_t)
    else:
        rec = wrist_traj.to_sim(rep, raw, meta_t, translate=WRIST_MODE == "image")
    n_total = len(rec["qpos"])
    lo = max(0, grasp_f - LEAD)
    hi = min(n_total, release_f + AFTER + 1)

    qpos = np.array(rec["qpos"])
    qadr = rep.scene_qpos_adr(rec)   # by joint name, see replay.py
    wrist_pos = np.array(rec["wrist_pos"])
    wrist_quat = np.array(rec["wrist_quat"])
    hand_free_adr = rep._hand_free_adr()
    fps = rec.get("fps", 30.0)
    cq = rep.cup_qadr
    geoms = attach_fit.HandCupGeoms(rep.model)

    def set_hand(t: int) -> None:
        rep.data.qpos[hand_free_adr:hand_free_adr + 3] = wrist_pos[t]
        rep.data.qpos[hand_free_adr + 3:hand_free_adr + 7] = wrist_quat[t]
        rep.data.qpos[qadr] = qpos[t]

    # cup seeded at the fingertip centroid at the annotated grasp frame
    # (same as replay.py); the table top is staged right under it
    cup_start = rep.tip_centroid_at_frame(rec, grasp_f)
    cup_rest_z = cup_start[2]
    table_top_z = cup_rest_z - CUP_HEIGHT / 2 - 0.005
    rep.model.body_pos[table_body] = [cup_start[0], cup_start[1],
                                      table_top_z - TABLE_HEIGHT / 2]
    rep.model.body_pos[sticky_body] = [0, 0, -10]  # hidden until we know the release spot

    def reset_cup() -> None:
        mujoco.mj_resetData(rep.model, rep.data)
        rep.data.qpos[cq:cq + 3] = cup_start
        rep.data.qpos[cq + 3:cq + 7] = [1, 0, 0, 0]

    # find the attach frame (first frame >=2 fingertips touch the static
    # cup, searched from frame 0 exactly like replay.py, so the fitted
    # offset is identical) and fit the palm-frame offset over the attached frames
    reset_cup()
    attach_f, fit = None, None
    for t in range(0, release_f):
        set_hand(t)
        mujoco.mj_forward(rep.model, rep.data)
        if fingertip_contacts(rep.model, rep.data, rep.tips, rep.cup_body) >= 2:
            attach_f = t
            start_p, start_q = attach_fit.cup_offset_now(rep.data, geoms, cq)
            fit = attach_fit.fit_offset(rep.model, rep.data, geoms, cq, set_hand,
                                        list(range(t, release_f)), start_p, start_q)
            break

    rep.cam.distance = CAM_DISTANCE
    rep.cam.azimuth = CAM_AZIMUTH if ORIENT_MODE == "knuckle" else PHONE_VIEW_AZIMUTH
    rep.cam.elevation = CAM_ELEVATION if ORIENT_MODE == "knuckle" else PHONE_VIEW_ELEVATION
    if ORIENT_MODE == "palm":
        rep.cam.distance = PHONE_VIEW_DISTANCE

    def simulate():
        """Yields (t, state) with the scene posed for frame t."""
        state = "on table"          # "on table" -> "ATTACHED" -> "released"
        drop_z, drop_t = None, None
        for t in range(lo, hi):
            set_hand(t)
            mujoco.mj_kinematics(rep.model, rep.data)
            if state == "on table" and attach_f is not None and t >= attach_f:
                state = "ATTACHED"
            elif state == "ATTACHED" and t >= release_f:
                # detach at the annotated release frame; the cup falls
                # straight down (drawn kinematically) and rests upright
                state = "released"
                drop_z, drop_t = float(rep.data.qpos[cq + 2]), t
                rep.data.qpos[cq + 3:cq + 7] = [1, 0, 0, 0]
            if state == "ATTACHED":
                attach_fit.place_cup_in_palm(rep.model, rep.data, geoms, cq,
                                             fit["offset_pos"], fit["offset_quat"])
            elif state == "released":
                dt = (t - drop_t) / fps
                rep.data.qpos[cq + 2] = max(cup_rest_z, drop_z - 0.5 * 9.81 * dt * dt)
            mujoco.mj_forward(rep.model, rep.data)
            yield t, state

    # pass 1: find where the cup is released, to stage the sticky note there
    reset_cup()
    for _ in simulate():
        pass
    release_xy = rep.data.qpos[cq:cq + 2].copy()
    rep.model.body_pos[sticky_body] = [release_xy[0], release_xy[1],
                                       table_top_z + STICKY_NOTE_THICKNESS / 2]

    # pass 2: render every frame of the full window and run the sim checks
    reset_cup()
    frames = []
    lookat = None
    for t, state in simulate():
        palm = rep.data.xpos[palm_body].copy()
        cup = rep.data.qpos[cq:cq + 3].copy()
        target = 0.5 * (palm + cup)      # follow the palm, keep the cup in frame too
        lookat = target if lookat is None else (
            (1 - LOOKAT_SMOOTH) * lookat + LOOKAT_SMOOTH * target)
        rep.cam.lookat[:] = lookat
        rep.renderer.update_scene(rep.data, rep.cam)
        img = rep.renderer.render().copy()
        frames.append({
            "t": t, "state": state, "img": cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            "inside": points_inside(rep, [palm] + [rep.data.xpos[i] for i in sorted(rep.tips)]),
            "pixels": hand_pixel_frac(rep, img, hand_geoms) >= MIN_HAND_PIXEL_FRAC,
            "pen_mm": geoms.penetration(rep.data) * 1000,
            "tips": geoms.tips_touching(rep.data),
            "cup_in_table": state == "ATTACHED" and cup[2] < cup_rest_z - 0.01,
        })

    # trim: longest run of frames where the hand stays inside BOTH panels,
    # grown outwards from the attached span (which must pass entirely)
    phone_ok = phone_landmarks_ok(meta_t)
    ok = [f["inside"] and bool(phone_ok[f["t"]]) for f in frames]
    core = [i for i, f in enumerate(frames) if f["state"] == "ATTACHED"] or \
           [i for i, f in enumerate(frames) if grasp_f <= f["t"] < release_f]
    a, b = core[0], core[-1]
    core_ok = all(ok[a:b + 1])
    while a > 0 and ok[a - 1]:
        a -= 1
    while b < len(frames) - 1 and ok[b + 1]:
        b += 1
    frames = frames[a:b + 1]
    lo_t, hi_t = frames[0]["t"], frames[-1]["t"] + 1

    # --- compose side by side + overlay ---
    phones, phone_checks = phone_frames(meta_t, lo_t, hi_t)
    out_frames = []
    for pf, f in zip(phones, frames):
        canvas = np.hstack([pf, f["img"]])
        t = f["t"]
        # The phase comes from the sim's attach state, not from the annotated
        # grasp frame: replay.py's rule can attach before the annotated grasp
        # (the fingertips already touch the seeded cup), and the overlay must
        # never say "approach" and ATTACHED at once.
        # ATTACHED comes from the sim's attach state, not the annotated grasp
        # frame (replay.py's rule can attach before it), and may only appear
        # inside [attach frame, annotated release frame)
        attached = attach_f is not None and attach_f <= t < release_f
        assert (f["state"] == "ATTACHED") == attached, \
            f"{trial} frame {t}: state {f['state']} but attached range is [{attach_f}, {release_f})"
        # 3 short lines max; the full description lives in the README caption
        lines = [(f"{trial}  frame {t}", (240, 240, 240))]
        if attached:
            lines.append(("ATTACHED (kinematic)", (0, 220, 0)))
            lines.append((f"penetration {f['pen_mm']:.1f} mm", (240, 240, 240)))
        overlay(canvas, lines)
        out_frames.append(canvas)

    att = [f for f in frames if f["state"] == "ATTACHED"]
    pens = np.array([f["pen_mm"] for f in att]) if att else np.zeros(1)
    return {"frames": out_frames, "ts": [f["t"] for f in frames],
            "states": [f["state"] for f in frames], "lo": lo_t, "hi": hi_t,
            "grasp_f": grasp_f, "attach_f": attach_f, "release_f": release_f,
            "core_ok": core_ok,
            "pass_rate": float(np.mean([f["inside"] and f["pixels"] for f in frames])),
            "inside_all": all(f["inside"] for f in frames),
            "pen_mean_mm": float(pens.mean()), "pen_max_mm": float(pens.max()),
            "tips2_frac": float(np.mean([f["tips"] >= 2 for f in att])) if att else 0.0,
            "fit_shift_mm": fit["shift_mm"] if fit else 0.0,
            "cup_in_table_frames": int(sum(f["cup_in_table"] for f in frames)),
            "carried_cm": float(np.linalg.norm(release_xy - cup_start[:2]) * 100),
            "cup_dxy_cm": ((release_xy - cup_start[:2]) * 100).tolist(),
            **phone_checks, "fps": fps}


def save_gif(frames_bgr: list[np.ndarray], dst: Path, fps: float) -> None:
    """Write frames straight to a GIF (no mp4 in between: video compression
    noise makes GIFs much bigger). One shared palette for all frames."""
    imgs = [Image.fromarray(cv2.cvtColor(
                cv2.resize(f, None, fx=GIF_SCALE, fy=GIF_SCALE,
                           interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            for f in frames_bgr[::GIF_STRIDE]]
    # build the palette from 6 frames spread over the clip, so colours that
    # only appear in some frames (green ATTACHED text, release overlay) keep
    # their own palette entries
    picks = [imgs[i] for i in np.linspace(0, len(imgs) - 1, 6).astype(int)]
    sheet = Image.new("RGB", (picks[0].width * len(picks), picks[0].height))
    for k, im in enumerate(picks):
        sheet.paste(im, (k * im.width, 0))
    palette = sheet.convert("P", palette=Image.ADAPTIVE, colors=GIF_COLORS)
    # no dithering: dither noise looks like speckle and makes the file bigger
    imgs = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in imgs]
    dst.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(dst, save_all=True, append_images=imgs[1:],
                 duration=int(1000 * GIF_STRIDE / fps), loop=0, optimize=True)


def save_stills(res: dict, trial: str, out_dir: Path) -> None:
    """Stills at the annotated grasp frame, the attach frame, mid-carry and
    the release frame (whichever are inside the trimmed clip)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = res["ts"]
    att = [i for i, s in enumerate(res["states"]) if s == "ATTACHED"]
    picks = {"1_grasp": res["grasp_f"], "2_attach": res["attach_f"],
             "3_carry": ts[att[len(att) // 2]] if att else None,
             "4_release": res["release_f"]}
    for name, t in picks.items():
        if t is not None and t in ts:
            cv2.imwrite(str(out_dir / f"{trial}_{name}.png"), res["frames"][ts.index(t)])


def main() -> None:
    global WRIST_MODE, ORIENT_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", type=Path, default=ROOT / "outputs/gif_stills",
                    help="scratch folder for 3 still frames per clip (not committed)")
    ap.add_argument("--trials", nargs="*", help="override the trial list (for trying candidates)")
    ap.add_argument("--gif-dir", type=Path, default=GIF_DIR)
    ap.add_argument("--wrist", choices=["image", "rotate-only"], default=WRIST_MODE)
    ap.add_argument("--orient", choices=["knuckle", "palm"], default=ORIENT_MODE)
    args = ap.parse_args()
    WRIST_MODE = args.wrist
    ORIENT_MODE = args.orient

    rep = Replayer(str(SCENE), mode="kinematic", width=MJ_W, height=MJ_H)
    hand_geoms = style_scene(rep.model)
    body = lambda n: mujoco.mj_name2id(rep.model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731

    trials = args.trials or list(TRIALS)
    for trial in trials:
        TRIALS.setdefault(trial, {"failure": False, "pen_limit": True})
        res = render_trial(rep, trial, hand_geoms, body("table"),
                           body("sticky_note"), body("rh_palm"))
        gif = args.gif_dir / f"{trial}.gif"
        save_gif(res["frames"], gif, res["fps"])
        save_stills(res, trial, args.stills)
        good = (res["core_ok"] and res["pass_rate"] >= PASS_RATE_REQUIRED
                and res["inside_all"]
                and (res["pen_max_mm"] < MAX_PEN_MM_GIF or not TRIALS[trial].get("pen_limit", True))
                and res["crop_below_face_line"]
                and res["hand_in_crop"] and res["cup_in_table_frames"] == 0)
        print(f"{trial}: {'OK' if good else 'FAIL'}  frames {res['lo']}-{res['hi'] - 1} "
              f"({len(res['frames'])}, attach {res['attach_f']}, release {res['release_f']}), "
              f"visible {res['pass_rate'] * 100:.1f}%, hand in sim image all frames "
              f"{res['inside_all']}, attached span kept whole {res['core_ok']}, "
              f"penetration mean {res['pen_mean_mm']:.1f} max {res['pen_max_mm']:.1f}mm, "
              f">=2 tips within 5mm {res['tips2_frac'] * 100:.0f}% of attached frames, "
              f"cup in table {res['cup_in_table_frames']} frames, "
              f"crop top y={res['crop'][1]} "
              f"(>= {FACE_LINE_Y}: {res['crop_below_face_line']}), hand in crop {res['hand_in_crop']}, "
              f"carried {res['carried_cm']:.1f}cm, {gif.name} {gif.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
