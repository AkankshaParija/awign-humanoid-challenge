"""Render the README's result GIFs: phone footage (with MediaPipe landmarks
drawn on top, cropped around the hand) side by side with the MuJoCo replay,
frame-synced, with a text overlay.

What the MuJoCo side shows (kinematic attach, NOT physics-based carrying):
  1. The cup starts resting on the table, at the hand's own fingertip
     position at the annotated grasp frame (same seeding as replay.py).
  2. Once >=2 fingertips touch it, the cup is ATTACHED: it keeps a fixed
     offset from the fingertip centroid and moves with the hand. This is
     the same attach rule as replay.py's kinematic mode. It stays attached
     until the release frame even if contact flickers. The replayed hand is
     not constrained by the table and often dips below it, so while
     attached the cup is not allowed below the table top: it slides along
     the surface instead of sinking through it. The number of frames where
     this kicks in is printed per clip ("cup held at table").
  3. At the annotated release frame it is detached, falls straight down
     (simple free fall, drawn kinematically) and rests upright on the table.
     (replay.py instead releases when contact drops to 0; that is what
     outputs/replay_summary.csv measures. Using the annotated frame here
     makes the clips follow my real grasp->release timing. No reported
     numbers are computed from these clips.)

Scene staging: there is no camera-to-table calibration, so the table and
sticky note are placed for a readable shot, not at measured positions. The
table top is put under the cup's starting position; the sticky note is drawn
where the cup is released. The clip does not show the cup landing on the
real sticky note location.

Camera: one fixed set of camera settings for every clip (distance,
azimuth, elevation). Only the look-at point moves: it follows the palm
(lightly smoothed) and is pulled halfway towards the cup, so both the hand
and the cup stay in frame.

Visibility check, printed per clip: for every frame, (a) palm and the 5
fingertip positions are projected to pixels and must land inside the
image, and (b) the frame is re-rendered with the hand hidden and enough
pixels must differ (i.e. the hand is actually visible, not hidden under
the table or off-screen). A clip below 95% passing frames is reported as FAIL.

Also saves 3 still frames per clip (approach, carrying, after release) to
a scratch folder given by --stills (default outputs/gif_stills, gitignored).

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
CAM_AZIMUTH = 150                     # degrees
CAM_ELEVATION = -50                   # degrees, looking down at the table (best of a
                                      # -20..-70 deg x 0.45..0.75 m sweep on 7 trials)
LOOKAT_SMOOTH = 0.3                   # EMA weight on the new palm position per frame

# Hand root pose: scripts/wrist_traj.py (approximate wrist trajectory from
# the image landmarks, rotated so the grasp's knuckle line is vertical).
# Set by --wrist; "image" is the default, "rotate-only" is the old
# rotate-in-place behaviour, for comparison.
WRIST_MODE = "image"

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

# Chosen by rendering all 32 to_target trials with the image-recovered wrist
# trajectory (python scripts/make_result_gifs.py --trials ...): 18 pass the
# 95% visibility check (2 did with the old rotate-in-place pose). Of those,
# these move the cup furthest towards image-right (where the real sticky
# note was) while rarely needing the table clamp (3%, 6%, 13% of frames).
# No failure clip: every trial reaches >=2 fingertip contacts in kinematic
# mode, so there is no honest "contact never happens" kinematic case.
TRIALS = {
    "ep_018_t02": {"failure": False},
    "ep_027_t02": {"failure": False},
    "ep_015_t01": {"failure": False},
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


def phone_frames(trial_meta: dict, lo: int, hi: int) -> list[np.ndarray]:
    ep = json.loads((EPISODES_DIR / trial_meta["source_episode"]).read_text())
    trial_image = np.array(trial_meta["image"])
    offset = find_episode_offset(trial_image, np.array(ep["image"]))
    cap = cv2.VideoCapture(str(EPISODES_DIR / trial_meta["source_clip"]))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, y0, x1, y1 = crop_box(trial_image[lo:hi], frame_w, frame_h)
    out = []
    for t in range(lo, hi):
        cap.set(cv2.CAP_PROP_POS_FRAMES, offset + t)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((frame_h, frame_w, 3), np.uint8)
        draw_landmarks(frame, trial_image[t], frame_w, frame_h)
        out.append(fit_panel(frame[y0:y1, x0:x1], PHONE_SIDE))
    cap.release()
    return out


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

def overlay(canvas: np.ndarray, lines: list[tuple[str, tuple]]) -> None:
    y = 24
    for text, color in lines:
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2, cv2.LINE_AA)
        y += 24


def label(canvas: np.ndarray, text: str, x: int) -> None:
    y = canvas.shape[0] - 10
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)


def render_trial(rep: Replayer, trial: str, hand_geoms: list[int],
                 table_body: int, sticky_body: int, palm_body: int) -> dict:
    meta_t = json.loads((TRIALS_DIR / f"{trial}.json").read_text())
    grasp_f, release_f = meta_t["grasp_frame"], meta_t["release_frame"]
    rec = wrist_traj.to_sim(rep, json.loads((QPOS_DIR / f"{trial}.json").read_text()),
                            meta_t, translate=WRIST_MODE == "image")
    n_total = len(rec["qpos"])
    lo = max(0, grasp_f - LEAD)
    hi = min(n_total, release_f + AFTER + 1)

    qpos = np.array(rec["qpos"])
    qadr = rep.scene_qpos_adr(rec)   # by joint name, see replay.py
    wrist_pos = np.array(rec["wrist_pos"])
    wrist_quat = np.array(rec["wrist_quat"])
    hand_free_adr = rep._hand_free_adr()
    fps = rec.get("fps", 30.0)
    tips = sorted(rep.tips)
    cq = rep.cup_qadr

    # cup seeded at the fingertip centroid at the annotated grasp frame
    # (same as replay.py); the table top is staged right under it
    cup_start = rep.tip_centroid_at_frame(rec, grasp_f)
    cup_rest_z = cup_start[2]
    table_top_z = cup_rest_z - CUP_HEIGHT / 2 - 0.005
    rep.model.body_pos[table_body] = [cup_start[0], cup_start[1],
                                      table_top_z - TABLE_HEIGHT / 2]
    rep.model.body_pos[sticky_body] = [0, 0, -10]  # hidden until we know the release spot

    mujoco.mj_resetData(rep.model, rep.data)
    rep.data.qpos[cq:cq + 3] = cup_start
    rep.data.qpos[cq + 3:cq + 7] = [1, 0, 0, 0]

    rep.cam.distance = CAM_DISTANCE
    rep.cam.azimuth = CAM_AZIMUTH
    rep.cam.elevation = CAM_ELEVATION

    # pass 1: run the attach logic once to find where the cup is released,
    # so the staged sticky note can already be on the table from frame 0
    def simulate():
        state = "on table"          # "on table" -> "ATTACHED" -> "released"
        grip_offset = None
        drop_z, drop_t = None, None
        for t in range(lo, hi):
            rep.data.qpos[hand_free_adr:hand_free_adr + 3] = wrist_pos[t]
            rep.data.qpos[hand_free_adr + 3:hand_free_adr + 7] = wrist_quat[t]
            rep.data.qpos[qadr] = qpos[t]
            mujoco.mj_forward(rep.model, rep.data)
            ncon = fingertip_contacts(rep.model, rep.data, rep.tips, rep.cup_body)
            tip_mean = np.mean([rep.data.xpos[i] for i in tips], axis=0)
            if state == "on table" and ncon >= 2 and t < release_f:
                # attach: remember the cup's offset from the fingertips
                state = "ATTACHED"
                grip_offset = rep.data.qpos[cq:cq + 3] - tip_mean
            elif state == "ATTACHED" and t >= release_f:
                # detach at the annotated release frame; from here the cup
                # falls straight down (simple free fall, drawn kinematically)
                # and comes to rest upright on the table
                state = "released"
                drop_z, drop_t = float(rep.data.qpos[cq + 2]), t
                rep.data.qpos[cq + 3:cq + 7] = [1, 0, 0, 0]
            if state == "released":
                dt = (t - drop_t) / fps
                rep.data.qpos[cq + 2] = max(cup_rest_z, drop_z - 0.5 * 9.81 * dt * dt)
            if state == "ATTACHED":
                rep.data.qpos[cq:cq + 3] = tip_mean + grip_offset
                # don't let the carried cup sink through the table top
                held_at_table = rep.data.qpos[cq + 2] < cup_rest_z - 0.01
                rep.data.qpos[cq + 2] = max(rep.data.qpos[cq + 2], cup_rest_z)
            else:
                held_at_table = False
            mujoco.mj_forward(rep.model, rep.data)
            yield t, ncon, state, held_at_table

    for _ in simulate():
        pass
    release_xy = rep.data.qpos[cq:cq + 2].copy()
    # how far the hand's wrist moves from grasp to release (sim frame)
    wb = mujoco.mj_name2id(rep.model, mujoco.mjtObj.mjOBJ_BODY, "rh_wrist")
    wrist_at = {}
    for f_ in (grasp_f, release_f):
        rep.tip_centroid_at_frame(rec, f_)
        wrist_at[f_] = rep.data.xpos[wb].copy()
    hand_cm = float(np.linalg.norm(wrist_at[release_f] - wrist_at[grasp_f]) * 100)
    carried_cm = float(np.linalg.norm(release_xy - cup_start[:2]) * 100)
    rep.model.body_pos[sticky_body] = [release_xy[0], release_xy[1],
                                       table_top_z + STICKY_NOTE_THICKNESS / 2]

    # pass 2: same thing again, this time rendering and checking visibility
    mujoco.mj_resetData(rep.model, rep.data)
    rep.data.qpos[cq:cq + 3] = cup_start
    rep.data.qpos[cq + 3:cq + 7] = [1, 0, 0, 0]
    mj_frames, states, contacts, passes, below_table = [], [], [], [], []
    inside_ok, pixels_ok = [], []
    lookat = None
    for t, ncon, state, held_at_table in simulate():
        palm = rep.data.xpos[palm_body].copy()
        cup = rep.data.qpos[cq:cq + 3].copy()
        target = 0.5 * (palm + cup)      # follow the palm, keep the cup in frame too
        lookat = target if lookat is None else (
            (1 - LOOKAT_SMOOTH) * lookat + LOOKAT_SMOOTH * target)
        rep.cam.lookat[:] = lookat
        rep.renderer.update_scene(rep.data, rep.cam)
        img = rep.renderer.render().copy()

        inside = points_inside(rep, [palm] + [rep.data.xpos[i] for i in tips])
        frac = hand_pixel_frac(rep, img, hand_geoms)
        inside_ok.append(inside)
        pixels_ok.append(frac >= MIN_HAND_PIXEL_FRAC)
        passes.append(inside and frac >= MIN_HAND_PIXEL_FRAC)
        mj_frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        states.append(state)
        contacts.append(ncon)
        below_table.append(held_at_table)  # hand went >1cm below where the cup could follow

    # --- compose side by side + overlay ---
    phones = phone_frames(meta_t, lo, hi)
    meta = TRIALS[trial]
    out_frames = []
    for i, (pf, mf) in enumerate(zip(phones, mj_frames)):
        canvas = np.hstack([pf, mf])
        t = lo + i
        phase = ("approach" if t < grasp_f else
                 "grasp..release" if t < release_f else "after release")
        lines = [(f"{trial}  frame {t}  [{phase}]", (235, 235, 235)),
                 (f"fingertips touching cup: {contacts[i]}", (235, 235, 235))]
        if states[i] == "ATTACHED":
            lines.append(("ATTACHED (kinematic)", (0, 220, 0)))
        elif states[i] == "released":
            lines.append(("released at annotated frame", (255, 200, 80)))
        if meta["failure"]:
            lines.append((f"FAILED: {meta['why']}", (80, 80, 255)))
        overlay(canvas, lines)
        label(canvas, "phone + MediaPipe", 10)
        label(canvas, "MuJoCo, kinematic attach (not physics)", PHONE_SIDE + 10)
        out_frames.append(canvas)

    return {"frames": out_frames, "states": states, "lo": lo, "hi": hi,
            "pass_rate": float(np.mean(passes)), "hand_cm": hand_cm,
            "cup_dxy_cm": ((release_xy - cup_start[:2]) * 100).tolist(),
            "inside_rate": float(np.mean(inside_ok)),
            "pixels_rate": float(np.mean(pixels_ok)), "carried_cm": carried_cm,
            "attached_frames": states.count("ATTACHED"),
            "below_table_frames": int(sum(below_table)),
            "fps": fps}


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
    """3 stills: mid-approach, middle of the carry, last frame (after release)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    states = res["states"]
    att = [i for i, s in enumerate(states) if s == "ATTACHED"]
    picks = {"1_start": 0,
             "2_carry": att[len(att) // 2] if att else len(states) // 2,
             "3_end": len(states) - 1}
    for name, i in picks.items():
        cv2.imwrite(str(out_dir / f"{trial}_{name}.png"), res["frames"][i])


def main() -> None:
    global WRIST_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", type=Path, default=ROOT / "outputs/gif_stills",
                    help="scratch folder for 3 still frames per clip (not committed)")
    ap.add_argument("--trials", nargs="*", help="override the trial list (for trying candidates)")
    ap.add_argument("--gif-dir", type=Path, default=GIF_DIR)
    ap.add_argument("--wrist", choices=["image", "rotate-only"], default=WRIST_MODE)
    args = ap.parse_args()
    WRIST_MODE = args.wrist

    rep = Replayer(str(SCENE), mode="kinematic", width=MJ_W, height=MJ_H)
    hand_geoms = style_scene(rep.model)
    body = lambda n: mujoco.mj_name2id(rep.model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731

    trials = args.trials or list(TRIALS)
    for trial in trials:
        TRIALS.setdefault(trial, {"failure": False})
        res = render_trial(rep, trial, hand_geoms, body("table"),
                           body("sticky_note"), body("rh_palm"))
        gif = args.gif_dir / f"{trial}.gif"
        save_gif(res["frames"], gif, res["fps"])
        save_stills(res, trial, args.stills)
        verdict = "ok" if res["pass_rate"] >= PASS_RATE_REQUIRED else "FAIL (<95%)"
        print(f"{trial}: frames {res['lo']}-{res['hi'] - 1} "
              f"({len(res['frames'])}), attached {res['attached_frames']} frames, "
              f"carried {res['carried_cm']:.1f}cm, "
              f"cup held at table {res['below_table_frames']} frames, "
              f"visible {res['pass_rate'] * 100:.1f}% {verdict} "
              f"(points in image {res['inside_rate'] * 100:.0f}%, "
              f"enough hand pixels {res['pixels_rate'] * 100:.0f}%), "
              f"{gif.name} {gif.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
