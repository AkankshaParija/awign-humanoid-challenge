"""Hand-cup penetration, and a penetration-minimising cup offset for the
kinematic attach mode.

Kinematic attach does not resolve collisions: the cup is placed relative to
the hand, and the fingers come from my hand retargeted onto a Shadow Hand
with different proportions (~42 mm fingertip error), so a naive attach puts
fingers inside the cup. This module:

  * measures penetration: for a frame, the deepest overlap between any hand
    collision geom and any cup geom, from mujoco.mj_geomDistance (signed
    distance, negative = overlapping). The hidden forearm stub (rh_forearm)
    is left out; it isn't drawn and isn't part of the hand.

  * fits the attach offset: once the cup attaches it is held FIXED in the
    palm frame (it moves and tilts with the palm). Its offset is chosen
    once per trial to minimise penetration over all attached frames, while
    pulling the two closest fingertips to within CONTACT_THRESHOLD_M of the
    cup surface. Only the translation is fitted: the cup is rotationally
    symmetric about its own axis, so a rotation about that axis changes
    nothing. Finger joint angles are never changed.

Note on "contact": every geom here has margin 0, so MuJoCo only reports a
fingertip contact while the fingertip actually penetrates the cup. With
penetration removed that count drops, so this module counts a fingertip as
"in contact" when its surface is within CONTACT_THRESHOLD_M of the cup.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import minimize

CONTACT_THRESHOLD_M = 0.005   # fingertip within 5 mm of the cup surface = touching
FIT_FRAME_STRIDE = 2          # fit on every 2nd attached frame (speed); report on all
NEAR_M = 0.03                 # pairs further apart than this (bounding spheres) are skipped
GRID_HALF_M, GRID_STEP_M = 0.02, 0.01   # coarse search box before Nelder-Mead
MIN_TIPS2_FRAC = 0.9          # >=2 fingertips within the threshold in >=90% of attached frames


class HandCupGeoms:
    """Geom ids and helpers for hand-cup distance queries on one model."""

    def __init__(self, model) -> None:
        self.model = model
        body = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""  # noqa: E731
        self.cup_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")
        self.palm_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")
        collides = lambda g: model.geom_contype[g] or model.geom_conaffinity[g]  # noqa: E731
        self.cup = [g for g in range(model.ngeom) if model.geom_bodyid[g] == self.cup_body]
        self.hand = [g for g in range(model.ngeom)
                     if body(model.geom_bodyid[g]).startswith("rh_")
                     and body(model.geom_bodyid[g]) != "rh_forearm" and collides(g)]
        # collision geoms of each fingertip (distal) body, one list per finger
        tips: dict[int, list[int]] = {}
        for g in self.hand:
            b = int(model.geom_bodyid[g])
            if "distal" in body(b):
                tips.setdefault(b, []).append(g)
        self.tips = list(tips.values())
        self._fromto = np.zeros(6)

    def _dist(self, data, g1: int, g2: int, distmax: float) -> float:
        # cheap reject with bounding spheres before the exact query
        gap = (np.linalg.norm(data.geom_xpos[g1] - data.geom_xpos[g2])
               - self.model.geom_rbound[g1] - self.model.geom_rbound[g2])
        if gap > distmax:
            return distmax
        return mujoco.mj_geomDistance(self.model, data, g1, g2, distmax, self._fromto)

    def penetration(self, data) -> float:
        """Deepest hand-cup overlap this frame, metres (0 = none)."""
        worst = 0.0
        for g in self.hand:
            for c in self.cup:
                d = self._dist(data, g, c, 0.0)
                worst = max(worst, -d)
        return worst

    def tip_distances(self, data) -> np.ndarray:
        """Signed distance from each fingertip to the cup (negative = inside)."""
        out = []
        for geoms in self.tips:
            out.append(min(self._dist(data, g, c, NEAR_M) for g in geoms for c in self.cup))
        return np.array(out)

    def tips_touching(self, data) -> int:
        return int(np.sum(self.tip_distances(data) <= CONTACT_THRESHOLD_M))


def place_cup_in_palm(model, data, geoms: HandCupGeoms, cup_qadr: int,
                      offset_pos: np.ndarray, offset_quat: np.ndarray) -> None:
    """Set the cup's free joint so its pose is palm_pose * offset."""
    palm_p = data.xpos[geoms.palm_body]
    palm_q = data.xquat[geoms.palm_body]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, palm_q)
    data.qpos[cup_qadr:cup_qadr + 3] = palm_p + R.reshape(3, 3) @ offset_pos
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, palm_q, offset_quat)
    data.qpos[cup_qadr + 3:cup_qadr + 7] = q


def cup_offset_now(data, geoms: HandCupGeoms, cup_qadr: int) -> tuple[np.ndarray, np.ndarray]:
    """The cup's current pose expressed in the palm frame."""
    palm_p = data.xpos[geoms.palm_body]
    palm_q = data.xquat[geoms.palm_body]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, palm_q)
    pos = R.reshape(3, 3).T @ (data.qpos[cup_qadr:cup_qadr + 3] - palm_p)
    inv = np.zeros(4)
    mujoco.mju_negQuat(inv, palm_q)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, inv, data.qpos[cup_qadr + 3:cup_qadr + 7])
    return pos, q


def fit_offset(model, data, geoms: HandCupGeoms, cup_qadr: int,
               set_hand_pose, frames: list[int],
               start_pos: np.ndarray, start_quat: np.ndarray) -> dict:
    """Choose the palm-frame cup translation for one trial's attached frames.

    set_hand_pose(t) must put the hand at frame t (qpos only); this function
    calls mj_kinematics itself. Starts from the cup pose at the moment of
    attach and returns the fitted offset plus before/after diagnostics."""
    fit_frames = frames[::FIT_FRAME_STRIDE]
    cup_geom_local = [(model.geom_pos[c].copy(), model.geom_quat[c].copy()) for c in geoms.cup]

    # hand geom poses don't depend on the cup: compute once per frame
    cache = []
    for t in fit_frames:
        set_hand_pose(t)
        mujoco.mj_kinematics(model, data)
        cache.append((data.geom_xpos[geoms.hand].copy(), data.geom_xmat[geoms.hand].copy(),
                      data.xpos[geoms.palm_body].copy(), data.xquat[geoms.palm_body].copy()))

    def pose_cup_geoms(palm_p, palm_q, off_p):
        """Write cup geom world poses directly (no kinematics call needed)."""
        q_cup = np.zeros(4)
        mujoco.mju_mulQuat(q_cup, palm_q, start_quat)
        Rp = np.zeros(9)
        mujoco.mju_quat2Mat(Rp, palm_q)
        p_cup = palm_p + Rp.reshape(3, 3) @ off_p
        Rc = np.zeros(9)
        mujoco.mju_quat2Mat(Rc, q_cup)
        Rc = Rc.reshape(3, 3)
        for c, (lp, lq) in zip(geoms.cup, cup_geom_local):
            qg = np.zeros(4)
            mujoco.mju_mulQuat(qg, q_cup, lq)
            Rg = np.zeros(9)
            mujoco.mju_quat2Mat(Rg, qg)
            data.geom_xpos[c] = p_cup + Rc @ lp
            data.geom_xmat[c] = Rg

    def per_frame(off_p):
        pens, gaps = [], []
        for hx, hm, pp, pq in cache:
            data.geom_xpos[geoms.hand] = hx
            data.geom_xmat[geoms.hand] = hm
            pose_cup_geoms(pp, pq, off_p)
            pens.append(geoms.penetration(data))
            d = np.sort(geoms.tip_distances(data))
            gaps.append(max(0.0, d[1] - CONTACT_THRESHOLD_M))   # 2nd-closest fingertip
        return np.array(pens), np.array(gaps)

    def cost(off_p):
        pens, gaps = per_frame(off_p)
        # the grasp constraint: >=2 fingertips within the threshold in at
        # least MIN_TIPS2_FRAC of frames. Violations cost far more than any
        # penetration, so the optimiser only trades penetration inside it.
        short = max(0.0, MIN_TIPS2_FRAC - float(np.mean(gaps == 0.0)))
        return pens.mean() + pens.max() + 2.0 * gaps.mean() + 1.0 * short

    p0 = np.array(start_pos, dtype=float)
    pens0, gaps0 = per_frame(p0)
    # the cost is bumpy (max over frames), so start Nelder-Mead from the best
    # point of a coarse +-2 cm grid around the attach pose, not just from it
    steps = np.arange(-GRID_HALF_M, GRID_HALF_M + 1e-9, GRID_STEP_M)
    grid = [p0 + np.array([a, b, c]) for a in steps for b in steps for c in steps]
    start = min(grid, key=cost)
    best = minimize(cost, start, method="Nelder-Mead",
                    options={"xatol": 5e-4, "fatol": 1e-5, "maxiter": 300,
                             "initial_simplex": [start] + [start + 0.005 * e for e in np.eye(3)]})
    pens1, gaps1 = per_frame(best.x)
    return {"tips2_frac_fit": float(np.mean(gaps1 == 0.0)),"offset_pos": best.x, "offset_quat": np.array(start_quat),
            "shift_mm": float(np.linalg.norm(best.x - p0) * 1000),
            "fit_pen_before_max_mm": float(pens0.max() * 1000),
            "fit_pen_after_max_mm": float(pens1.max() * 1000)}
