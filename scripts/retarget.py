"""Retarget human hand keypoints onto a MuJoCo hand, frame by frame.

This follows DexMV's "demonstration translation" idea: the human hand has more
degrees of freedom than the robot hand, so retargeting is a projection from a
higher-dimensional pose onto the robot's joint space.

Both hands are first expressed in their own palm frame (see handframe.py), so
camera orientation and model conventions cancel out and only the shape of the
pose is compared. Then, for each frame, we solve a small bounded least-squares
problem for the finger joints that best place the robot's fingertips, warm
started from the previous frame so the trajectory stays smooth.

Wrist pose is not optimised. MediaPipe gives it directly and it is saved
alongside the joint angles for the replay step to drive a free joint.

    # check what got auto-detected before running anything
    python scripts/retarget.py --model mujoco_menagerie/shadow_hand/right_hand.xml --probe

    # one episode
    python scripts/retarget.py data/episodes/ep_001.json outputs/qpos/ep_001.json \
        --model mujoco_menagerie/shadow_hand/right_hand.xml

    # all episodes
    python scripts/retarget.py data/episodes outputs/qpos \
        --model mujoco_menagerie/shadow_hand/right_hand.xml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handframe import (  # noqa: E402
    MCPS, TIPS, WRIST, mat_to_quat, umeyama, wrist_pose_sequence,
)

# Palm landmarks used to find the rotation between the two hands.
#
# The wrist is deliberately NOT used. MediaPipe's wrist landmark sits at the
# base of the palm, while a model's "wrist" body sits at the forearm joint,
# much further back; including it inflated the fitted scale to 1.6 and pushed
# every fingertip target out of reach. The knuckles have honest counterparts
# on both hands, and the thumb knuckle keeps the set off a single line.
PALM_HUMAN = [MCPS["index"], MCPS["middle"], MCPS["ring"], MCPS["little"],
              MCPS["thumb"]]
PALM_ROBOT_ORDER = ["index", "middle", "ring", "little", "thumb"]

# Short codes used by Shadow-style models (rh_ffdistal, rh_thdistal, ...) and
# the words other models use. Codes are matched as a suffix together with a
# tip word, which avoids the trap that rh_ffmiddle is the INDEX finger's
# middle phalanx and has nothing to do with the middle finger.
FINGER_CODES = {
    "thumb": ["th", "thumb"],
    "index": ["ff", "index", "fore", "if"],
    "middle": ["mf", "middle", "mid"],
    "ring": ["rf", "ring"],
    "little": ["lf", "little", "pinky", "pinkie"],
}
TIP_WORDS = ["distal", "tip", "fingertip", "3"]
# never treat these as part of a finger
EXCLUDE_BODY = ["forearm", "wrist", "palm", "world", "base", "metacarpal"]
EXCLUDE_JOINT = ["wrj", "wrist"]
FINGER_ORDER = ["thumb", "index", "middle", "ring", "little"]


def names_of(model, objtype, count):
    return [mujoco.mj_id2name(model, objtype, i) for i in range(count)]


class HandModel:
    """Wraps a MuJoCo hand: finds fingertips, finger joints and the palm frame."""

    def __init__(self, path: str, fingers: list[str] | None = None) -> None:
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.bodies = names_of(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.nbody)
        self.joints = names_of(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.njnt)

        self.tip_bodies = self._guess_tips()
        if fingers:
            self.tip_bodies = {f: self.tip_bodies[f] for f in fingers
                               if f in self.tip_bodies}
        self.fingers = [f for f in FINGER_ORDER if f in self.tip_bodies]
        if not self.fingers:
            raise SystemExit("could not identify any fingertip bodies; "
                             "run with --probe and pass --tip-bodies manually")

        self.tip_ids = {f: self.bodies.index(self.tip_bodies[f]) for f in self.fingers}
        # root first: the chains are defined as "everything below the palm"
        self.root_id = self._root_id()
        self.chains = {f: self._chain(self.tip_ids[f]) for f in self.fingers}
        self.mcp_ids = {f: (self.chains[f][0] if self.chains[f] else self.tip_ids[f])
                        for f in self.fingers}

        self.jids = sorted({j for f in self.fingers for j in self._finger_joints(f)})
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.jids])
        self.dofadr = np.array([self.model.jnt_dofadr[j] for j in self.jids])
        lo = np.array([self.model.jnt_range[j][0] for j in self.jids], dtype=float)
        hi = np.array([self.model.jnt_range[j][1] for j in self.jids], dtype=float)
        unlimited = lo >= hi
        lo[unlimited], hi[unlimited] = -np.pi, np.pi
        self.lo, self.hi = lo, hi

        self._rest()
        self.knuckle_ids = self._find_knuckles()
        self.wrist_id = self._find_wrist()
        self.palm_points = self._palm_points()
        self.palm_len = float(np.linalg.norm(
            self.palm_points[1] - self.palm_points[0]))

    # -- detection ---------------------------------------------------------

    def _guess_tips(self) -> dict[str, str]:
        """Find each fingertip body by matching <code><tipword> as a suffix.

        Matching the code and the tip word together as a suffix is what makes
        this reliable: rh_ffdistal ends with 'ffdistal', while rh_ffmiddle
        (the index finger's middle phalanx) ends with 'ffmiddle' and so never
        gets claimed by the middle finger.
        """
        out = {}
        for finger, codes in FINGER_CODES.items():
            best = None
            for n in self.bodies:
                if not n:
                    continue
                low = n.lower()
                if any(x in low for x in EXCLUDE_BODY):
                    continue
                for ci, code in enumerate(codes):
                    for wi, word in enumerate(TIP_WORDS):
                        if low.endswith(code + word):
                            rank = (ci, wi)
                            if best is None or rank < best[0]:
                                best = (rank, n)
            if best:
                out[finger] = best[1]

        # fall back to a looser search for any finger still missing
        for finger, codes in FINGER_CODES.items():
            if finger in out:
                continue
            for n in self.bodies:
                if not n:
                    continue
                low = n.lower()
                if any(x in low for x in EXCLUDE_BODY):
                    continue
                if any(c in low for c in codes) and any(w in low for w in TIP_WORDS):
                    out[finger] = n
                    break
        return out

    def _chain(self, tip_id: int) -> list[int]:
        """Bodies from just below the palm down to the tip."""
        chain, b = [], tip_id
        while b > 0 and b != self.root_id:
            chain.append(b)
            b = int(self.model.body_parentid[b])
        chain.reverse()
        return chain

    def _root_id(self) -> int:
        """Deepest body shared by every finger chain, i.e. the palm.

        Every fingertip traces back through the palm, so the deepest common
        ancestor is the palm itself. Taking the deepest (highest id) one is
        what keeps the forearm and wrist out of the finger chains.
        """
        sets = [set(self._full_chain(self.tip_ids[f])) for f in self.fingers]
        common = set.intersection(*sets) if sets else {0}
        return max(common) if common else 0

    def _full_chain(self, tip_id: int) -> list[int]:
        chain, b = [], tip_id
        while b > 0:
            chain.append(b)
            b = int(self.model.body_parentid[b])
        chain.append(0)
        return chain

    def _finger_joints(self, finger: str) -> list[int]:
        """Hinge joints on this finger's chain, excluding anything on the palm."""
        ids = []
        for b in self.chains[finger]:
            start = int(self.model.body_jntadr[b])
            num = int(self.model.body_jntnum[b])
            for j in range(start, start + num):
                if j < 0 or self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                    continue
                name = (self.joints[j] or "").lower()
                # the wrist is driven from the recorded wrist pose, not fitted
                if any(x in name for x in EXCLUDE_JOINT):
                    continue
                ids.append(j)
        return ids

    # -- kinematics --------------------------------------------------------

    def _rest(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_kinematics(self.model, self.data)

    def _find_knuckles(self) -> dict[str, int]:
        """The body each finger hangs off the palm by.

        This has to be the knuckle, not simply the first body in the chain:
        on Shadow the little finger starts at rh_lfmetacarpal, which sits far
        closer to the wrist than a knuckle does, and using it tilts the whole
        alignment.
        """
        out = {}
        for f in self.fingers:
            chain = self.chains[f]
            pick = None
            for b in chain:
                n = (self.bodies[b] or "").lower()
                if "knuckle" in n:
                    pick = b
                    break
            if pick is None:
                for b in chain:
                    n = (self.bodies[b] or "").lower()
                    if "proximal" in n:
                        pick = b
                        break
            out[f] = pick if pick is not None else (chain[0] if chain
                                                    else self.tip_ids[f])
        return out

    def _find_wrist(self) -> int:
        for i, n in enumerate(self.bodies):
            if n and "wrist" in n.lower():
                return i
        return self.root_id

    def _palm_points(self) -> np.ndarray:
        """Robot counterparts of the human palm landmarks, at rest."""
        self._rest()
        pts = []
        for f in PALM_ROBOT_ORDER:
            bid = self.knuckle_ids.get(f, self.root_id)
            pts.append(self.data.xpos[bid].copy())
        return np.array(pts)

    def knuckle_positions(self) -> np.ndarray:
        """Knuckle position per optimised finger, in model frame, at rest.

        The palm is rigid and the hand is fixed to the world, so these do not
        move while the finger joints are fitted.
        """
        self._rest()
        return np.array([self.data.xpos[self.knuckle_ids[f]].copy()
                         for f in self.fingers])

    def finger_lengths(self) -> np.ndarray:
        """Knuckle-to-tip distance per finger in the rest (extended) pose."""
        self._rest()
        return np.array([
            float(np.linalg.norm(self.data.xpos[self.tip_ids[f]]
                                 - self.data.xpos[self.knuckle_ids[f]]))
            for f in self.fingers])

    def tips_in_model(self, q: np.ndarray) -> np.ndarray:
        """Fingertip positions in the model frame."""
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return np.array([self.data.xpos[self.tip_ids[f]] for f in self.fingers])

    def tip_jacobian(self, q: np.ndarray) -> np.ndarray:
        """d(tips_in_model)/dq, shape (3 * n_fingers, n_joints).

        MuJoCo gives exact body Jacobians, which beats finite differences on
        both speed and accuracy: a numerical Jacobian over 22 joints costs 23
        forward-kinematics calls per solver iteration and is what made the
        first version stall before converging.
        """
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))          # unused, but some bindings want it
        jac_root = np.zeros((3, nv))
        mujoco.mj_jacBody(self.model, self.data, jac_root, jacr, self.root_id)

        blocks = []
        for f in self.fingers:
            jacp[:] = 0.0
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.tip_ids[f])
            # subtracting the root Jacobian is a no-op when the hand is fixed
            # to the world, and keeps this correct if it ever gets a free joint
            blocks.append((jacp - jac_root)[:, self.dofadr])
        return np.vstack(blocks)

    def describe(self) -> str:
        lines = [
            f"model      : nq={self.model.nq} nbody={self.model.nbody}",
            f"palm body  : {self.bodies[self.root_id]}  "
            f"(wrist->index knuckle {self.palm_len*1000:.1f}mm)",
            f"wrist body : {self.bodies[self.wrist_id]}",
            f"optimising : {len(self.jids)} joints over {len(self.fingers)} fingers",
            "",
        ]
        for f in self.fingers:
            js = [self.joints[j] for j in self._finger_joints(f)]
            kn = self.bodies[self.knuckle_ids.get(f, self.root_id)]
            lines.append(f"  {f:<7} tip={self.tip_bodies[f]:<16} "
                         f"knuckle={kn:<18} joints={len(js)}")
        return "\n".join(lines)


class Retargeter:
    def __init__(self, hand: HandModel, smooth_w: float = 0.01,
                 analytic_jac: bool = True,
                 human_len: np.ndarray | None = None) -> None:
        self.hand = hand
        self.smooth_w = smooth_w
        self.analytic_jac = analytic_jac
        # Finger lengths belong to the person, not to one trial. Measuring
        # them per trial means a trial where the hand never fully opens gets
        # a short estimate and a wrong scale, which is why the ratios wobbled
        # between trials. Calibrating once over the whole dataset fixes that.
        self.human_len = human_len

    def fit_frame(self, target: np.ndarray, q0: np.ndarray,
                  smooth_w: float | None = None, max_nfev: int = 300) -> np.ndarray:
        w = self.smooth_w if smooth_w is None else smooth_w
        n = len(q0)
        smooth_block = w * np.eye(n)

        def residual(q):
            err = (self.hand.tips_in_model(q) - target).ravel()
            return np.concatenate([err, w * (q - q0)])

        def jac(q):
            return np.vstack([self.hand.tip_jacobian(q), smooth_block])

        sol = least_squares(
            residual, q0, bounds=(self.hand.lo, self.hand.hi), method="trf",
            jac=jac if self.analytic_jac else "2-point",
            xtol=1e-6, ftol=1e-6, max_nfev=max_nfev)
        return sol.x

    def init_frame(self, target: np.ndarray, n_restarts: int = 4,
                   seed: int = 0) -> np.ndarray:
        """Solve the first frame properly.

        DexMV notes that initialising the optimisation at zero gives poor
        results; there is no previous frame to warm start from, so we drop the
        smoothness term, allow more iterations and try a few random restarts,
        keeping the best. Every later frame inherits this one.
        """
        rng = np.random.default_rng(seed)
        mid = 0.5 * (self.hand.lo + self.hand.hi)
        starts = [np.clip(np.zeros_like(mid), self.hand.lo, self.hand.hi), mid]
        for _ in range(n_restarts):
            starts.append(rng.uniform(self.hand.lo, self.hand.hi))

        best, best_err = None, np.inf
        for q0 in starts:
            q = self.fit_frame(target, q0, smooth_w=0.0, max_nfev=1000)
            err = float(np.linalg.norm(
                self.hand.tips_in_model(q) - target, axis=1).mean())
            if err < best_err:
                best, best_err = q, err
        return best

    def run(self, world: np.ndarray) -> dict:
        n = len(world)

        # Vector retargeting, as DexPilot and AnyTeleop do it.
        #
        # Matching absolute fingertip positions asks one similarity transform
        # to reconcile palm size and finger length at once, which fails as
        # soon as the proportions differ: a scale that fits the palm leaves
        # the fingertips unreachable. Instead each finger is matched by the
        # knuckle-to-tip VECTOR, rotated into the robot's frame and rescaled
        # by that finger's own length. Direction and curl carry over; absolute
        # reach stays whatever the robot can actually do.
        fingers = self.hand.fingers
        tip_idx = [TIPS[f] for f in fingers]
        knuckle_idx = [MCPS[f] for f in fingers]
        knuckle_pos = self.hand.knuckle_positions()
        robot_len = self.hand.finger_lengths()

        human_vecs = world[:, tip_idx] - world[:, knuckle_idx]
        if self.human_len is not None:
            human_len = self.human_len
        else:
            human_len = np.linalg.norm(human_vecs, axis=2).max(axis=0)
        human_len = np.where(human_len < 1e-6, 1.0, human_len)
        per_finger_scale = robot_len / human_len

        rots = np.empty((n, 3, 3))
        fit_res = np.empty(n)
        for t in range(n):
            _, R, _ = umeyama(world[t][PALM_HUMAN], self.hand.palm_points,
                              with_scale=True)
            rots[t] = R
            s2, R2, t2 = umeyama(world[t][PALM_HUMAN], self.hand.palm_points)
            aligned = (s2 * (R2 @ world[t][PALM_HUMAN].T).T) + t2
            fit_res[t] = float(np.linalg.norm(
                aligned - self.hand.palm_points, axis=1).mean())

        targets = np.empty((n, len(fingers), 3))
        for t in range(n):
            rotated = (rots[t] @ human_vecs[t].T).T
            targets[t] = knuckle_pos + rotated * per_finger_scale[:, None]

        scale = float(np.median(per_finger_scale))
        print("    finger length ratios: " + "  ".join(
            f"{f}={r:.2f}" for f, r in zip(fingers, per_finger_scale)))
        print(f"    palm rotation fit residual {fit_res.mean()*1000:.1f}mm")

        wrist_pos, wrist_rot = wrist_pose_sequence(world)
        wrist_quat = np.array([mat_to_quat(R) for R in wrist_rot])

        qs, errs, per_finger = [], [], []
        q = self.init_frame(targets[0])
        for t in range(n):
            q = self.fit_frame(targets[t], q)
            qs.append(q.copy())
            d = np.linalg.norm(self.hand.tips_in_model(q) - targets[t], axis=1)
            per_finger.append(d)
            errs.append(float(d.mean()))
            if t % 50 == 0:
                print(f"    frame {t}/{n}  tip err {errs[-1]*1000:5.1f}mm", flush=True)

        pf = np.array(per_finger).mean(axis=0)
        print("    per-finger mean error: " + "  ".join(
            f"{f}={e*1000:.1f}mm" for f, e in zip(self.hand.fingers, pf)))

        return {
            "per_finger_error_m": {f: float(e)
                                   for f, e in zip(self.hand.fingers, pf)},
            "scale": float(scale),
            "finger_scales": {f: float(r)
                              for f, r in zip(fingers, per_finger_scale)},
            "robot_finger_len_m": {f: float(r)
                                   for f, r in zip(fingers, robot_len)},
            "human_finger_len_m": {f: float(r)
                                   for f, r in zip(fingers, human_len)},
            "palm_fit_residual_m": float(fit_res.mean()),
            "method": "vector_retargeting",
            "robot_palm_m": float(self.hand.palm_len),
            "fingers": self.hand.fingers,
            "joint_names": [self.hand.joints[j] for j in self.hand.jids],
            "qpos_adr": self.hand.qadr.tolist(),
            "qpos": np.array(qs).tolist(),
            "tip_error_m": errs,
            "wrist_pos": wrist_pos.tolist(),
            "wrist_quat": wrist_quat.tolist(),
        }


def calibrate_finger_lengths(paths: list[Path], fingers: list[str],
                             pct: float = 95.0) -> np.ndarray:
    """Measure the person's finger lengths once, over every trial.

    Takes a high percentile rather than the max so one bad frame of tracking
    cannot set the length, and pools all trials so it does not matter whether
    the hand fully opened in any particular one.
    """
    tip_idx = [TIPS[f] for f in fingers]
    knuckle_idx = [MCPS[f] for f in fingers]
    acc = [[] for _ in fingers]
    for p in paths:
        w = np.array(json.loads(p.read_text())["world"])
        d = np.linalg.norm(w[:, tip_idx] - w[:, knuckle_idx], axis=2)
        for i in range(len(fingers)):
            acc[i].append(d[:, i])
    lens = np.array([np.percentile(np.concatenate(a), pct) for a in acc])
    print("calibrated finger lengths over "
          f"{len(paths)} trials (p{pct:.0f}):")
    print("  " + "  ".join(f"{f}={l*1000:.1f}mm"
                           for f, l in zip(fingers, lens)))
    return lens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--fingers", nargs="*", default=None)
    ap.add_argument("--smooth", type=float, default=0.01,
                    help="weight pulling each frame toward the previous one")
    ap.add_argument("--numeric-jac", action="store_true",
                    help="use finite differences instead of MuJoCo's analytic "
                         "Jacobian (slower; only for comparison)")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure finger lengths once across every trial in "
                         "the input folder instead of per trial")
    args = ap.parse_args()

    hand = HandModel(args.model, args.fingers)
    print(hand.describe())
    if args.probe:
        return
    if not args.episode or not args.out:
        raise SystemExit("need an episode and an output path (or use --probe)")

    src, out = Path(args.episode), Path(args.out)
    episodes = sorted(
        p for p in src.glob("ep_*.json")
        if not p.stem.endswith("_phases")) if src.is_dir() else [src]
    (out if src.is_dir() else out.parent).mkdir(parents=True, exist_ok=True)

    human_len = None
    if args.calibrate:
        pool = episodes
        if not src.is_dir():
            pool = sorted(p for p in src.parent.glob("ep_*.json")
                          if not p.stem.endswith("_phases"))
        human_len = calibrate_finger_lengths(pool, hand.fingers)

    rt = Retargeter(hand, smooth_w=args.smooth,
                    analytic_jac=not args.numeric_jac,
                    human_len=human_len)

    for ep_path in episodes:
        print(f"\n{ep_path.name}")
        ep = json.loads(ep_path.read_text())
        res = rt.run(np.array(ep["world"]))
        res["source_episode"] = ep_path.name
        res["fps"] = ep.get("fps", 30.0)
        dst = (out / ep_path.name) if src.is_dir() else out
        dst.write_text(json.dumps(res))
        print(f"  -> {dst}   mean tip error "
              f"{np.mean(res['tip_error_m'])*1000:.1f}mm")


if __name__ == "__main__":
    main()
