"""Replay a retargeted trial in the MuJoCo scene and record it.

Drives the Shadow hand from the retargeted joint angles and the recorded
wrist pose, with the cup and sticky note present, and reports whether the grasp
actually took hold using fingertip contact rather than just "the fingers
closed".

Two modes:

  kinematic (default)  the hand is placed exactly where the data says, and
                       the cup is attached to the hand while fingertip
                       contact holds. Always produces a clean video, and is
                       what you want for showing the retargeting itself.

  physics              the cup is left to the simulator and is only moved by
                       contact forces. Honest, and more likely to drop the
                       cup. Worth reporting both.

    python scripts/replay.py outputs/qpos/ep_001_t01.json \
        --scene mujoco_menagerie/shadow_hand/scene_cup.xml \
        --out outputs/videos/ep_001_t01.mp4

    python scripts/replay.py outputs/qpos --scene ... --out outputs/videos --summary outputs/replay_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CUP_BOTTOM_RADIUS, STICKY_NOTE_SIDE  # noqa: E402

TIP_BODY_HINT = "distal"
# "on the sticky note" = cup center within the note's half-side plus the
# cup's own footprint, so a cup resting with its base anywhere over the
# note counts, not just a pixel-perfect center placement.
ON_TARGET_RADIUS = STICKY_NOTE_SIDE / 2 + CUP_BOTTOM_RADIUS


def tip_body_ids(model) -> list[int]:
    out = []
    for i in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if TIP_BODY_HINT in n.lower():
            out.append(i)
    return out


def body_of_geom(model, gid: int) -> int:
    return int(model.geom_bodyid[gid])


def fingertip_contacts(model, data, tip_ids: set[int], cup_body: int) -> int:
    """How many distinct fingertips are touching the cup this step."""
    touching = set()
    for c in range(data.ncon):
        con = data.contact[c]
        b1 = body_of_geom(model, con.geom1)
        b2 = body_of_geom(model, con.geom2)
        if b1 == cup_body and b2 in tip_ids:
            touching.add(b2)
        elif b2 == cup_body and b1 in tip_ids:
            touching.add(b1)
    return len(touching)


class Replayer:
    def __init__(self, scene_path: str, mode: str = "kinematic",
                 width: int = 960, height: int = 720) -> None:
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        # the default offscreen buffer is 640x480 and the renderer refuses
        # anything bigger, so raise it before constructing the Renderer
        self.model.vis.global_.offwidth = max(
            int(self.model.vis.global_.offwidth), width)
        self.model.vis.global_.offheight = max(
            int(self.model.vis.global_.offheight), height)
        self.data = mujoco.MjData(self.model)
        self.mode = mode

        self.cup_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
        self.sticky_note_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "sticky_note")
        if self.cup_body < 0:
            raise SystemExit("scene has no body named 'cup'")

        self.cup_qadr = None
        for j in range(self.model.njnt):
            if (self.model.jnt_bodyid[j] == self.cup_body
                    and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE):
                self.cup_qadr = int(self.model.jnt_qposadr[j])
                break
        if self.cup_qadr is None:
            raise SystemExit("the cup needs a freejoint")

        self.tips = set(tip_body_ids(self.model))
        if not self.tips:
            raise SystemExit("found no fingertip bodies in the scene")

        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance = 0.9
        self.cam.azimuth = 135
        self.cam.elevation = -25
        self.cam.lookat[:] = self.model.body_pos[self.cup_body]

    def _apply_pose(self, qpos_adr, q, wrist_pos, wrist_quat, hand_free_adr):
        if hand_free_adr is not None:
            self.data.qpos[hand_free_adr:hand_free_adr + 3] = wrist_pos
            self.data.qpos[hand_free_adr + 3:hand_free_adr + 7] = wrist_quat
        self.data.qpos[qpos_adr] = q

    def _hand_free_adr(self) -> int | None:
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                b = int(self.model.jnt_bodyid[j])
                if b != self.cup_body:
                    return int(self.model.jnt_qposadr[j])
        return None

    def tip_centroid_at_frame(self, rec: dict, frame: int) -> np.ndarray:
        """Where the robot's fingertips are at a given frame, in the scene's
        world frame. Used to start the cup near the hand instead of at the
        scene's arbitrary placeholder position: wrist_pos/qpos come straight
        from MediaPipe's camera-relative coordinates, which have no fixed
        relationship to wherever the scene file happened to place the cup."""
        qpos = np.array(rec["qpos"])
        qadr = np.array(rec["qpos_adr"])
        wrist_pos = np.array(rec["wrist_pos"])
        wrist_quat = np.array(rec["wrist_quat"])
        hand_free_adr = self._hand_free_adr()

        mujoco.mj_resetData(self.model, self.data)
        self._apply_pose(qadr, qpos[frame], wrist_pos[frame], wrist_quat[frame],
                         hand_free_adr)
        mujoco.mj_forward(self.model, self.data)
        return np.mean([self.data.xpos[i] for i in sorted(self.tips)], axis=0)

    def run(self, rec: dict, out_path: Path | None,
            cup_start: np.ndarray | None = None) -> dict:
        qpos = np.array(rec["qpos"])
        qadr = np.array(rec["qpos_adr"])
        wrist_pos = np.array(rec["wrist_pos"])
        wrist_quat = np.array(rec["wrist_quat"])
        fps = rec.get("fps", 30.0)
        n = len(qpos)

        # a free joint on the hand root, if the scene gives it one
        hand_free_adr = self._hand_free_adr()

        mujoco.mj_resetData(self.model, self.data)
        if cup_start is not None:
            self.data.qpos[self.cup_qadr:self.cup_qadr + 3] = cup_start
            self.data.qpos[self.cup_qadr + 3:self.cup_qadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

        cup_home = self.data.qpos[self.cup_qadr:self.cup_qadr + 3].copy()
        sticky_note_xy = (self.data.xpos[self.sticky_note_body][:2]
                      if self.sticky_note_body >= 0 else None)

        frames, contacts, held = [], [], False
        grip_offset = None

        for t in range(n):
            self._apply_pose(qadr, qpos[t], wrist_pos[t], wrist_quat[t],
                             hand_free_adr)
            mujoco.mj_forward(self.model, self.data)

            ncon = fingertip_contacts(self.model, self.data, self.tips,
                                      self.cup_body)
            contacts.append(ncon)

            if self.mode == "kinematic":
                palm = self.data.xpos[self.cup_body]
                if ncon >= 2 and not held:
                    held = True
                    tip_mean = np.mean(
                        [self.data.xpos[i] for i in sorted(self.tips)], axis=0)
                    grip_offset = self.data.qpos[
                        self.cup_qadr:self.cup_qadr + 3] - tip_mean
                elif ncon < 1 and held:
                    held = False
                if held and grip_offset is not None:
                    tip_mean = np.mean(
                        [self.data.xpos[i] for i in sorted(self.tips)], axis=0)
                    self.data.qpos[self.cup_qadr:self.cup_qadr + 3] = \
                        tip_mean + grip_offset
                    mujoco.mj_forward(self.model, self.data)
            else:
                mujoco.mj_step(self.model, self.data)

            if out_path is not None:
                self.renderer.update_scene(self.data, self.cam)
                frames.append(self.renderer.render().copy())

        cup_end = self.data.qpos[self.cup_qadr:self.cup_qadr + 3].copy()
        moved = float(np.linalg.norm(cup_end - cup_home))
        on_sticky_note = None
        if sticky_note_xy is not None:
            on_sticky_note = bool(
                np.linalg.norm(cup_end[:2] - sticky_note_xy) < ON_TARGET_RADIUS
                and cup_end[2] > cup_home[2] - 0.03)

        if out_path is not None and frames:
            write_video(frames, out_path, fps)

        return {
            "n_frames": n,
            "max_fingertip_contacts": int(max(contacts) if contacts else 0),
            "frames_with_grasp": int(sum(1 for c in contacts if c >= 2)),
            "cup_moved_m": moved,
            "cup_end": cup_end.tolist(),
            "on_sticky_note": on_sticky_note,
            "mode": self.mode,
        }


def write_video(frames, path: Path, fps: float) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("qpos", help="retargeted JSON or a folder of them")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", help="output mp4 or folder; omit to skip rendering")
    ap.add_argument("--mode", choices=["kinematic", "physics"], default="kinematic")
    ap.add_argument("--summary", help="write a CSV of per-trial results")
    ap.add_argument("--trials-dir", default="data/trials",
                    help="used to look up each trial's annotated grasp_frame, "
                         "so the cup can start near the hand instead of the "
                         "scene's placeholder position (wrist_pos is in "
                         "MediaPipe's camera frame, unrelated to the scene's "
                         "table/cup coordinates)")
    args = ap.parse_args()

    src = Path(args.qpos)
    files = sorted(src.glob("*.json")) if src.is_dir() else [src]
    if not files:
        raise SystemExit(f"no JSON in {src}")

    trials_dir = Path(args.trials_dir)

    rep = Replayer(args.scene, mode=args.mode)
    print(f"scene: {args.scene}")
    print(f"fingertip bodies: {len(rep.tips)}   mode: {args.mode}\n")

    rows = []
    for f in files:
        rec = json.loads(f.read_text())
        out_path = None
        if args.out:
            out = Path(args.out)
            out_path = (out / (f.stem + ".mp4")) if src.is_dir() else out

        cup_start = None
        trial_path = trials_dir / f"{f.stem}.json"
        if trial_path.exists():
            grasp_frame = json.loads(trial_path.read_text()).get("grasp_frame")
            if grasp_frame is not None and 0 <= grasp_frame < len(rec["qpos"]):
                cup_start = rep.tip_centroid_at_frame(rec, grasp_frame)

        res = rep.run(rec, out_path, cup_start=cup_start)
        res["trial"] = f.stem
        rows.append(res)
        print(f"  {f.stem}: contacts<={res['max_fingertip_contacts']}  "
              f"grasped {res['frames_with_grasp']}/{res['n_frames']} frames  "
              f"cup moved {res['cup_moved_m'] * 100:.1f}cm  "
              f"on_sticky_note={res['on_sticky_note']}")

    if rows:
        ok = sum(1 for r in rows if r["frames_with_grasp"] > 0)
        print(f"\n{ok}/{len(rows)} trials registered a grasp "
              f"(>=2 fingertips touching the cup)")
        placed = [r for r in rows if r["on_sticky_note"]]
        if placed:
            print(f"{len(placed)}/{len(rows)} ended with the cup on the sticky note")

    if args.summary:
        p = Path(args.summary)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
