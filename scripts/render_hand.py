"""Render a retargeted trial as video: the robot hand, optionally beside your clip.

This is the check that numbers cannot give you. A tip error in millimetres
says nothing about whether the hand is curling the right way; looking at it
says so immediately.

    python scripts/render_hand.py outputs/qpos/ep_004_t01.json \
        --model mujoco_menagerie/shadow_hand/right_hand.xml \
        --out outputs/hand_ep004.mp4

    # side by side with the source clip
    python scripts/render_hand.py outputs/qpos/ep_004_t01.json \
        --model mujoco_menagerie/shadow_hand/right_hand.xml \
        --clip data/episodes/ep_004.mp4 --trial-json data/trials/ep_004_t01.json \
        --out outputs/compare_ep004.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np


def render_frames(model_path: str, qpos, qadr, width: int, height: int,
                  azimuth: float, elevation: float, distance: float):
    model = mujoco.MjModel.from_xml_path(model_path)
    # models default to a 640x480 offscreen buffer and the renderer refuses
    # anything larger, so raise it to whatever was asked for
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    # a bare hand model ships no lights, and Shadow's meshes are near black,
    # so the render comes out unreadable without turning the headlight up
    model.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
    model.vis.headlight.diffuse[:] = [0.75, 0.75, 0.75]
    model.vis.headlight.specular[:] = [0.15, 0.15, 0.15]
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    # centre on the hand rather than the world origin
    cam.lookat[:] = data.xpos[1:].mean(axis=0) if model.nbody > 1 else [0, 0, 0]

    out = []
    for q in qpos:
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam)
        out.append(renderer.render().copy())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("qpos_json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", help="source episode mp4 for a side-by-side")
    ap.add_argument("--trial-json", help="trial file, to locate the clip frames")
    ap.add_argument("--size", type=int, nargs=2, default=[640, 640])
    ap.add_argument("--brighten", type=float, default=1.0,
                    help="extra gain on the rendered image, e.g. 1.6")
    ap.add_argument("--azimuth", type=float, default=140.0)
    ap.add_argument("--elevation", type=float, default=-20.0)
    ap.add_argument("--distance", type=float, default=0.5)
    args = ap.parse_args()

    rec = json.loads(Path(args.qpos_json).read_text())
    qpos = np.array(rec["qpos"])
    qadr = np.array(rec["qpos_adr"])
    fps = rec.get("fps", 30.0)
    w, h = args.size

    print(f"rendering {len(qpos)} frames...")
    frames = render_frames(args.model, qpos, qadr, w, h,
                           args.azimuth, args.elevation, args.distance)
    frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
    if args.brighten != 1.0:
        frames = [np.clip(f.astype(np.float32) * args.brighten, 0, 255)
                  .astype(np.uint8) for f in frames]

    if args.clip and args.trial_json:
        tr = json.loads(Path(args.trial_json).read_text())
        cap = cv2.VideoCapture(args.clip)
        # the trial window inside the episode is not recorded explicitly, so
        # take the first n frames of the clip that line up by length
        src = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            src.append(fr)
        cap.release()
        n = min(len(src), len(frames))
        combined = []
        for i in range(n):
            left = cv2.resize(src[i], (w, h))
            combined.append(np.hstack([left, frames[i]]))
        frames = combined
        print(f"side by side: {n} frames")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hh, ww = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
