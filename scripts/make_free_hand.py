"""Produce a copy of a hand model whose root body carries a free joint.

Menagerie's right_hand.xml welds the hand to the world, which is right for a
teleoperation rig and wrong for pick-and-place: the wrist has to travel to the
cup and carry it to the sticky note. MuJoCo's <include> cannot add a joint to a
body defined elsewhere, so the model is copied with a <freejoint/> inserted
into its root body.

    python scripts/make_free_hand.py mujoco_menagerie/shadow_hand/right_hand.xml
    -> writes mujoco_menagerie/shadow_hand/right_hand_free.xml
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def root_bodies(tree: ET.ElementTree) -> list[ET.Element]:
    wb = tree.getroot().find("worldbody")
    return [] if wb is None else list(wb.findall("body"))


def add_freejoint(src: Path, dst: Path, joint_name: str = "hand_free",
                  body_name: str | None = None) -> tuple[str, bool]:
    tree = ET.parse(src)
    bodies = root_bodies(tree)
    if not bodies:
        raise SystemExit(f"{src} has no top-level body under <worldbody>")

    target = None
    if body_name:
        for b in bodies:
            if b.get("name") == body_name:
                target = b
                break
        if target is None:
            raise SystemExit(f"no top-level body named {body_name!r} in {src}")
    else:
        target = bodies[0]

    existing = [c for c in target
                if c.tag == "freejoint"
                or (c.tag == "joint" and c.get("type") == "free")]
    if existing:
        tree.write(dst, encoding="utf-8", xml_declaration=False)
        return target.get("name", "?"), False

    fj = ET.Element("freejoint", {"name": joint_name})
    target.insert(0, fj)
    tree.write(dst, encoding="utf-8", xml_declaration=False)
    return target.get("name", "?"), True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", help="defaults to <model>_free.xml beside it")
    ap.add_argument("--body", help="top-level body to attach the joint to")
    ap.add_argument("--joint-name", default="hand_free")
    args = ap.parse_args()

    src = Path(args.model)
    if not src.exists():
        sys.exit(f"missing {src}")
    dst = Path(args.out) if args.out else src.with_name(src.stem + "_free.xml")

    name, added = add_freejoint(src, dst, args.joint_name, args.body)
    verb = "added freejoint to" if added else "already had a free joint:"
    print(f"{verb} body {name!r}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
