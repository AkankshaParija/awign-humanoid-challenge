"""Copy the scene file next to the hand model so mesh paths resolve, then
print the command to open it.

Usage:
    python scripts/build_scene.py
    python scripts/build_scene.py --hand tetheria_aero_hand_open
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="shadow_hand", help="menagerie folder name")
    ap.add_argument("--scene", default="sim/scene_cup.xml")
    args = ap.parse_args()

    src = ROOT / args.scene
    hand_dir = ROOT / "mujoco_menagerie" / args.hand

    if not src.exists():
        sys.exit(f"missing {src}")
    if not hand_dir.is_dir():
        sys.exit(f"missing {hand_dir}. did you clone mujoco_menagerie?")

    dst = hand_dir / src.name
    shutil.copy2(src, dst)
    print(f"copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}\n")
    print("available model files in that folder:")
    for p in sorted(hand_dir.glob("*.xml")):
        print(f"  {p.name}")
    print("\nopen it with:")
    print(f"  python -m mujoco.viewer --mjcf={dst}")


if __name__ == "__main__":
    main()
