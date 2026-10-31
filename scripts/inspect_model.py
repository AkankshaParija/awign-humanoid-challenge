"""Dump joints, actuators and bodies of a MuJoCo model.

Usage:
    python scripts/inspect_model.py <path_to_xml> [filter_substring]

Example:
    python scripts/inspect_model.py mujoco_menagerie/shadow_hand/right_hand.xml
    python scripts/inspect_model.py mujoco_menagerie/unitree_g1/g1.xml wrist
"""

import sys

import mujoco

JOINT_TYPES = ["free", "ball", "slide", "hinge"]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    needle = sys.argv[2].lower() if len(sys.argv) > 2 else None

    model = mujoco.MjModel.from_xml_path(path)

    print(f"model: {path}")
    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}\n")

    def keep(name: str) -> bool:
        return needle is None or (name is not None and needle in name.lower())

    print("--- JOINTS ---")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not keep(name):
            continue
        jtype = JOINT_TYPES[model.jnt_type[i]]
        lo, hi = model.jnt_range[i]
        print(f"{i:3d}  {str(name):<32} {jtype:<6} range=[{lo:+.3f}, {hi:+.3f}]")

    print("\n--- ACTUATORS ---")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if keep(name):
            print(f"{i:3d}  {name}")

    print("\n--- BODIES ---")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if keep(name):
            print(f"{i:3d}  {name}")

    print("\n--- SITES ---")
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        if keep(name):
            print(f"{i:3d}  {name}")


if __name__ == "__main__":
    main()
