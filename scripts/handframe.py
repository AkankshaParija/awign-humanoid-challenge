"""Palm-frame canonicalisation, shared by the retargeter.

MediaPipe world landmarks live in a camera-aligned frame; a MuJoCo hand's
bodies live in the model's own frame. Comparing fingertip positions directly
is meaningless until both are expressed in a frame attached to the palm.

The palm is rigid, so we build a frame from wrist + index MCP + little MCP:

    origin = wrist
    x      = wrist -> index MCP          (along the palm)
    z      = x cross (wrist -> little MCP)   (palm normal)
    y      = z cross x

Doing the same construction on both hands makes the two sets of fingertip
positions directly comparable, up to the overall size difference, which is
handled separately by a scale factor.
"""

from __future__ import annotations

import numpy as np

WRIST = 0
TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "little": 20}
MCPS = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "little": 17}


def palm_basis(wrist: np.ndarray, index_mcp: np.ndarray,
               little_mcp: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation whose columns are the palm frame axes."""
    x = index_mcp - wrist
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        return np.eye(3)
    x = x / nx

    v = little_mcp - wrist
    z = np.cross(x, v)
    nz = np.linalg.norm(z)
    if nz < 1e-9:
        # degenerate: pick any axis orthogonal to x
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(tmp, x)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        z = np.cross(x, tmp)
        nz = np.linalg.norm(z)
    z = z / nz

    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def to_palm_frame(landmarks: np.ndarray) -> tuple[np.ndarray, float]:
    """Express one frame of MediaPipe landmarks in its own palm frame.

    landmarks: (21, 3)
    returns (local landmarks (21,3), palm length used for scaling)
    """
    wrist = landmarks[WRIST]
    R = palm_basis(wrist, landmarks[MCPS["index"]], landmarks[MCPS["little"]])
    local = (landmarks - wrist) @ R
    palm_len = float(np.linalg.norm(landmarks[MCPS["index"]] - wrist))
    return local, palm_len


def sequence_to_palm_frame(world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply to_palm_frame over (T, 21, 3). Returns (T,21,3) and (T,) palm lengths."""
    out = np.empty_like(world)
    lengths = np.empty(len(world))
    for t in range(len(world)):
        out[t], lengths[t] = to_palm_frame(world[t])
    return out, lengths


def wrist_pose_sequence(world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wrist position and palm-frame rotation per frame.

    Returns (T,3) positions and (T,3,3) rotations, for driving a free joint.
    """
    pos = world[:, WRIST].copy()
    rot = np.empty((len(world), 3, 3))
    for t in range(len(world)):
        rot[t] = palm_basis(world[t, WRIST], world[t, MCPS["index"]],
                            world[t, MCPS["little"]])
    return pos, rot


def umeyama(src: np.ndarray, dst: np.ndarray,
            with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Best-fit similarity transform mapping src onto dst.

    Returns (s, R, t) minimising ||s * R @ src_i + t - dst_i||^2, which is the
    closed-form least-squares answer over all the correspondences rather than
    a frame hand-built from two of them.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = len(src)

    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    cs, cd = src - mu_s, dst - mu_d

    cov = (cd.T @ cs) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0          # keep it a rotation, never a reflection
    R = U @ S @ Vt

    if with_scale:
        var_s = (cs ** 2).sum() / n
        s = float((D * np.diag(S)).sum() / var_s) if var_s > 1e-12 else 1.0
    else:
        s = 1.0

    t = mu_d - s * R @ mu_s
    return s, R, t


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to MuJoCo quaternion (w, x, y, z)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)
