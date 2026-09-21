"""Small utilities shared by sim_eval envs (no torch / no leap deps)."""

from __future__ import annotations

import numpy as np


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert a ``(x, y, z, w)`` quaternion to a 3-D axis-angle vector.

    Port of the implementation starVLA uses in ``eval_libero.py`` (itself
    taken from robosuite's ``quat2axisangle``). Returns a 3-vector whose
    direction is the rotation axis and whose magnitude is the rotation
    angle in radians.

    Handles the trivial case ``w≈1`` (no rotation) by returning zeros.
    """
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    # Clip to avoid arccos(>1) from small numerical noise
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    if den < 1e-8:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arccos(w)
    return (quat[:3] / den) * angle


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resize an ``(H, W, 3) uint8`` image to ``(size, size, 3)``.

    Uses simple nearest-neighbour index sampling so the function has no
    dependency on PIL / opencv (which would complicate the conda libero env
    install). Matches what starVLA does via its ``_resize_image`` helper.
    """
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    row_idx = (np.arange(size) * h / size).astype(int)
    col_idx = (np.arange(size) * w / size).astype(int)
    return img[np.ix_(row_idx, col_idx)]
