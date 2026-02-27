from __future__ import annotations
import numpy as np
import math


def world_to_pixel(sim, cam, p_world, W, H):
    """Project a world-space point to pixel (u,v) for the vision sensor."""
    M = np.array(sim.getObjectMatrix(cam, -1), float).reshape(3, 4)
    R, t = M[:, :3], M[:, 3]
    Pc = R.T @ (np.array(p_world, float) - t)
    X, Y, Z = Pc
    if Z <= 1e-6:
        return None
    fovy = sim.getObjectFloatParam(cam, sim.visionfloatparam_perspective_angle)
    fy = (H / 2.0) / math.tan(fovy / 2.0)
    fx = fy * (W / float(H))
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    u = cx - fx * (X / Z)
    v = cy + fy * (Y / Z)
    if 0 <= u < W and 0 <= v < H:
        return int(u), int(v)
    return None
