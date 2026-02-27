from __future__ import annotations
import numpy as np


def get_plane_cross_time(
    p_prev,
    p_curr,
    t_prev: float,
    t_curr: float,
    o_plane,
    n_plane,
) -> float | None:
    """
    Return simulation time when the ball center crosses the plane
        dot(p - o_plane, n_plane) = 0
    between (p_prev, t_prev) and (p_curr, t_curr).
    """
    p_prev = np.asarray(p_prev, dtype=float)
    p_curr = np.asarray(p_curr, dtype=float)
    o_plane = np.asarray(o_plane, dtype=float)
    n_plane = np.asarray(n_plane, dtype=float)

    n_norm = np.linalg.norm(n_plane)
    if n_norm < 1e-12:
        return None
    n = n_plane / n_norm

    d_prev = float(np.dot(p_prev - o_plane, n))
    d_curr = float(np.dot(p_curr - o_plane, n))

    if not (d_prev >= 0.0 and d_curr <= 0.0):
        return None

    denom = d_curr - d_prev
    if abs(denom) < 1e-12:
        return None

    tau = d_prev / (d_prev - d_curr)
    if not (0.0 <= tau <= 1.0):
        return None

    t_hit = float(t_prev + tau * (t_curr - t_prev))
    return t_hit
