from __future__ import annotations
import numpy as np
from .hit_detection import WALL_THICKNESS


def rearm_if_needed(sim, hole_handle, ball_handle, state, *, margin=0.20):
    """
    Arm the detectors when the ball is far from the wall; disarm via cooldown.
    Returns True if armed (detectors should run this frame).
    """
    M = np.array(sim.getObjectMatrix(hole_handle, -1), float).reshape(3, 4)
    n = M[:, 2]; n /= (np.linalg.norm(n) + 1e-12)
    o = M[:, 3]
    p = np.array(sim.getObjectPosition(ball_handle, -1), float)
    d = float(np.dot(p - o, n))

    FAR = WALL_THICKNESS * 0.5 + margin

    cd = state.get("cooldown", 0)
    if cd > 0:
        state["cooldown"] = cd - 1
        state["armed"] = False
        return False

    if not state.get("armed", False):
        if abs(d) > FAR:
            for k in ("prev_pos_hole", "prev_pos_wall", "face_origin",
                      "face_normal", "last_hole_hit", "last_wall_hit"):
                state.pop(k, None)
            state["armed"] = True
    return state.get("armed", False)
