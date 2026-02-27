from __future__ import annotations
import numpy as np
from ui.overlay import push_hit, _fmt3


WALL_THICKNESS = 0.10  # meters (4x4 wall is 0.1m thick)


def _front_face_origin(oh, n, p_ref, thickness=WALL_THICKNESS):
    """
    Choose the wall face on the same side as p_ref (the ball's previous center).
    oh: hole center (world)
    n:  wall normal (+Z of the hole frame), unit vector
    p_ref: reference world point (we use ball's previous center)
    """
    d = float(np.dot(p_ref - oh, n))
    s = 1.0 if d >= 0.0 else -1.0
    return oh + s * n * (0.5 * thickness), s


def mark_when_ball_crosses_hole_plane(
    sim,
    ball_handle: int,
    hole_frame_handle: int,
    hole_radius: float,
    *,
    ball_radius: float = 0.10,
    dot_size: float = 0.025,
    state: dict | None = None,
    parent_for_marks: int = -1,
    push_to_overlay: bool = True,
    overlay_life: int = 60,
    min_sep: float = 0.02,
):
    if state is None:
        state = {}

    p_curr = np.array(sim.getObjectPosition(ball_handle, -1), float)
    p_prev = state.get("prev_pos_hole")
    state["prev_pos_hole"] = p_curr.copy()
    if p_prev is None:
        return False, None

    M = np.array(sim.getObjectMatrix(hole_frame_handle, -1), float).reshape(3, 4)
    R, oh = M[:, :3], M[:, 3]
    n = R[:, 2]
    n /= (np.linalg.norm(n) + 1e-12)

    if "face_origin" not in state or "face_normal" not in state:
        o_face, s = _front_face_origin(oh, n, p_prev, WALL_THICKNESS)
        state["face_origin"] = o_face
        state["face_normal"] = s * n

    o_face = state["face_origin"]
    n_face = state["face_normal"]

    v = p_curr - p_prev
    d_prev = float(np.dot(p_prev - o_face, n_face))
    d_curr = float(np.dot(p_curr - o_face, n_face))
    denom  = d_curr - d_prev
    if abs(denom) < 1e-12:
        return False, None

    if not (d_prev >= 0.0 and d_curr <= 0.0):
        return False, None

    t = d_prev / (d_prev - d_curr)
    p_hit_center = p_prev + t * v

    pL = R.T @ (p_hit_center - oh)
    r_eff = max(hole_radius - ball_radius, 0.0)
    inside = (pL[0]**2 + pL[1]**2) <= r_eff**2

    if not inside:
        return False, None

    last = state.get("last_hole_hit")
    if last is not None and np.linalg.norm(p_hit_center - np.array(last, float)) < min_sep:
        return True, {"hit_point_world": p_hit_center.tolist(), "inside_hole": True}

    draw = state.get("draw_hole")
    if draw is None:
        try:
            draw = sim.addDrawingObject(sim.drawing_points, dot_size, 0.0,
                                        parent_for_marks, 500)
            try:
                sim.setDrawingObjectColor(draw, None, [1, 0, 0])
            except Exception:
                pass
            state["draw_hole"] = draw
        except Exception:
            draw = None
    if draw is not None:
        try:
            sim.addDrawingObjectItem(draw, list(map(float, p_hit_center)))
        except Exception:
            pass
    if push_to_overlay:
        push_hit(p_hit_center, life=overlay_life)

    state["last_hole_hit"] = p_hit_center.tolist()
    print(f"[HIT-ONCE] HOLE (inside) at {_fmt3(p_hit_center)}")
    info = {"hit_point_world": p_hit_center.tolist(), "inside_hole": True}
    return True, info


def mark_wall_touch_with_dot(
    sim,
    *,
    ball_handle: int,
    wall_handle: int,
    hole_handle: int,
    wall_half_size=(2.0, 2.0),
    ball_radius: float = 0.10,
    hole_radius: float = 0.15,
    dot_size: float = 0.025,
    state: dict | None = None,
    push_to_overlay: bool = True,
    overlay_life: int = 60,
    min_sep: float = 0.02,
) -> bool:
    if state is None:
        state = {}

    p_curr = np.array(sim.getObjectPosition(ball_handle, -1), float)
    p_prev = state.get("prev_pos_wall")
    state["prev_pos_wall"] = p_curr.copy()
    if p_prev is None:
        return False

    Mh = np.array(sim.getObjectMatrix(hole_handle, -1), float).reshape(3, 4)
    Rh, oh = Mh[:, :3], Mh[:, 3]
    n = Rh[:, 2]
    n /= (np.linalg.norm(n) + 1e-12)

    if "face_origin" not in state or "face_normal" not in state:
        o_face, s = _front_face_origin(oh, n, p_prev, WALL_THICKNESS)
        state["face_origin"] = o_face
        state["face_normal"] = s * n

    o_face = state["face_origin"]
    n_face = state["face_normal"]

    v = p_curr - p_prev
    d_prev = float(np.dot(p_prev - o_face, n_face))
    d_curr = float(np.dot(p_curr - o_face, n_face))
    denom  = d_curr - d_prev
    if abs(denom) < 1e-12:
        return False

    if not (d_prev > ball_radius and d_curr <= ball_radius):
        return False

    t = (ball_radius - d_prev) / (d_curr - d_prev)
    p_c = p_prev + t * v
    p_hit = p_c - n_face * ball_radius

    Mw = np.array(sim.getObjectMatrix(wall_handle, -1), float).reshape(3, 4)
    ow = Mw[:, 3]
    pL = Rh.T @ (p_hit - oh)
    wL = Rh.T @ (ow  - oh)
    x_rel, y_rel = pL[0] - wL[0], pL[1] - wL[1]
    hx, hy = wall_half_size
    inside_rect = (abs(x_rel) <= hx) and (abs(y_rel) <= hy)

    r_eff = max(hole_radius - ball_radius, 0.0)
    r_xy  = float(np.hypot(pL[0], pL[1]))
    inside_hole = (r_xy <= r_eff + 1e-9)
    if not inside_rect or inside_hole:
        return False

    last = state.get("last_wall_hit")
    if last is not None and np.linalg.norm(p_hit - np.array(last, float)) < min_sep:
        return True

    draw = state.get("draw_wall")
    if draw is None:
        try:
            draw = sim.addDrawingObject(sim.drawing_points, dot_size, 0.0,
                                        wall_handle, 1000)
            try:
                sim.setDrawingObjectColor(draw, None, [1, 0, 0])
            except Exception:
                pass
            state["draw_wall"] = draw
        except Exception:
            draw = None
    if draw is not None:
        try:
            sim.addDrawingObjectItem(draw, list(map(float, p_hit)))
        except Exception:
            pass

    if push_to_overlay:
        push_hit(p_hit, life=overlay_life)

    state["last_wall_hit"] = p_hit.tolist()
    print(f"[HIT-ONCE] WALL at {_fmt3(p_hit)} (x_rel={x_rel:.3f}, y_rel={y_rel:.3f})")
    return True
