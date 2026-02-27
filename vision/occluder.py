from __future__ import annotations
import numpy as np


def apply_software_occluder(
    frame,
    *,
    sim=None,
    KF=None,
    mode="fixed",       # 'fixed' | 'kf' | 'bar_h' | 'bar_v'
    t_window=None,      # (t0, t1) in sim seconds; None => always on
    size_px=50,
    color=(0, 0, 0),
):
    """
    Paints an occluder into 'frame' (in-place) BEFORE detection and returns info.
    Returns: {'rect':(u1,v1,u2,v2), 'mode':..., 'center':(u0,v0)} or None.
    """
    if not frame.flags.writeable:
        frame = frame.copy()
    color = np.array(color, dtype=np.uint8)
    H, W = frame.shape[:2]

    if t_window is not None and sim is not None:
        t0, t1 = t_window
        t = sim.getSimulationTime()
        if not (t0 <= t <= t1):
            return None

    def _clamp_rect(u1, v1, u2, v2):
        u1 = max(0, min(W, int(u1))); u2 = max(0, min(W, int(u2)))
        v1 = max(0, min(H, int(v1))); v2 = max(0, min(H, int(v2)))
        if u2 <= u1 or v2 <= v1:
            return None
        return (u1, v1, u2, v2)

    rect = None
    center = None

    if mode == "fixed":
        u0 = int(0.5 * W)
        v0 = int(0.5 * H)
        center = (u0, v0)
        rect = _clamp_rect(u0 - size_px, v0 - size_px,
                           u0 + size_px, v0 + size_px)

    elif mode == "kf":
        if KF is None or getattr(KF, "x", None) is None:
            return None
        u0 = int(KF.x[0]); v0 = int(KF.x[1])
        center = (u0, v0)
        rect = _clamp_rect(u0 - size_px, v0 - size_px,
                           u0 + size_px, v0 + size_px)

    elif mode == "bar_h":
        if t_window is None or sim is None:
            v0 = H // 2
        else:
            t0, t1 = t_window
            alpha = (sim.getSimulationTime() - t0) / max(1e-6, (t1 - t0))
            alpha = max(0.0, min(1.0, alpha))
            v0 = int(alpha * (H - 1))
        center = (W // 2, v0)
        rect = _clamp_rect(0, v0 - size_px, W, v0 + size_px)

    elif mode == "bar_v":
        if t_window is None or sim is None:
            u0 = W // 2
        else:
            t0, t1 = t_window
            alpha = (sim.getSimulationTime() - t0) / max(1e-6, (t1 - t0))
            alpha = max(0.0, min(1.0, alpha))
            u0 = int(alpha * (W - 2))
        center = (u0, H // 2)
        rect = _clamp_rect(u0 - size_px, 0, u0 + size_px, H)

    else:
        return None

    if rect is None:
        return None

    u1, v1, u2, v2 = rect
    frame[v1:v2, u1:u2, :] = color
    return {"rect": rect, "mode": mode, "center": center}


def draw_occluder_overlay(img, info, color=(0, 255, 255)):
    """Draw a thin box showing where the occluder was applied (for debugging)."""
    if not info or "rect" not in info:
        return
    u1, v1, u2, v2 = info["rect"]
    import cv2
    cv2.rectangle(img, (u1, v1), (u2, v2), color, 1, cv2.LINE_AA)
