from __future__ import annotations
from collections import deque


OVERLAY_HITS = deque(maxlen=256)


def push_hit(p_world, life=45):
    OVERLAY_HITS.append([
        float(p_world[0]),
        float(p_world[1]),
        float(p_world[2]),
        int(life),
    ])


def _fmt3(v):
    return f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})"
