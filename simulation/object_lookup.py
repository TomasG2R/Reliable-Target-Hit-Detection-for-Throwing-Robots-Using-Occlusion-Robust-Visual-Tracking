from __future__ import annotations


def _resolve_any(sim, path_candidates: list[str], name_candidates: list[str]):
    # Try absolute alias paths first
    for p in path_candidates:
        try:
            return sim.getObject(p)
        except Exception:
            pass
    # Fallback: legacy handle-by-name
    for n in name_candidates:
        try:
            return sim.getObjectHandle(n)
        except Exception:
            pass
    raise RuntimeError(
        f"Object not found. Tried paths={path_candidates}, names={name_candidates}"
    )


def get_thrower_vars(sim):
    """Return handles + world positions for Thrower_Box, Embedded_Ball, Thrower_Cam."""
    thrower_box = _resolve_any(
        sim,
        ['/Thrower_Box'],
        ['Thrower_Box'],
    )
    embedded_ball = _resolve_any(
        sim,
        ['/Embedded_Ball', '/Thrower_Box/Embedded_Ball'],
        ['Embedded_Ball'],
    )
    thrower_cam = _resolve_any(
        sim,
        ['/Thrower_Cam', '/Thrower_Box/ThrowerCam'],
        ['Thrower_Cam', 'ThrowerCam'],
    )
    wall_wall = _resolve_any(
        sim,
        ['/Wall_4x4'],
        ['Wall4x4', 'Wall_4x4'],
    )
    wall_hole = _resolve_any(
        sim,
        ['/WallHole_Cylinder', '/Wall_4x4/WallHole_Cylinder'],
        ['WallHole_Cylinder', 'WallHoleCylinder'],
    )

    wall_hole_pos = sim.getObjectPosition(wall_hole, -1)
    wall_wall_pos = sim.getObjectPosition(wall_wall, -1)
    thrower_box_pos   = sim.getObjectPosition(thrower_box, -1)
    embedded_ball_pos = sim.getObjectPosition(embedded_ball, -1)
    thrower_cam_pos   = sim.getObjectPosition(thrower_cam, -1)

    return (thrower_box, thrower_box_pos,
            embedded_ball, embedded_ball_pos,
            thrower_cam, thrower_cam_pos,
            wall_hole, wall_hole_pos,
            wall_wall, wall_wall_pos)
