from __future__ import annotations
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from simulation import (
    get_thrower_vars,
    move_ball_during_sim,
    mark_when_ball_crosses_hole_plane,
    mark_wall_touch_with_dot,
    rearm_if_needed,
)
from ui import camera_view as camview
from ui.overlay import OVERLAY_HITS


def main():
    """
    Connect to an already-open CoppeliaSim instance (no loadScene here!).
    Assumes the scene is already loaded and simulation will be started
    by this script (autostart in the camera view).
    """
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    print("Connected to CoppeliaSim (ZMQ Remote API). Using existing scene.")

    (
        thrower_box, thrower_box_pos,
        embedded_ball, embedded_ball_pos,
        thrower_cam, thrower_cam_pos,
        wall_hole, wall_hole_pos,
        wall_wall, wall_wall_pos,
    ) = get_thrower_vars(sim)

    HOLE_RADIUS = 0.30 / 2.0
    BALL_RADIUS = 0.20 / 2.0

    _TOUCH_STATE: dict = {}
    _TOUCH_STATE.clear()
    OVERLAY_HITS.clear()

    def tick_touch():
        """
        Called each frame from camera_view.run_thrower_cam_matplotlib.
        Handles wall/hole hit marking and sets plane hit time/type.
        """
        if camview.PLANE_HIT_TIME_SIM is not None:
            return

        sim_t = sim.getSimulationTime()

        if not rearm_if_needed(sim, wall_hole, embedded_ball, _TOUCH_STATE):
            return

        hit_hole, _ = mark_when_ball_crosses_hole_plane(
            sim,
            ball_handle=embedded_ball,
            hole_frame_handle=wall_hole,
            hole_radius=HOLE_RADIUS,
            ball_radius=BALL_RADIUS,
            state=_TOUCH_STATE,
            parent_for_marks=wall_wall,
        )

        wall_hit = False
        if not hit_hole:
            wall_hit = mark_wall_touch_with_dot(
                sim,
                ball_handle=embedded_ball,
                wall_handle=wall_wall,
                hole_handle=wall_hole,
                wall_half_size=(2.0, 2.0),
                ball_radius=BALL_RADIUS,
                hole_radius=HOLE_RADIUS,
                state=_TOUCH_STATE,
            )

        if hit_hole or wall_hit:
            camview.PLANE_HIT_TIME_SIM = sim_t
            camview.HIT_TYPE = "hole" if hit_hole else "wall"

            print(f"[TIME] Plane hit at t={sim_t:.4f}s type={camview.HIT_TYPE}")

            _TOUCH_STATE["plane_hit_time"] = sim_t
            _TOUCH_STATE["armed"] = False
            _TOUCH_STATE["cooldown"] = 12

    try:
        try:
            ball_path = sim.getObjectAlias(embedded_ball, 1)
        except Exception:
            ball_path = '/Embedded_Ball'
            print(f"Warning: Could not get full alias, falling back to '{ball_path}'")  # noqa: E501

        if sim.getSimulationState() != sim.simulation_stopped:
            sim.stopSimulation()
            while sim.getSimulationState() != sim.simulation_stopped:
                time.sleep(0.05)

        camview.run_thrower_cam_matplotlib(
            sim,
            thrower_cam,
            autostart=True,
            work=lambda: move_ball_during_sim(
                ball_path_or_name=ball_path,
                speed=10, # power
                elevation_deg=42.0, # up and down
                azimuth_deg=0.0, # left right 
                gravity=-9.81,
                duration=20.0,
            ),
            stop_when_done=False,
            title="Thrower Cam",
            tick=tick_touch,
        )

        print("Simulation has finished.")

    finally:
        pass


if __name__ == "__main__":
    main()
