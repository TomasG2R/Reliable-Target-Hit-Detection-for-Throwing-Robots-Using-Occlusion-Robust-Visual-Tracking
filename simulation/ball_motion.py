from __future__ import annotations
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import numpy as np
import math


def move_ball_during_sim(
    ball_path_or_name: str,
    speed=6.0,              # m/s launch speed
    elevation_deg=-15.0,    # degrees above horizon
    azimuth_deg=0.0,        # yaw: 0 = +Y, +90 = +X
    gravity=-9.81,          # m/s^2 along +Z up (so negative pulls down)
    duration=5.0,           # seconds max
    detach_from_parent=True # reparent to world at throw start
):
    """
    Kinematic 'throw': updates position each tick using v(t) and gravity.
    Works in its own ZMQ client (thread-safe). Coords: +Y fwd, +Z up, +X right.
    """
    client = None
    try:
        client = RemoteAPIClient()
        sim = client.getObject('sim')

        # Resolve handle by path or legacy name
        try:
            ball = sim.getObject(ball_path_or_name)
        except Exception:
            ball = sim.getObjectHandle(ball_path_or_name)

        # Wait until the sim is running
        while sim.getSimulationState() == sim.simulation_stopped:
            time.sleep(0.02)

        # Optionally detach so the ball is independent of the thrower box
        if detach_from_parent:
            try:
                sim.setObjectParent(ball, -1, True)
            except Exception:
                pass

        # Initial world position and velocity
        p = np.array(sim.getObjectPosition(ball, -1), dtype=float)

        elev = math.radians(elevation_deg)
        yaw  = math.radians(azimuth_deg)

        dir_xy = np.array([math.sin(yaw), math.cos(yaw), 0.0])
        dir_up = np.array([0.0, 0.0, 1.0])
        dir_3d = math.cos(elev) * dir_xy + math.sin(elev) * dir_up
        dir_3d /= np.linalg.norm(dir_3d)

        v = dir_3d * float(speed)
        g = np.array([0.0, 0.0, gravity])

        launch_t = sim.getSimulationTime()
        prev_t   = launch_t
        end_t    = launch_t + float(duration)

        print(f"Work thread: Throw started at sim t={launch_t:.4f}s")  # noqa: E501

        while sim.getSimulationState() != sim.simulation_stopped:
            sim_t = sim.getSimulationTime()
            dt = sim_t - prev_t
            if dt <= 0:
                time.sleep(0.001)
                continue
            prev_t = sim_t

            if sim_t >= end_t:
                break

            v = v + g * dt
            p = p + v * dt

            if p[2] < 0.02:
                p[2] = 0.02

            try:
                sim.setObjectPosition(ball, -1, p.tolist())
            except Exception:
                break

            time.sleep(0.001)

        print("Work thread: Throw finished.")

    except Exception as e:
        print(f"Work thread: CRITICAL ERROR: {e}")
    finally:
        if client:
            pass
