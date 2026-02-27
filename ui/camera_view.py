from __future__ import annotations
import time
import threading
from collections import deque
import numpy as np
import cv2
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from kalman import PixelKF_CA, draw_kf_overlay, predict_positions, make_hit_decision
from vision import (
    detect_green_ball_and_hole,
    apply_software_occluder,
    draw_occluder_overlay,
    world_to_pixel,
)
from .overlay import OVERLAY_HITS

# --- globals for KF hit/miss decisions and timing ---
PLANE_HIT_TIME_SIM = None     # sim time when ball actually hits wall/hole plane
HIT_TYPE = None               # "hole" or "wall"

KF_DECISIONS = []             # list of decision dicts from make_hit_decision(...)

# snapshot at occlusion start (for forward prediction)
_OCC_SNAPSHOT = None          # dict or None
IMPACT_DECISION_DONE = False
OCCLUSION_DECISION_DONE = False


def run_thrower_cam_matplotlib(
    sim,
    thrower_cam,
    *,
    autostart=False,
    work=None,
    stop_when_done=False,
    title="Thrower Cam",
    tick=None,
):
    """
    Show a live Matplotlib window for a Vision Sensor (thrower_cam) during
    the simulation. This runs in the MAIN thread.
    """
    global PLANE_HIT_TIME_SIM, HIT_TYPE, KF_DECISIONS
    global _OCC_SNAPSHOT, IMPACT_DECISION_DONE, OCCLUSION_DECISION_DONE

    try:
        if autostart and sim.getSimulationState() == sim.simulation_stopped:
            sim.startSimulation()

        while sim.getSimulationState() == sim.simulation_stopped:
            time.sleep(0.02)

    except Exception as e:
        print(f"Main thread: Error during startup: {e}")
        return

    ret = sim.getVisionSensorImg(thrower_cam)
    if isinstance(ret, (tuple, list)) and len(ret) == 2:
        a, b = ret
        if isinstance(a, (list, tuple)) and len(a) == 2:
            res, _ = a, b
        else:
            res, _ = b, a
    else:
        res = sim.getVisionSensorRes(thrower_cam)
    W, H = int(res[0]), int(res[1])

    plt.ion()
    fig, ax = plt.subplots()
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass
    im = ax.imshow(np.zeros((H, W, 3), dtype=np.uint8))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter('thrower_cam_record.mp4', fourcc, 30, (W, H))
    print("Video recording enabled: thrower_cam_record.mp4")
    ax.axis('off')
    plt.show(block=False)

    # --- Kalman setup (simulation-time based) ---
    KF = PixelKF_CA(qa=3600.0, r_px=5.0, gate_nis=9.21)
    print("KF:", KF.__class__.__module__, KF.__class__.__name__,
          "qa=", KF.qa, "r_px=", KF.r_px, "gate=", KF.gate_nis)
    last_sim_t = sim.getSimulationTime()
    last_hole  = None
    occl_time  = 0.0
    KF.reset()

    prev_kf_t = None
    prev_mu = None
    prev_Sig = None

    ball_px_r = 8.0
    occl_prev_active = False

    throw_t0 = None
    occl_window = None

    worker_t = None
    if callable(work):
        KF.reset()
        last_hole = None
        occl_time = 0.0

        PLANE_HIT_TIME_SIM = None
        HIT_TYPE = None
        _OCC_SNAPSHOT = None
        IMPACT_DECISION_DONE = False
        OCCLUSION_DECISION_DONE = False

        while sim.getSimulationState() == sim.simulation_stopped:
            time.sleep(0.01)

        throw_t0 = sim.getSimulationTime()
        occl_window = (throw_t0 +1, throw_t0 + 1.14)

        worker_t = threading.Thread(target=work, daemon=True)
        worker_t.start()

    try:
        while True:
            occl_active = False

            if worker_t and not worker_t.is_alive():
                print("Main thread: Background task (work) has completed.")
                if stop_when_done:
                    print("Main thread: Stopping simulation (stop_when_done=True).")
                    try:
                        sim.stopSimulation()
                    except Exception:
                        pass
                    break
                else:
                    worker_t = None

            if callable(tick):
                tick()

            frame = None
            try:
                if sim.getSimulationState() == sim.simulation_stopped:
                    print("Main thread: Simulation was stopped externally.")
                    break

                try:
                    sim.handleVisionSensor(thrower_cam)
                except Exception:
                    pass

                ret = sim.getVisionSensorImg(thrower_cam)
                if isinstance(ret, (tuple, list)) and len(ret) == 2:
                    a, b = ret
                    if isinstance(a, (list, tuple)) and len(a) == 2:
                        res, img = a, b
                    else:
                        img, res = a, b
                else:
                    img, res = ret, sim.getVisionSensorRes(thrower_cam)
                W, H = int(res[0]), int(res[1])

                buf = (np.frombuffer(img, dtype=np.uint8)
                       if isinstance(img, (bytes, bytearray))
                       else np.asarray(img, dtype=np.uint8))
                frame = buf.reshape(H, W, 3).copy()

            except Exception:
                break

            if frame is not None:
                occl_info = apply_software_occluder(
                    frame,
                    sim=sim,
                    KF=KF,
                    mode="fixed",
                    t_window=occl_window,
                    size_px=90,
                    color=(255, 0, 255),
                )

                occl_active = (occl_info is not None)

                annotated, det = detect_green_ball_and_hole(
                    frame,
                    ignore_hole=occl_active
                )


                if occl_info is not None:
                    draw_occluder_overlay(annotated, occl_info,
                                          color=(0, 255, 255))

                if det["hole"] is not None:
                    hx, hy = det["hole"]["center"]
                    hr = det["hole"]["radius"]
                    if last_hole is None:
                        last_hole = {
                            "c": np.array([hx, hy], float),
                            "r": float(hr),
                        }
                    else:
                        alpha = 0.2
                        last_hole["c"] = ((1 - alpha) * last_hole["c"] +
                                          alpha * np.array([hx, hy], float))
                        last_hole["r"] = ((1 - alpha) * last_hole["r"] +
                                          alpha * float(hr))

                sim_t = sim.getSimulationTime()
                dt = sim_t - last_sim_t
                last_sim_t = sim_t
                dt = max(1e-3, min(0.2, dt))
                cv2.putText(annotated,
                            f"dt={dt*1000:.1f}ms qa={KF.qa:.0f} r={KF.r_px:.1f}",
                            (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (180, 255, 255), 1, cv2.LINE_AA)

                z = det["ball"]["center"] if det["ball"] is not None else None
                if z is None:
                    occl_time += dt
                else:
                    occl_time = 0.0

                if z is not None and occl_time < 1e-6:
                    KF.gate_nis = 400.0
                else:
                    KF.gate_nis = 9.21

                out = KF.step(z, dt)

                if z is not None and out is not None and out["has_update"]                         and (not out["accepted"]):
                    if (out["nis"] is not None) and (out["nis"] > 40.0):
                        KF.reset()
                        out = KF.step(z, 1e-3)

                if out is not None:
                    mu, Sig = out["mu"], out["Sigma"]
                    cv2.putText(
                        annotated,
                        f"upd={out['has_update']} acc={out['accepted']} "
                        f"NIS={-1 if out['nis'] is None else out['nis']:.1f}",
                        (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 255, 255), 1, cv2.LINE_AA,
                    )

                    draw_kf_overlay(
                        annotated, mu, Sig,
                        color=(100, 150, 100),
                        label=("KF upd" if (out["has_update"]
                                            and out["accepted"]) else "KF pred"),
                    )

                    if (PLANE_HIT_TIME_SIM is not None
                        and not IMPACT_DECISION_DONE
                        and last_hole is not None
                        and prev_kf_t is not None):

                        if prev_kf_t <= PLANE_HIT_TIME_SIM <= sim_t:
                            dt_to_hit = PLANE_HIT_TIME_SIM - prev_kf_t

                            mu_hit, Sig_hit = KF.propagate_mean_cov(
                                prev_mu,
                                prev_Sig,
                                dt_to_hit,
                            )

                            dec = make_hit_decision(
                                mu_hit,
                                Sig_hit,
                                hole_center=last_hole["c"],
                                hole_radius_px=float(last_hole["r"]),
                                ball_radius_px=float(ball_px_r),
                                t_decision=sim_t,
                                t_eval=PLANE_HIT_TIME_SIM,
                                context=f"impact_{HIT_TYPE or 'unknown'}",
                                conf_sigma=2.0,
                            )
                            KF_DECISIONS.append(dec)
                            IMPACT_DECISION_DONE = True

                            print(
                                f"[KF] impact decision: label={dec['label']}, "
                                f"hit_bool={dec['hit_bool']} "
                                f"prob={dec.get('prob_hit', 0.0):.2f} "
                                f"(true={HIT_TYPE}, d={dec['distance']:.1f}, "
                                f"R_eff={dec['eff_radius']:.1f}, "
                                f"sigma_max={dec['sigma_major']:.2f}, "
                                f"t_dec={dec['t_decision']:.3f}, "
                                f"t_eval={dec['t_eval']:.3f})"
                            )

                    if det["ball"] is not None:
                        ball_px_r = float(det["ball"]["radius"])

                    occl_active = (occl_info is not None)

                    if (occl_active
                        and not occl_prev_active
                        and _OCC_SNAPSHOT is None
                        and last_hole is not None):
                        covers_ball = (z is None)
                        covers_hole = (det["hole"] is None)

                        if covers_ball and covers_hole:
                            ctx = "occlusion_both"
                        elif covers_ball:
                            ctx = "occlusion_ball"
                        elif covers_hole:
                            ctx = "occlusion_hole"
                        else:
                            ctx = "occlusion_none"

                        if covers_ball or covers_hole:
                            _OCC_SNAPSHOT = {
                                "t_occ": sim_t,
                                "mu_occ": mu.copy(),
                                "Sig_occ": Sig.copy(),
                                "hole_c": last_hole["c"].copy(),
                                "hole_r": float(last_hole["r"]),
                                "ball_r": float(ball_px_r),
                                "ctx": ctx,
                            }
                            print(f"[KF] occlusion start snapshot "
                                  f"(ctx={ctx}, t={sim_t:.3f})")

                    occl_prev_active = occl_active

                    if (_OCC_SNAPSHOT is not None
                        and PLANE_HIT_TIME_SIM is not None
                        and not OCCLUSION_DECISION_DONE):

                        t_occ = _OCC_SNAPSHOT["t_occ"]

                        if PLANE_HIT_TIME_SIM <= t_occ:
                            OCCLUSION_DECISION_DONE = True
                            print("[KF] occlusion decision skipped "
                                  "(impact before occlusion).")
                        else:
                            dt_fwd = PLANE_HIT_TIME_SIM - t_occ

                            mu_fwd, Sig_fwd = KF.propagate_mean_cov(
                                _OCC_SNAPSHOT["mu_occ"],
                                _OCC_SNAPSHOT["Sig_occ"],
                                dt_fwd,
                            )

                            dec = make_hit_decision(
                                mu_fwd,
                                Sig_fwd,
                                hole_center=_OCC_SNAPSHOT["hole_c"],
                                hole_radius_px=_OCC_SNAPSHOT["hole_r"],
                                ball_radius_px=_OCC_SNAPSHOT["ball_r"],
                                t_decision=t_occ,
                                t_eval=PLANE_HIT_TIME_SIM,
                                context=_OCC_SNAPSHOT["ctx"],
                                conf_sigma=2.0,
                            )
                            KF_DECISIONS.append(dec)
                            OCCLUSION_DECISION_DONE = True

                            print(
                                f"[KF] occlusion decision: label={dec['label']}, "
                                f"hit_bool={dec['hit_bool']} "
                                f"prob={dec.get('prob_hit', 0.0):.2f} "
                                f"(true={HIT_TYPE}, "
                                f"d={dec['distance']:.1f}, "
                                f"R_eff={dec['eff_radius']:.1f}, "
                                f"sigma_max={dec['sigma_major']:.2f}, "
                                f"t_dec={dec['t_decision']:.3f}, "
                                f"t_eval={dec['t_eval']:.3f})"
                            )

                    prev_kf_t = sim_t
                    prev_mu = mu.copy()
                    prev_Sig = Sig.copy()

                    if (det["ball"] is None) and (KF.x is not None):
                        for (gu, gv) in predict_positions(
                            KF.x.copy(), steps=12, dt=dt
                        ):
                            cv2.circle(
                                annotated,
                                (int(gu), int(gv)),
                                1,
                                (200, 200, 0),
                                -1,
                            )

                    if last_hole is not None:
                        cv2.circle(
                            annotated,
                            tuple(last_hole["c"].astype(int)),
                            int(last_hole["r"]),
                            (0, 255, 255),
                            1,
                        )

                # overlay queued 3D hit points onto the camera frame
                try:
                    newq = deque(maxlen=OVERLAY_HITS.maxlen)
                    for x, y, z, life in list(OVERLAY_HITS):
                        if life <= 0:
                            continue
                        px = world_to_pixel(sim, thrower_cam,
                                            (x, y, z), W, H)
                        if px is not None:
                            u, v = px
                            cv2.circle(
                                annotated,
                                (u, v),
                                6,
                                (255, 0, 0),
                                -1,
                            )
                        newq.append([x, y, z, life - 1])
                    OVERLAY_HITS.clear()
                    OVERLAY_HITS.extend(newq)
                except Exception as e:
                    print(f"[OVERLAY] {e}")

                im.set_data(annotated)
                video_out.write(annotated)
                ax.set_title(
                    f"{title} | ball: "
                    f"{det['ball']['center'] if det['ball'] else '-'} "
                    f"| hole: "
                    f"{det['hole']['center'] if det['hole'] else '-'}"
                )

            plt.pause(0.01)

    finally:
        try:
            video_out.release()
            print("🎞 Video saved successfully: thrower_cam_record.mp4")
        except Exception as e:
            print("Video Writer cleanup failed:", e)

        plt.close(fig)
        print("Main thread: Closing window.")
        if worker_t and not worker_t.is_alive() and stop_when_done:
            try:
                if sim.getSimulationState() != sim.simulation_stopped:
                    print("Main thread: Final cleanup stop.")
                    sim.stopSimulation()
            except Exception:
                pass
