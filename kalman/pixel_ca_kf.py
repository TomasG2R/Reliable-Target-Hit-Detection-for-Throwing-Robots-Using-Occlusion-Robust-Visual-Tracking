from __future__ import annotations
import numpy as np
import cv2
import math 


class PixelKF_CA:
    """
    Constant-acceleration (CA) model in image pixels.
    - qa: process-noise intensity (higher = more responsive to curvature/jerk)
    - r_px: measurement noise stdev (px) of your detector
    - gate_nis: chi^2 gate on 2D innovation (5.99≈95%, 9.21≈99%, 16.3≈99.9%)
    """
    def __init__(self, qa: float = 400.0, r_px: float = 3.5, gate_nis: float = 9.21):
        self.qa = float(qa)
        self.r_px = float(r_px)
        self.gate_nis = float(gate_nis)

        self.x: np.ndarray | None = None  # 6x1
        self.P: np.ndarray | None = None  # 6x6
        self.R = (self.r_px ** 2) * np.eye(2)

    # ---------- model matrices ----------
    @staticmethod
    def _F(dt: float) -> np.ndarray:
        dt2 = 0.5 * dt * dt
        F = np.eye(6, dtype=float)
        # position update with velocity & acceleration
        F[0, 2] = dt;  F[0, 4] = dt2   # u
        F[1, 3] = dt;  F[1, 5] = dt2   # v
        # velocity affected by acceleration
        F[2, 4] = dt
        F[3, 5] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        # White-jerk noise driving acceleration (standard CA process noise)
        dt2, dt3, dt4, dt5 = dt*dt, dt**3, dt**4, dt**5
        q = self.qa
        Q1 = np.array([
            [dt5/20, dt4/8,  dt3/6],
            [dt4/8,  dt3/3,  dt2/2],
            [dt3/6,  dt2/2,  dt    ]
        ], dtype=float) * q
        Q = np.zeros((6,6), dtype=float)
        Q[:3,:3] = Q1
        Q[3:,3:] = Q1
        return Q

    @staticmethod
    def _H() -> np.ndarray:
        H = np.zeros((2,6), dtype=float)
        H[0,0] = 1.0; H[1,1] = 1.0
        return H

    # ---------- API ----------
    def reset(self):
        """Drop state (useful between throws or after very long occlusions)."""
        self.x = None
        self.P = None

    def step(self, z: tuple[int,int] | list[float] | np.ndarray | None, dt: float):
        """
        One predict/update step.
        - z: None if occluded this frame, else (u,v)
        - dt: timestep in seconds (use simulation time, not wall-clock)
        Returns dict with keys: mu, Sigma, has_update, accepted, nis
        or None if still uninitialized and z is None.
        """
        if dt <= 0:
            dt = 1e-3

        F, Q, H = self._F(dt), self._Q(dt), self._H()

        # Initialize on first detection
        if self.x is None:
            if z is None:
                return None
            u, v = float(z[0]), float(z[1])
            self.x = np.array([u, v, 0, 0, 0, 0], dtype=float)
            self.P = np.diag([25, 25, 400, 400, 1600, 1600]).astype(float)
            return {"mu": self.x.copy(), "Sigma": self.P.copy(),
                    "has_update": True, "accepted": True, "nis": 0.0}

        # Predict
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        has_update, accepted, nis = False, False, None

        # Update if measurement available (with chi-square gating)
        if z is not None:
            has_update = True
            z = np.asarray(z, dtype=float).reshape(2,1)
            y = z - (H @ x_pred).reshape(2,1)           # innovation
            S = H @ P_pred @ H.T + self.R               # innovation cov
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(S)
            nis = float((y.T @ Sinv @ y).ravel()[0])    # normalized innovation squared

            if nis < self.gate_nis:                     # accept update
                K = P_pred @ H.T @ Sinv
                x_new = x_pred + (K @ y).ravel()
                P_new = (np.eye(6) - K @ H) @ P_pred
                self.x, self.P = x_new, P_new
                accepted = True
            else:                                       # reject: use prediction
                self.x, self.P = x_pred, P_pred
        else:
            self.x, self.P = x_pred, P_pred

        return {"mu": self.x.copy(), "Sigma": self.P.copy(),
                "has_update": has_update, "accepted": accepted, "nis": nis}


    def propagate_mean_cov(self, mu: np.ndarray, Sigma: np.ndarray, dt: float):
        """
        Propagate an arbitrary Gaussian state (mu,Sigma) forward by dt seconds
        using the SAME CA model and process noise as this filter.

        This does NOT touch self.x, self.P; it is a pure helper.
        """
        if dt <= 0.0:
            return mu.copy(), Sigma.copy()

        F = self._F(dt)
        Q = self._Q(dt)
        mu2 = F @ mu
        Sigma2 = F @ Sigma @ F.T + Q
        return mu2, Sigma2



def draw_kf_overlay(img_rgb, mu: np.ndarray, Sig: np.ndarray,
                    color=(255,255,0), label="KF"):
    """Draws the mean point and 1-sigma ellipse of the positional covariance."""
    u, v = int(mu[0]), int(mu[1])
    C = Sig[:2, :2]
    # eigen-decomp for ellipse axes/orientation
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 1e-6, None)
    axes = (int(2*np.sqrt(vals[0])), int(2*np.sqrt(vals[1])))
    angle = float(np.degrees(np.arctan2(vecs[1,1], vecs[0,1])))
    cv2.circle(img_rgb, (u, v), 2, color, -1)
    cv2.ellipse(img_rgb, (u, v), axes, angle, 0, 360, color, 1)
    cv2.putText(img_rgb, label, (u, max(12, v-12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def predict_positions(mu: np.ndarray, steps: int, dt: float) -> np.ndarray:
    """
    Utility: fast mean-only rollout (for a small 'ghost' trajectory).
    Returns array shape (steps, 2) of (u,v).
    """
    traj = []
    x = mu.copy()
    for _ in range(steps):
        x[0] += dt * x[2] + 0.5 * dt * dt * x[4]
        x[1] += dt * x[3] + 0.5 * dt * dt * x[5]
        x[2] += dt * x[4]
        x[3] += dt * x[5]
        traj.append((x[0], x[1]))
    return np.array(traj, dtype=float)



def classify_hit_gaussian(
    mu: np.ndarray,
    Sigma: np.ndarray,
    hole_center,
    hole_radius_px: float,
    ball_radius_px: float,
    conf_sigma: float = 2.0,
):
    """Probabilistic hit classifier in 2D image space."""
    if mu is None or Sigma is None or hole_center is None:
        return {
            "hit_bool": None,
            "label": "miss",
            "distance": float("nan"),
            "eff_radius": 0.0,
            "sigma_major": 0.0,
            "margin_px": 0.0,
            "prob_hit": 0.0,
        }

    pos = np.asarray(mu[:2], float)
    hole_c = np.asarray(hole_center, float)
    d = float(np.linalg.norm(pos - hole_c))

    # --- 1) Positional uncertainty from KF (largest 1σ direction) ---
    C = np.asarray(Sigma[:2, :2], float)
    vals, _ = np.linalg.eigh(C)
    vals = np.clip(vals, 1e-12, None)
    sigma_major = float(np.sqrt(vals.max()))

    # --- 2) Geometric scale from hole radius (how big is "close"?) ---
    geom_frac = 0.7
    sigma_geom = max(4.0, geom_frac * float(hole_radius_px))

    # --- 3) Combine geometry + KF uncertainty into sigma_effective ---
    sigma_effective = math.sqrt(sigma_geom**2 + sigma_major**2)

    # --- 4) Probability that this distance is "small enough" ---
    if sigma_effective <= 1e-6:
        prob_hit = 1.0 if d < 1.0 else 0.0
    else:
        prob_hit = math.exp(-0.5 * (d / sigma_effective) ** 2)

    # --- 5) Turn probability into YES/NO decision ---
    prob_thresh = 0.5
    hit_bool = (prob_hit > prob_thresh)
    label = "hit" if hit_bool else "miss"

    return {
        "hit_bool": hit_bool,
        "label": label,
        "distance": d,
        "eff_radius": sigma_effective,
        "sigma_major": sigma_major,
        "margin_px": 0.0,
        "prob_hit": prob_hit,
    }



def classify_hit_from_state(
    mu: np.ndarray,
    hole_center,
    hole_radius_px: float,
    ball_radius_px: float,
    margin_px: float = 0.0,
):
    """
    Pure geometry: given KF state mean and hole geometry in pixels,
    decide if the ball center is inside the shrunk hole at this instant.
    """
    if mu is None or hole_center is None:
        return False, float("nan"), 0.0

    pos = np.asarray(mu[:2], dtype=float)
    hole_c = np.asarray(hole_center, dtype=float)

    d = float(np.linalg.norm(pos - hole_c))
    eff_r = float(hole_radius_px) - float(ball_radius_px) - float(margin_px)
    eff_r = max(1.0, eff_r)  # don't let it go negative or zero

    hit = (d <= eff_r)
    return hit, d, eff_r


def make_hit_decision(
    mu: np.ndarray,
    Sigma: np.ndarray,
    hole_center,
    hole_radius_px: float,
    ball_radius_px: float,
    t_decision: float,
    t_eval: float,
    context: str,
    conf_sigma: float = 2.0,
):
    """
    Wrap a Gaussian hit classification with event metadata.
    """
    base = classify_hit_gaussian(
        mu,
        Sigma,
        hole_center,
        hole_radius_px,
        ball_radius_px,
        conf_sigma=conf_sigma,
    )
    base.update(
        {
            "t_decision": float(t_decision),
            "t_eval": float(t_eval),
            "context": str(context),
        }
    )
    return base
