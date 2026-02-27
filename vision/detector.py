from __future__ import annotations
import numpy as np
import cv2
import math


def detect_green_ball_and_hole(frame_rgb: np.ndarray, ignore_hole: bool = False):
    """
    Detect green ball and dark circular hole in an RGB frame.

    Returns (annotated_rgb, detections_dict)
      detections_dict = {
        "ball": {"center":(x,y),"radius":r} or None,
        "hole": {"center":(x,y),"radius":r} or None
      }
    """
    h, w = frame_rgb.shape[:2]
    detections = {"ball": None, "hole": None}

    # --- GREEN BALL (HSV threshold + largest contour) ---
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    green_low  = np.array([35,  60,  60], dtype=np.uint8)
    green_high = np.array([85, 255, 255], dtype=np.uint8)

    mask_g = cv2.inRange(hsv, green_low, green_high)
    mask_g = cv2.medianBlur(mask_g, 5)
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > 0.0005 * (w * h):  # ignore tiny specks
            (x, y), r = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] > 1e-6:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                detections["ball"] = {"center": (cx, cy), "radius": int(r)}

    # --- BLACK HOLE (Hough circle; then dark/roundness fallback) ---
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

    hole = None
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h // 4,
        param1=100, param2=20, minRadius=5, maxRadius=min(h, w) // 3
    )
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        best, best_mean = None, 255.0
        for x, y, r in circles:
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            m = cv2.mean(gray, mask=mask)[0]
            if m < best_mean:
                best_mean, best = m, (int(x), int(y), int(r))
        if best is not None:
            hole = {"center": (best[0], best[1]), "radius": best[2]}

    if hole is None:
        # fallback: threshold dark blobs, pick the roundest big one
        _, mask_dark = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)
        mask_dark = cv2.medianBlur(mask_dark, 5)
        cnts, _ = cv2.findContours(mask_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 0.0005 * (w * h):
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 1e-6:
                continue
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity > 0.7:   # ~circle
                (x, y), r = cv2.minEnclosingCircle(c)
                hole = {"center": (int(x), int(y)), "radius": int(r)}
                break

    if ignore_hole:
        hole = None

    detections["hole"] = hole

    # --- disambiguate: avoid treating the ball as a "hole" ---
    if detections["ball"] is not None and detections["hole"] is not None:
        (bx, by) = detections["ball"]["center"]
        br       = float(detections["ball"]["radius"])

        (hx, hy) = detections["hole"]["center"]
        hr       = float(detections["hole"]["radius"])

        d = math.hypot(bx - hx, by - hy)

        if d < 0.7 * br and hr < 1.5 * br:
            detections["hole"] = None
            hole = None

    # --- Draw overlays (convert to BGR for cv2 drawing, then back to RGB) ---
    out_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if detections["ball"] is not None:
        (cx, cy), r = detections["ball"]["center"], detections["ball"]["radius"]
        cv2.circle(out_bgr, (cx, cy), r, (0, 0, 255), 2)
        cv2.drawMarker(out_bgr, (cx, cy), (0, 0, 255),
                       markerType=cv2.MARKER_CROSS, thickness=2)
        cv2.putText(out_bgr, f"BALL ({cx},{cy})",
                    (cx + r + 6, max(15, cy - r - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    1, cv2.LINE_AA)

    if detections["hole"] is not None:
        (hx, hy), r = detections["hole"]["center"], detections["hole"]["radius"]
        cv2.circle(out_bgr, (hx, hy), r, (255, 0, 0), 2)
        cv2.putText(out_bgr, f"HOLE ({hx},{hy})",
                    (hx + r + 6, hy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    1, cv2.LINE_AA)

    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb, detections
