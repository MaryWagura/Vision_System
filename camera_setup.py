"""
camera_setup.py - live camera view for setting up the workstation.

Run this on the Raspberry Pi BEFORE the vision script, while positioning the
camera, the paper and the ArUco marker. It only streams the camera with
setup checks drawn on top - no YOLO, no robot connection.

    python3 camera_setup.py            # then open http://192.168.140.113:8080
    python3 camera_setup.py --port 8081

Checks shown on the live view (and printed in the terminal once a second):
  - Marker    : is the ArUco marker found, its id, centre and rotation
  - Tilt      : opposite sides of the marker should be equal length when the
                camera looks straight down (difference under 1 % = OK)
  - Scale     : mm per pixel from the 100 mm marker
  - Focus     : sharpness score - turn the focus ring until it is highest
  - Exposure  : average brightness and % of blown-out (pure white) pixels
  - Centre    : crosshair + grid, to centre the paper under the camera

Don't run it at the same time as the vision script: both use the camera and
port 8080. Stop it with Ctrl + C.
"""

import argparse
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response
from pypylon import pylon

# --- Marker settings (must match mary_vision_common_with_yolo.py) ---
ARUCO_DICT = cv2.aruco.DICT_5X5_100
MARKER_SIZE_MM = 100.0

# --- Pass/fail limits for the setup checks ---
TILT_OK_PCT = 1.0          # max difference between opposite marker sides (the marker is small,
                           # so a few % of tilt across the paper shows as ~1 % here)
OVEREXPOSED_MAX_PCT = 1.0  # max % of pure-white pixels
BRIGHTNESS_RANGE = (70, 200)  # comfortable average brightness (0-255)

DISPLAY_WIDTH = 1024       # live view width; height keeps the camera's aspect ratio

GREEN, RED, YELLOW, WHITE = (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 255, 255)

_detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters())

app = Flask(__name__)
shared = {"frame": None, "status": "starting..."}
lock = threading.Lock()


def analyse(img):
    """Measure everything the setup checks need from one camera frame."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = {
        "brightness": float(gray.mean()),
        "overexposed_pct": float((gray >= 250).mean() * 100.0),
        # Variance of the Laplacian: higher = sharper. Only compare it with
        # itself while turning the focus ring - the absolute value depends on the scene.
        "focus": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "marker": None,
    }

    corners, ids, _ = _detector.detectMarkers(gray)
    if ids is not None and len(ids) > 0:
        c = corners[0][0]                      # TL, TR, BR, BL
        sides = [float(np.linalg.norm(c[i] - c[(i + 1) % 4])) for i in range(4)]
        top, right, bottom, left = sides
        # Opposite sides differ when the camera is tilted (perspective)
        tilt_lr = abs(top - bottom) / max(top, bottom) * 100.0     # tilt towards top/bottom
        tilt_ud = abs(left - right) / max(left, right) * 100.0     # tilt towards left/right
        edge = c[1] - c[0]
        result["marker"] = {
            "id": int(np.asarray(ids).flatten()[0]),   # shape differs between OpenCV versions
            "corners": c,
            "center": c.mean(axis=0),
            "mm_per_px": MARKER_SIZE_MM / float(np.mean(sides)),
            "angle_deg": float(np.degrees(np.arctan2(edge[1], edge[0]))),
            "tilt_pct": max(tilt_lr, tilt_ud),
            "tilt_top_bottom_pct": tilt_lr,
            "tilt_left_right_pct": tilt_ud,
            "count": len(ids),
        }
    return result


def draw_overlay(img, r):
    """Draw the setup checks on a display-sized copy of the frame."""
    h, w = img.shape[:2]
    scale = DISPLAY_WIDTH / w
    disp = cv2.resize(img, (DISPLAY_WIDTH, int(round(h * scale))))
    dh, dw = disp.shape[:2]

    # Grid (thirds) + centre crosshair to centre the paper under the camera
    for k in (1, 2):
        cv2.line(disp, (dw * k // 3, 0), (dw * k // 3, dh), (90, 90, 90), 1)
        cv2.line(disp, (0, dh * k // 3), (dw, dh * k // 3), (90, 90, 90), 1)
    cx, cy = dw // 2, dh // 2
    cv2.line(disp, (cx - 25, cy), (cx + 25, cy), YELLOW, 2)
    cv2.line(disp, (cx, cy - 25), (cx, cy + 25), YELLOW, 2)

    lines = []
    m = r["marker"]
    if m is None:
        lines.append(("Marker: NOT FOUND - check it is in view and well lit", RED))
    else:
        pts = (m["corners"] * scale).astype(np.int32)
        cv2.polylines(disp, [pts], True, GREEN, 2)
        for name, p in zip(("TL", "TR", "BR", "BL"), pts):
            cv2.putText(disp, name, (int(p[0]) + 4, int(p[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)
        mc = (m["center"] * scale).astype(int)
        cv2.circle(disp, (int(mc[0]), int(mc[1])), 5, RED, -1)
        cv2.line(disp, (cx, cy), (int(mc[0]), int(mc[1])), (180, 180, 180), 1)

        extra = f" ({m['count']} markers seen!)" if m["count"] > 1 else ""
        lines.append((f"Marker: id {m['id']}  rotation {m['angle_deg']:.1f} deg{extra}",
                      GREEN if m["count"] == 1 else YELLOW))
        tilt_ok = m["tilt_pct"] <= TILT_OK_PCT
        lines.append((f"Tilt: {m['tilt_pct']:.1f} % "
                      f"(top/bottom {m['tilt_top_bottom_pct']:.1f}, left/right {m['tilt_left_right_pct']:.1f}) "
                      f"{'OK' if tilt_ok else '- straighten camera'}", GREEN if tilt_ok else RED))
        lines.append((f"Scale: {m['mm_per_px']:.3f} mm/px", WHITE))

    bright_ok = BRIGHTNESS_RANGE[0] <= r["brightness"] <= BRIGHTNESS_RANGE[1]
    over_ok = r["overexposed_pct"] <= OVEREXPOSED_MAX_PCT
    lines.append((f"Brightness: {r['brightness']:.0f}  blown-out: {r['overexposed_pct']:.1f} % "
                  f"{'OK' if bright_ok and over_ok else '- adjust light/exposure'}",
                  GREEN if bright_ok and over_ok else RED))
    lines.append((f"Focus: {r['focus']:.0f}  (turn the ring until this is highest)", WHITE))

    y = 26
    for text, colour in lines:
        cv2.putText(disp, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(disp, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        y += 26
    return disp, " | ".join(t for t, _ in lines)


def generate_video():
    while True:
        with lock:
            frame = shared["frame"]
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.05)


@app.route("/")
def video_feed():
    return Response(generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    parser = argparse.ArgumentParser(description="Live camera view for workstation setup")
    parser.add_argument("--port", type=int, default=8080, help="web port (default 8080)")
    args = parser.parse_args()

    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=args.port, debug=False,
                                            use_reloader=False), daemon=True).start()
    print(f"Camera setup view: http://192.168.140.113:{args.port}   (Ctrl + C to stop)")

    camera = None
    try:
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        print("Camera:", camera.GetDeviceInfo().GetModelName())
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment.Value = "MsbAligned"
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        last_print = 0.0
        while camera.IsGrabbing():
            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                img = converter.Convert(grab).GetArray()
                disp, status = draw_overlay(img, analyse(img))
                with lock:
                    shared["frame"] = disp
                if time.time() - last_print >= 1.0:
                    print(f"[{img.shape[1]}x{img.shape[0]}] {status}")
                    last_print = time.time()
            grab.Release()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if camera is not None and camera.IsOpen():
            camera.StopGrabbing()
            camera.Close()


if __name__ == "__main__":
    main()
