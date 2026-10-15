import cv2
import numpy as np
import socket
import threading
from pypylon import pylon
from flask import Flask, Response
from ultralytics import YOLO

# Import the shared vision utilities (marker detection + calibration only now)
import vision_common_v8_new as vc

# --- CONFIGURATION ---
HOST = "192.168.125.207"
PORT = 5000  # Matches ABB RAPID code

# *** PHYSICAL CALIBRATION TARGETS ***
# Robot work-object frame coordinates of the marker's physical center.
# One-time hand-eye calibration anchor — NOT computed by vision. See note below.
MARKER_ROBOT_X_MM = 226.67
MARKER_ROBOT_Y_MM = 170.16
# ************************************

# --- YOLO MODEL ---
print("Loading YOLO model...")
model = YOLO("best.onnx")          # or "best.onnx" if you export that way instead
CLASS_NAMES = model.names        # {0: 'circle', 1: 'square', 2: 'star'} — comes from training, no manual mapping needed
CONF_THRESHOLD = 0.5

# --- GLOBALS ---
app = Flask(__name__)
shared_data = {"latest_objects": [], "frame": None}


def process_frame(img):
    """
    Takes a frame, finds the marker, detects shapes with YOLO,
    calculates real-world robot coordinates and orientation.
    """
    detected_objects = []

    # 1. Find the ArUco marker first — needed for pixel-to-mm conversion
    marker_info = vc.detect_marker(img)
    if marker_info is None:
        print("no marker")
        return detected_objects
    print("marker found via:", marker_info["method"])

    # 2. Run YOLO on the frame
    results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
    print("YOLO detections:", len(results.boxes))

    mx, my = marker_info["bbox"][0], marker_info["bbox"][1]
    mw, mh = marker_info["bbox"][2], marker_info["bbox"][3]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        shape_name = CLASS_NAMES[cls_id]

        # Skip anything that's really just the marker itself getting detected as a shape
        cx_box, cy_box = (x1 + x2) / 2, (y1 + y2) / 2
        if mx <= cx_box <= mx + mw and my <= cy_box <= my + mh:
            continue

        center_px = (cx_box, cy_box)

        # 3. Convert pixel offset to robot-frame mm, same as before
        dx_mm, dy_mm = vc.pixel_to_robot_offset(center_px, marker_info)
        pick_x = MARKER_ROBOT_X_MM + dx_mm
        pick_y = MARKER_ROBOT_Y_MM + dy_mm

        # 4. Orientation — YOLO boxes don't carry rotation, so estimate it
        #    from the box's contour inside the crop using minAreaRect.
        angle_deg = estimate_angle(img, (int(x1), int(y1), int(x2), int(y2)), shape_name)

        detected_objects.append((
            shape_name,
            round(pick_x, 2),
            round(pick_y, 2),
            round(angle_deg, 2),
            int(center_px[0]),
            int(center_px[1]),
        ))

    return detected_objects


SHAPE_SYMMETRY_DEGREES = {
    "circle": None,
    "square": 90.0,
    "star": 72.0,
}


def estimate_angle(img, box_xyxy, shape_name):
    """
    Crop the detected box, find the shape's contour inside it, and use
    minAreaRect to estimate rotation — same idea as vision_common's
    shape_pick_angle, just working from a YOLO box instead of a contour
    found on the full frame.
    """
    symmetry = SHAPE_SYMMETRY_DEGREES.get(shape_name)
    if not symmetry:
        return 0.0

    x1, y1, x2, y2 = box_xyxy
    pad = 5
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)
    (_, _), (_, _), raw_angle = cv2.minAreaRect(cnt)
    return float(raw_angle % symmetry)


# --- TCP SERVER ---
def tcp_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"TCP Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"ABB Robot connected from {addr}")
                while True:
                    try:
                        data = conn.recv(1024).decode('utf-8').strip()
                        if not data:
                            break
                        if data == "REQUEST_COORDS":
                            objects = shared_data.get("latest_objects", [])
                            if objects:
                                target = objects[0]
                                response = f"{target[0]},{target[1]},{target[2]},{target[3]}"
                            else:
                                response = "NO_TARGET,0,0,0"
                            conn.sendall(response.encode('utf-8'))
                            print(f"Sent to Robot: {response}")
                    except Exception as e:
                        print(f"TCP Error: {e}")
                        break


# --- WEB VIDEO STREAMER ---
def generate_video():
    while True:
        frame = shared_data.get("frame")
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')


# --- MAIN LOOP ---
def main():
    threading.Thread(target=tcp_server, daemon=True).start()

    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False), daemon=True).start()
    print("Live Video Feed available at: http://192.168.140.113:8080")

    try:
        tl_factory = pylon.TlFactory.GetInstance()
        camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
        camera.Open()
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment.Value = "MsbAligned"
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print("Camera active. Waiting for robot...")

        while camera.IsGrabbing():
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                img = converter.Convert(grabResult).GetArray()

                objects = process_frame(img)
                print("Objects found:", objects)
                shared_data["latest_objects"] = objects

                display_img = cv2.resize(img, (1024, 768))
                marker_info = vc.detect_marker(img)
                if marker_info:
                    mxp, myp = marker_info["center_px"]
                    dmx, dmy = int(mxp * (1024/img.shape[1])), int(myp * (768/img.shape[0]))
                    cv2.circle(display_img, (dmx, dmy), 5, (0, 0, 255), -1)
                    cv2.putText(display_img, "Origin", (dmx + 10, dmy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                for obj in objects:
                    shape, pick_x, pick_y, angle, cx, cy = obj
                    dx, dy = int(cx * (1024/img.shape[1])), int(cy * (768/img.shape[0]))
                    cv2.circle(display_img, (dx, dy), 5, (0, 255, 0), -1)
                    cv2.putText(display_img, f"{shape} {angle}deg", (dx + 10, dy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_img, f"X:{pick_x} Y:{pick_y}", (dx + 10, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                shared_data["frame"] = display_img

            grabResult.Release()

    except KeyboardInterrupt:
        print("\nShutting down system...")
    finally:
        if 'camera' in locals():
            camera.StopGrabbing()
            camera.Close()


if __name__ == "__main__":
    main()
