import cv2
import numpy as np
import socket
import threading
from pypylon import pylon
from flask import Flask, Response

# Import the shared vision utilities (marker detection + calibration only now)
import mary_vision_common_with_yolo as vc

# --- CONFIGURATION ---
HOST = "192.168.125.207"
PORT = 5000  # Matches ABB RAPID code

# *** PHYSICAL CALIBRATION TARGETS ***
MARKER_ROBOT_X_MM = 145.6
MARKER_ROBOT_Y_MM =  205.4  

# --- YOLO MODEL (Using OpenCV DNN instead of Ultralytics) ---
print("Loading YOLO model via OpenCV DNN...")
net = cv2.dnn.readNetFromONNX("mary_best.onnx")
# Hardcode the class names since ONNX doesn't expose them automatically
CLASS_NAMES = {0: 'circle', 1: 'square', 2: 'star'}
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
YOLO_INPUT_SIZE = 640  # Standard YOLOv8 size. Change if you trained on a different size.

# --- GLOBALS ---
app = Flask(__name__)
shared_data = {"latest_objects": [], "frame": None}

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image to a square while preserving aspect ratio (padding with gray)."""
    shape = img.shape[:2]  # current shape [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2  # divide padding into 2 sides

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def process_frame(img):
    """
    Takes a frame, finds the marker, detects shapes with YOLO via OpenCV DNN,
    calculates real-world robot coordinates and orientation.
    """
    detected_objects = []

    # 1. Find the ArUco marker first — needed for pixel-to-mm conversion
    marker_info = vc.detect_marker(img)
    if marker_info is None:
        print("no marker")
        return detected_objects
    print("marker found via:", marker_info["method"])

    # 2. Run YOLO on the frame using OpenCV DNN
    # Pad the image just like Ultralytics does to preserve bounding box accuracy
    img_padded, ratio, (pad_w, pad_h) = letterbox(img, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
    
    blob = cv2.dnn.blobFromImage(img_padded, 1/255.0, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    
    # YOLOv8 ONNX output shape is [1, classes_count + 4, 8400] -> Transpose to easily iterate
    predictions = np.squeeze(outputs).T 
    
    boxes = []
    scores = []
    class_ids = []

    # Parse raw ONNX output
    for row in predictions:
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        score = classes_scores[class_id]

        if score >= CONF_THRESHOLD:
            cx, cy, w_box, h_box = row[0:4]
            
            # Revert the padding and scaling from the letterbox step
            cx = (cx - pad_w) / ratio
            cy = (cy - pad_h) / ratio
            w_box = w_box / ratio
            h_box = h_box / ratio

            # Calculate top-left corner required for NMSBoxes
            x_min = int(cx - (w_box / 2))
            y_min = int(cy - (h_box / 2))

            boxes.append([x_min, y_min, int(w_box), int(h_box)])
            scores.append(float(score))
            class_ids.append(class_id)

    # Apply Non-Maximum Suppression to filter overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
    print("YOLO detections:", len(indices) if len(indices) > 0 else 0)

    mx, my = marker_info["bbox"][0], marker_info["bbox"][1]
    mw, mh = marker_info["bbox"][2], marker_info["bbox"][3]

    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, w, h = boxes[i]
            x2, y2 = x1 + w, y1 + h
            cls_id = class_ids[i]
            shape_name = CLASS_NAMES.get(cls_id, "unknown")

            cx_box = x1 + w / 2
            cy_box = y1 + h / 2

            # Skip anything that's really just the marker itself getting detected as a shape
            if mx <= cx_box <= mx + mw and my <= cy_box <= my + mh:
                continue

            center_px = (cx_box, cy_box)

            # 3. Convert pixel offset to robot-frame mm
            dx_mm, dy_mm = vc.pixel_to_robot_offset(center_px, marker_info)
            pick_x = MARKER_ROBOT_X_MM + dx_mm
            pick_y = MARKER_ROBOT_Y_MM + dy_mm

            # 4. Orientation
            angle_deg = estimate_angle(img, (x1, y1, x2, y2), shape_name)

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
    # (Unchanged from your original code)
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
    # (Unchanged from your original code)
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
    # (Unchanged from your original code)
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
    # (Unchanged from your original code)
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
