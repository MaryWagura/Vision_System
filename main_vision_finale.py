import cv2
import numpy as np
import socket
import threading
import time
from pypylon import pylon
from flask import Flask, Response

# Import the shared vision utilities (marker detection + calibration only now)
import mary_vision_common_with_yolo as vc

# --- CONFIGURATION ---
HOST = "192.168.125.207"
PORT = 5000  # Matches ABB RAPID code

# *** PHYSICAL CALIBRATION TARGETS ***
# Must match the X/Y of the RAPID 'aruco' target (tool1234, wobj1212)
MARKER_ROBOT_X_MM = 118.64
MARKER_ROBOT_Y_MM = 193.32

# --- GRADE D: colour AND shape are chosen on the FlexPendant each cycle ---
# Robot commands (the pendant only offers what the camera actually sees):
#   LIST_COLORS                       -> "blue,red"         (or "NONE")
#   LIST_SHAPES,<colour>              -> "square,star"      (or "NONE")
#   REQUEST_COORDS,<colour>,<shape>   -> "square,x,y,angle" (or "NO_TARGET,0,0,0")
# Any field may be left out or set to "any" ("REQUEST_COORDS" = any shape).
VALID_COLORS = ("black", "red", "green", "blue", "white")
VALID_SHAPES = ("circle", "square", "star")
FRESH_FRAMES = 2          # answer only from frames taken AFTER the request
FRESH_TIMEOUT_S = 5.0     # give up waiting for fresh frames after this long

# --- YOLO MODEL (Using OpenCV DNN instead of Ultralytics) ---
print("Loading YOLO model via OpenCV DNN...")
net = cv2.dnn.readNetFromONNX("best.onnx")

# --- LEVEL III: inference time must not exceed 3 seconds ---
INFERENCE_LIMIT_S = 3.0
INFERENCE_LOG = "inference_log.csv"   # one row per frame - evidence for the report
# Hardcode the class names since ONNX doesn't expose them automatically
CLASS_NAMES = {0: 'circle', 1: 'square', 2: 'star'}
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
YOLO_INPUT_SIZE = 640  # Standard YOLOv8 size. Change if you trained on a different size.

# --- GLOBALS ---
app = Flask(__name__)
shared_data = {"latest_objects": [], "frame": None, "frame_id": 0, "last_request": "-",
               "latest_timing": None}
inference_stats = {"frames": 0, "sum_s": 0.0, "max_s": 0.0, "over_limit": 0}


def warm_up_model():
    """The first forward pass is always slow (memory allocation); do it at startup."""
    dummy = np.zeros((1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), np.float32)
    t0 = time.perf_counter()
    net.setInput(dummy)
    net.forward()
    print(f"Model warm-up: {(time.perf_counter() - t0) * 1000:.0f} ms (not counted)")


def report_inference_time(timing, n_objects):
    """Print the inference time for this frame, check the 3 s limit, log it to CSV."""
    total_s = timing["total_s"]
    model_s = timing["model_s"]
    if model_s is None:
        # No marker -> YOLO never ran; don't let these frames distort the stats
        print("[INFERENCE] skipped (no marker in view)")
        return
    inference_stats["frames"] += 1
    inference_stats["sum_s"] += total_s
    inference_stats["max_s"] = max(inference_stats["max_s"], total_s)
    ok = total_s <= INFERENCE_LIMIT_S
    if not ok:
        inference_stats["over_limit"] += 1

    model_txt = f"{model_s * 1000:.0f} ms"
    avg_s = inference_stats["sum_s"] / inference_stats["frames"]
    print(f"[INFERENCE] YOLO model: {model_txt} | total detection: {total_s * 1000:.0f} ms "
          f"| {n_objects} shape(s) | {'OK' if ok else 'OVER'} (limit {INFERENCE_LIMIT_S:.0f} s) "
          f"| avg {avg_s * 1000:.0f} ms, max {inference_stats['max_s'] * 1000:.0f} ms, "
          f"over limit {inference_stats['over_limit']}/{inference_stats['frames']}")
    if not ok:
        print(f"[WARNING] Inference took {total_s:.2f} s - more than the {INFERENCE_LIMIT_S:.0f} s limit!")

    try:
        new_file = inference_stats["frames"] == 1
        with open(INFERENCE_LOG, "a") as f:
            if new_file:
                f.write("timestamp,model_ms,total_ms,shapes,within_limit\n")
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},"
                    f"{round(model_s * 1000, 1)},"
                    f"{round(total_s * 1000, 1)},{n_objects},{ok}\n")
    except OSError as e:
        print(f"Could not write {INFERENCE_LOG}: {e}")


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


# --- Geometric check: reject YOLO detections whose outline doesn't match ---
STAR_MAX_SOLIDITY = 0.80    # star ~0.5-0.6, everything else ~1.0
CIRCLE_MIN_FILL = 0.86      # area / min enclosing circle: circle ~0.92, hexagon ~0.80 (measured)
SQUARE_MIN_EXTENT = 0.85    # area / min-area rectangle: square ~1.0
DEBUG_GEOMETRY = True       # print measurements so you can tune the thresholds


def _largest_blob(mask, w, h, x0, y0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.2 * w * h:
        return None
    return cnt + np.array([[x0, y0]])


def shape_contour_in_box(img, box_xywh, pad=5):
    """Outline of the shape inside a YOLO box, in full-image pixel coordinates."""
    x, y, w, h = box_xywh
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    # 1) Shapes darker than the white paper (black / red / green / blue)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnt = _largest_blob(mask, w, h, x0, y0)
    if cnt is not None:
        return cnt

    # 2) White shapes on white paper: fall back to the shape's edges/shadow
    edges = cv2.Canny(gray, 30, 90)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return _largest_blob(edges, w, h, x0, y0)


def geometry_matches(cnt, shape_name):
    """True only if the outline's geometry agrees with YOLO's class."""
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if area == 0 or peri == 0:
        return False

    solidity = area / max(cv2.contourArea(cv2.convexHull(cnt)), 1e-6)
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    extent = area / max(rw * rh, 1e-6)
    aspect = min(rw, rh) / max(rw, rh, 1e-6)
    (_, _), r = cv2.minEnclosingCircle(cnt)
    circle_fill = area / max(np.pi * r * r, 1e-6)
    corners = len(cv2.approxPolyDP(cnt, 0.04 * peri, True))

    if DEBUG_GEOMETRY:
        print(f"  {shape_name}: corners={corners} solidity={solidity:.2f} "
              f"extent={extent:.2f} aspect={aspect:.2f} circle_fill={circle_fill:.2f}")

    if shape_name == "star":
        return solidity < STAR_MAX_SOLIDITY
    if shape_name == "circle":
        return solidity >= STAR_MAX_SOLIDITY and circle_fill >= CIRCLE_MIN_FILL
    if shape_name == "square":
        return corners == 4 and extent >= SQUARE_MIN_EXTENT and aspect >= 0.8
    return False


# --- Colour classification (HSV; OpenCV hue runs 0-179) ---
COLOR_HUE_RANGES = {
    "red":   [(0, 12), (165, 179)],   # red wraps around 0
    "green": [(35, 90)],
    "blue":  [(95, 135)],
}
COLOR_MIN_SAT = 70        # pixels below this saturation count as grey (black/white)
COLOR_MIN_VAL = 40        # very dark pixels have unreliable hue - ignore them for colour
COLORED_MIN_FRACTION = 0.35   # share of coloured pixels needed to call it red/green/blue
BLACK_MAX_RATIO = 0.45    # shape brightness / paper brightness below this = black
DEBUG_COLOR = True        # print measurements so you can tune the thresholds

# BGR colours for the on-screen labels
DRAW_COLORS = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 120, 0),
               "black": (60, 60, 60), "white": (255, 255, 255)}


def classify_color(img, cnt):
    """
    Colour of the shape inside the contour: 'black', 'red', 'green', 'blue',
    'white', or 'unknown'. Samples the inside of the shape only (eroded away
    from the edge and shadow) and compares brightness with the paper around it.
    """
    x, y, w, h = cv2.boundingRect(cnt)
    pad = max(10, int(0.3 * max(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    crop = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    shape_mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.drawContours(shape_mask, [cnt - np.array([[x0, y0]])], -1, 255, -1)

    # Inside of the shape, away from edges and shadows
    k = max(3, int(0.15 * min(w, h)) | 1)
    inner = cv2.erode(shape_mask, np.ones((k, k), np.uint8)) > 0
    if inner.sum() < 20:
        inner = shape_mask > 0

    # Paper around the shape (outside a dilated outline) for a brightness reference
    outer = cv2.dilate(shape_mask, np.ones((k, k), np.uint8)) == 0
    paper_v = float(np.percentile(V[outer], 75)) if outer.sum() > 20 else 255.0

    h_in, s_in, v_in = H[inner], S[inner], V[inner]
    colored = (s_in >= COLOR_MIN_SAT) & (v_in >= COLOR_MIN_VAL)
    colored_frac = float(colored.mean())
    v_ratio = float(np.median(v_in)) / max(paper_v, 1.0)

    color = "unknown"
    if colored_frac >= COLORED_MIN_FRACTION:
        hues = h_in[colored]
        votes = {}
        for name, ranges in COLOR_HUE_RANGES.items():
            votes[name] = int(sum(((hues >= lo) & (hues <= hi)).sum() for lo, hi in ranges))
        best = max(votes, key=votes.get)
        if votes[best] >= 0.5 * len(hues):
            color = best
    elif v_ratio < BLACK_MAX_RATIO:
        color = "black"
    else:
        color = "white"

    if DEBUG_COLOR:
        print(f"  colour={color}: coloured_frac={colored_frac:.2f} "
              f"median_hue={float(np.median(h_in)):.0f} v_ratio={v_ratio:.2f}")
    return color


SHAPE_SYMMETRY_DEGREES = {
    "circle": None,
    "square": 90.0,
    "star": 72.0,
}


def shape_angle(cnt, shape_name):
    """Gripper rotation for the shape, reduced by its symmetry."""
    symmetry = SHAPE_SYMMETRY_DEGREES.get(shape_name)
    if not symmetry:
        return 0.0
    (_, _), (_, _), raw_angle = cv2.minAreaRect(cnt)
    return float(raw_angle % symmetry)


def process_frame(img):
    """
    Finds the marker, detects shapes with YOLO, checks shape geometry and
    identifies each shape's colour. Returns (objects, ignored, timing).
      objects: (shape, pick_x, pick_y, angle, cx, cy, color) - every valid shape
      ignored: (label, cx, cy) - wrong-geometry detections, shown on the feed only
      timing:  {"model_s": YOLO forward pass, "total_s": frame in -> results out}
    """
    t_start = time.perf_counter()
    timing = {"model_s": None, "total_s": None}
    detected_objects = []
    ignored = []

    # 1. Find the ArUco marker first - needed for pixel-to-mm conversion
    marker_info = vc.detect_marker(img)
    if marker_info is None:
        print("no marker")
        timing["total_s"] = time.perf_counter() - t_start
        return detected_objects, ignored, timing

    # 2. Run YOLO on the frame using OpenCV DNN
    img_padded, ratio, (pad_w, pad_h) = letterbox(img, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
    blob = cv2.dnn.blobFromImage(img_padded, 1/255.0, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), swapRB=True, crop=False)
    t_model = time.perf_counter()
    net.setInput(blob)
    outputs = net.forward()
    timing["model_s"] = time.perf_counter() - t_model

    # YOLOv8 ONNX output shape is [1, classes_count + 4, 8400] -> one row per candidate box
    predictions = np.squeeze(outputs).T

    # Vectorised decoding (much faster on the Pi than looping over 8400 rows)
    class_scores = predictions[:, 4:]
    class_ids_all = class_scores.argmax(axis=1)
    conf_all = class_scores[np.arange(len(class_scores)), class_ids_all]
    keep = conf_all >= CONF_THRESHOLD
    p = predictions[keep]
    # Revert the padding and scaling from the letterbox step
    cx = (p[:, 0] - pad_w) / ratio
    cy = (p[:, 1] - pad_h) / ratio
    w_box = p[:, 2] / ratio
    h_box = p[:, 3] / ratio
    boxes = np.stack([cx - w_box / 2, cy - h_box / 2, w_box, h_box], axis=1).astype(int).tolist()
    scores = conf_all[keep].astype(float).tolist()
    class_ids = class_ids_all[keep].tolist()

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
    print("YOLO detections:", len(indices) if len(indices) > 0 else 0)

    mx, my, mw, mh = marker_info["bbox"]

    for i in (indices.flatten() if len(indices) > 0 else []):
        x1, y1, w, h = boxes[i]
        shape_name = CLASS_NAMES.get(class_ids[i], "unknown")
        cx_box, cy_box = x1 + w / 2, y1 + h / 2

        # Skip anything that's really just the marker itself getting detected as a shape
        if mx <= cx_box <= mx + mw and my <= cy_box <= my + mh:
            continue

        # Reject shapes YOLO has never been trained on (triangle, hexagon, ...)
        cnt = shape_contour_in_box(img, boxes[i])
        if cnt is None or not geometry_matches(cnt, shape_name):
            print(f"Ignored: YOLO said '{shape_name}' but geometry doesn't match")
            ignored.append(("not " + shape_name, int(cx_box), int(cy_box)))
            continue

        # Use the outline's true centre instead of the box centre
        m = cv2.moments(cnt)
        center_px = (m["m10"] / m["m00"], m["m01"] / m["m00"])

        # Grade D: identify the colour (the robot chooses which colour to pick)
        color = classify_color(img, cnt)

        # 3. Convert pixel offset to robot-frame mm (wobj1212)
        mx_c, my_c = marker_info["center_px"]
        s = marker_info["mm_per_px"]
        right_mm = (center_px[0] - mx_c) * s   # + = right in image
        down_mm = (center_px[1] - my_c) * s    # + = down in image

        pick_x = MARKER_ROBOT_X_MM + down_mm    # robot +X = image down
        pick_y = MARKER_ROBOT_Y_MM + right_mm   # robot +Y = image right

        # 4. Orientation from the clean outline
        angle_deg = shape_angle(cnt, shape_name)

        detected_objects.append((
            shape_name,
            round(pick_x, 2),
            round(pick_y, 2),
            round(angle_deg, 2),
            int(center_px[0]),
            int(center_px[1]),
            color,
        ))

    timing["total_s"] = time.perf_counter() - t_start
    return detected_objects, ignored, timing


# --- TCP SERVER ---
def wait_for_fresh_objects():
    """
    Wait until FRESH_FRAMES new frames have been processed since the request,
    so the answer never comes from a frame taken while the robot was still
    moving the previous shape (or blocking the camera).
    """
    start = shared_data["frame_id"]
    deadline = time.time() + FRESH_TIMEOUT_S
    while shared_data["frame_id"] < start + FRESH_FRAMES and time.time() < deadline:
        time.sleep(0.05)
    return list(shared_data.get("latest_objects", []))


def choose_target(objects, color, shape="any"):
    """
    One object of the requested colour and shape. If several match (e.g. two
    red squares), take the one closest to the marker; the next cycle picks
    the next one.
    """
    matches = [o for o in objects
               if (color == "any" or o[6] == color) and (shape == "any" or o[0] == shape)]
    if not matches:
        return None
    return min(matches, key=lambda o: (o[1] - MARKER_ROBOT_X_MM) ** 2 + (o[2] - MARKER_ROBOT_Y_MM) ** 2)


RAPID_MAX_STR = 80  # RAPID strings cannot be longer than 80 characters


def handle_command(data):
    """Answer one robot command (see the command list at the top of the file)."""
    parts = [x.strip().lower() for x in data.split(",")]
    command = parts[0]

    if command == "list_colors":
        # First question of each cycle, asked right after the robot has moved:
        # wait for frames taken AFTER the request. Only valid, pickable colours.
        objects = wait_for_fresh_objects()
        colors = sorted({o[6] for o in objects if o[6] in VALID_COLORS})
        shared_data["last_request"] = "colour list"
        reply = ",".join(colors) if colors else "NONE"

    elif command == "list_shapes":
        # Asked right after LIST_COLORS while the robot stands still -> latest frame is fresh
        color = parts[1] if len(parts) > 1 and parts[1] else "any"
        objects = list(shared_data.get("latest_objects", []))
        shapes = sorted({o[0] for o in objects
                         if o[0] in VALID_SHAPES and (color == "any" or o[6] == color)})
        shared_data["last_request"] = f"{color} shape list"
        reply = ",".join(shapes) if shapes else "NONE"

    elif command == "request_coords":
        # "REQUEST_COORDS,red,square" -> red square; missing fields = any
        color = parts[1] if len(parts) > 1 and parts[1] else "any"
        shape = parts[2] if len(parts) > 2 and parts[2] else "any"
        shared_data["last_request"] = f"{color} {shape}"

        target = None
        if color != "any" and color not in VALID_COLORS:
            print(f"Unknown colour requested: '{color}'")
        elif shape != "any" and shape not in VALID_SHAPES:
            print(f"Unknown shape requested: '{shape}'")
        else:
            # Robot has been waiting at the pendant menus -> latest frame is fresh
            target = choose_target(list(shared_data.get("latest_objects", [])), color, shape)
        reply = f"{target[0]},{target[1]},{target[2]},{target[3]}" if target else "NO_TARGET,0,0,0"

    else:
        reply = "UNKNOWN_COMMAND"

    if len(reply) > RAPID_MAX_STR:
        # Cut at the last complete item so RAPID can still store it
        reply = reply[:RAPID_MAX_STR].rsplit(",", 1)[0]
    return reply


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
                        print(f"Robot asked: {data}")
                        response = handle_command(data)
                        conn.sendall(response.encode('utf-8'))
                        print(f"Sent to Robot: {response}")
                        t = shared_data.get("latest_timing")
                        if t and t["total_s"] is not None:
                            print(f"  (from a frame with inference time {t['total_s'] * 1000:.0f} ms)")
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
    warm_up_model()
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

                objects, ignored, timing = process_frame(img)
                print("Objects found:", objects)
                report_inference_time(timing, len(objects))
                shared_data["latest_objects"] = objects
                shared_data["latest_timing"] = timing
                shared_data["frame_id"] += 1

                sx, sy = 1024 / img.shape[1], 768 / img.shape[0]
                display_img = cv2.resize(img, (1024, 768))
                cv2.putText(display_img, f"Robot asked for: {shared_data['last_request']}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                within = timing["total_s"] <= INFERENCE_LIMIT_S
                cv2.putText(display_img, f"Inference: {timing['total_s'] * 1000:.0f} ms "
                            f"(limit {INFERENCE_LIMIT_S:.0f} s) {'OK' if within else 'OVER'}",
                            (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if within else (0, 0, 255), 2)

                marker_info = vc.detect_marker(img)
                if marker_info:
                    mxp, myp = marker_info["center_px"]
                    dmx, dmy = int(mxp * sx), int(myp * sy)
                    cv2.circle(display_img, (dmx, dmy), 5, (0, 0, 255), -1)
                    cv2.putText(display_img, "Origin", (dmx + 10, dmy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Every valid shape: green dot + colour/shape + robot X/Y
                for shape, pick_x, pick_y, angle, cx, cy, color in objects:
                    dx, dy = int(cx * sx), int(cy * sy)
                    cv2.circle(display_img, (dx, dy), 6, (0, 255, 0), -1)
                    cv2.putText(display_img, f"{color} {shape} {angle}deg", (dx + 10, dy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_img, f"X:{pick_x} Y:{pick_y}", (dx + 10, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Ignored detections (not circle/square/star): grey cross + label
                for label, cx, cy in ignored:
                    dx, dy = int(cx * sx), int(cy * sy)
                    cv2.drawMarker(display_img, (dx, dy), (160, 160, 160), cv2.MARKER_TILTED_CROSS, 14, 2)
                    cv2.putText(display_img, f"ignored: {label}", (dx + 10, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 2)

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
