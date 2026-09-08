import os
import time
import threading
import cv2
from flask import Flask, Response, render_template_string
from pypylon import pylon

# --- Configuration ---
NUM_IMAGES = 150
DELAY_SECONDS = 10
SAVE_FOLDER = "dataset"
SHAPE_NAME = "capture"  # Update prefix per shape (e.g., "Hexagon", "Square", "Circle", "Star")
FLASK_PORT = 5000

# --- Global Frame Buffer & Lock ---
output_frame = None
frame_lock = threading.Lock()

app = Flask(__name__)

# --- Web Interface ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Basler Live Capture Feed</title>
    <style>
        body { background-color: #1a1a1a; color: #f0f0f0; font-family: Arial, sans-serif; text-align: center; margin: 0; padding: 20px; }
        h1 { margin-bottom: 10px; }
        .stream-container { display: inline-block; border: 3px solid #00ffcc; border-radius: 8px; overflow: hidden; }
        img { width: 100%; max-width: 800px; height: auto; display: block; }
    </style>
</head>
<body>
    <h1>Basler Live Dataset Collection</h1>
    <p>Monitor position below. High-resolution raw frames are saved automatically every 10 seconds.</p>
    <div class="stream-container">
        <img src="/video_feed" alt="Live Camera Stream">
    </div>
</body>
</html>
"""

def generate_stream():
    global output_frame, frame_lock
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            # Encode frame to JPEG for HTTP transmission
            success, encoded_image = cv2.imencode(".jpg", output_frame)
            if not success:
                continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
        time.sleep(0.03)  # Stream throttle (~30 FPS)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, use_reloader=False)

# --- Camera & Capture Loop ---
def main():
    global output_frame, frame_lock

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    # Start Flask web server in a background daemon thread
    server_thread = threading.Thread(target=run_flask, daemon=True)
    server_thread.start()
    print(f"\n[INFO] Live feed active at: http://192.168.140.113:{FLASK_PORT}\n")

    try:
        tl_factory = pylon.TlFactory.GetInstance()
        camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())

        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        camera.Open()
        print(f"Connected to: {camera.GetDeviceInfo().GetModelName()}")
        
        # Grab continuously keeping only the latest frame in buffer (zero lag)
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print(f"Target: {NUM_IMAGES} images with {DELAY_SECONDS}s interval.\n")

        saved_count = 0
        last_capture_time = time.time()

        while camera.IsGrabbing() and saved_count < NUM_IMAGES:
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                pylon_img = converter.Convert(grab_result)
                frame = pylon_img.GetArray()

                current_time = time.time()
                elapsed = current_time - last_capture_time
                time_remaining = max(0, int(DELAY_SECONDS - elapsed))

                # Periodic save trigger
                if elapsed >= DELAY_SECONDS:
                    saved_count += 1
                    filename = os.path.join(SAVE_FOLDER, f"{SHAPE_NAME}_{saved_count}.jpg")
                    
                    # Save the clean frame (without text overlays) to disk
                    cv2.imwrite(filename, frame)
                    print(f"[{saved_count}/{NUM_IMAGES}] Saved: {filename}")
                    last_capture_time = current_time

                # Create preview frame with HUD overlay for the web stream
                preview = frame.copy()
                cv2.putText(preview, f"Saved: {saved_count}/{NUM_IMAGES}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(preview, f"Next capture: {time_remaining}s", (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                # Flash indicator on the frame right when saved
                if elapsed < 0.5 and saved_count > 0:
                    cv2.putText(preview, "CAPTURED!", (30, 145),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

                with frame_lock:
                    output_frame = preview

            grab_result.Release()

        camera.StopGrabbing()
        camera.Close()
        print("\nDataset collection complete!")

    except Exception as e:
        print(f"\nHardware/Pylon error: {e}")
        print("Verify camera connection to the blue USB 3.0 port.")

if __name__ == "__main__":
    main()
