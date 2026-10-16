"""
camera_setup.py - start the Basler camera and show the live view.

    python3 camera_setup.py      # then open http://192.168.140.xxx:8080

Stop with Ctrl + C before running the vision script (both use the camera and port 8080).
"""

import threading

import cv2
from flask import Flask, Response
from pypylon import pylon

app = Flask(__name__)
latest = {"frame": None}


def generate_video():
    while True:
        frame = latest["frame"]
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.route("/")
def video_feed():
    return Response(generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True).start()
    print("Live view: http://192.168.140.xxx:8080   (Ctrl + C to stop)")

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    try:
        while camera.IsGrabbing():
            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                img = converter.Convert(grab).GetArray()
                latest["frame"] = cv2.resize(img, (1024, int(1024 * img.shape[0] / img.shape[1])))
            grab.Release()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        camera.StopGrabbing()
        camera.Close()


if __name__ == "__main__":
    main()
