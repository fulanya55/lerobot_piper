#!/usr/bin/env python3
"""Low-overhead MJPEG preview bridge for the three ROS camera topics."""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

TOPICS = {"front": "/camera_f/color/image_raw", "left": "/camera_l/color/image_raw", "right": "/camera_r/color/image_raw"}
latest = {key: None for key in TOPICS}
stamps = {key: 0.0 for key in TOPICS}
lock = threading.Lock()
bridge = CvBridge()


def callback(message, key):
    try:
        image = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        h, w = image.shape[:2]
        if w > 640:
            image = cv2.resize(image, (640, round(h * 640 / w)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with lock:
                latest[key] = encoded.tobytes(); stamps[key] = time.time()
    except Exception as exc:
        rospy.logwarn_throttle(5.0, f"{key} preview: {exc}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return
    def do_GET(self):
        key = self.path.rsplit("/", 1)[-1]
        if key not in TOPICS:
            self.send_error(404); return
        self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
        try:
            while not rospy.is_shutdown():
                with lock: frame = latest[key]
                if frame:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                    self.wfile.flush()
                time.sleep(0.08)
        except (BrokenPipeError, ConnectionResetError):
            pass


rospy.init_node("piper_preview_server", anonymous=True, disable_signals=True)
for name, topic in TOPICS.items():
    rospy.Subscriber(topic, Image, callback, callback_args=name, queue_size=1, buff_size=2**24)
port = 8766
print(f"preview server: http://127.0.0.1:{port}/stream/front", flush=True)
ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
