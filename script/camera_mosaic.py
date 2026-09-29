#!/usr/bin/env python3
"""Show the three ROS RGB camera streams in one OpenCV window."""

import argparse
import threading

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", default="/camera_f/color/image_raw")
    parser.add_argument("--left", default="/camera_l/color/image_raw")
    parser.add_argument("--right", default="/camera_r/color/image_raw")
    parser.add_argument("--width", type=int, default=640, help="Main/front display width")
    return parser.parse_args()


class MosaicViewer:
    def __init__(self, args):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.images = {"front": None, "left": None, "right": None}
        for label, topic in (("front", args.front), ("left", args.left), ("right", args.right)):
            rospy.Subscriber(topic, Image, self._callback, callback_args=label, queue_size=1)
        self.width = args.width

    def _callback(self, message, label):
        try:
            rgb = self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")
            with self.lock:
                self.images[label] = rgb
        except Exception as exc:
            rospy.logwarn_throttle(2.0, f"{label} image conversion failed: {exc}")

    @staticmethod
    def _render_panel(image, label, width, height, main=False):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        if image is None:
            text = f"waiting: {label}"
            text_scale = 0.8 if main else 0.6
            cv2.putText(
                panel,
                text,
                (20, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                2,
            )
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            scale = min(width / bgr.shape[1], height / bgr.shape[0])
            resized_width = max(1, round(bgr.shape[1] * scale))
            resized_height = max(1, round(bgr.shape[0] * scale))
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            resized = cv2.resize(bgr, (resized_width, resized_height), interpolation=interpolation)
            x = (width - resized_width) // 2
            y = (height - resized_height) // 2
            panel[y : y + resized_height, x : x + resized_width] = resized

        title = "FRONT (main)" if main else label.upper()
        text_scale = 0.8 if main else 0.6
        cv2.rectangle(panel, (0, 0), (width, 42 if main else 32), (0, 0, 0), -1)
        cv2.putText(
            panel,
            title,
            (14, 30 if main else 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 255, 0),
            2,
        )
        return panel

    def _compose(self, snapshot):
        front_width = max(2, self.width)
        left_width = front_width // 2
        right_width = front_width - left_width
        front_height = max(1, round(front_width * 3 / 4))
        side_height = max(1, round(front_height / 2))

        front = self._render_panel(
            snapshot["front"], "front", front_width, front_height, main=True
        )
        left = self._render_panel(snapshot["left"], "left", left_width, side_height)
        right = self._render_panel(snapshot["right"], "right", right_width, side_height)
        bottom = np.hstack((left, right))
        return np.vstack((front, bottom))

    def run(self):
        window = "PiPER cameras - FRONT main / LEFT RIGHT (q closes preview only)"
        # Use the compact GUI and lock the client area to the mosaic size.  The
        # default resizable Qt window can leave a white toolbar/status area at
        # the bottom and cover part of the lower camera row.
        window_flags = cv2.WINDOW_AUTOSIZE
        if hasattr(cv2, "WINDOW_GUI_NORMAL"):
            window_flags |= cv2.WINDOW_GUI_NORMAL
        cv2.namedWindow(window, window_flags)
        while not rospy.is_shutdown():
            with self.lock:
                snapshot = dict(self.images)
            cv2.imshow(window, self._compose(snapshot))
            if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                break
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    rospy.init_node("piper_camera_mosaic", anonymous=True)
    MosaicViewer(args).run()


if __name__ == "__main__":
    main()
