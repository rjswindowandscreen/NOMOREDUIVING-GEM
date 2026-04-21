import os
import sys

import torch
import json
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.parameter import Parameter

from worldgt import WorldGT
from line_fit import lane_fit, final_viz, perspective_transform, \
                    closest_point_on_polynomial, Line
from model_utils import load_model, inference
import rich
import cv2
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray

SMOOTH_N = 5

# Max allowed x-position jump between frames (pixels at bottom of BEV).
# Increase if turns cause false discards. Decrease if noise jumps through.
MAX_POSITION_JUMP = 150


class LaneVisualizer(Node):
    def __init__(self):
        super().__init__("lane_visualizer")
        self._lane_error_pub = self.create_publisher(Float32MultiArray, '/lane_error', 10)
        sim_time_param = Parameter('use_sim_time', Parameter.Type.BOOL, True)
        self.set_parameters([sim_time_param])

        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._model = load_model()
            self._model = self._model.to(self._dev).eval()
            rich.print("[green]loaded SimpleEnet :o")
        except Exception as e:
            self.get_logger().error(f"could not load SimpleEnet model: {e}")
            exit(1)

        try:
            with open(os.path.join("data", "bev_config.json")) as f:
                self._bev_cfg = json.load(f)
        except FileNotFoundError as e:
            self.get_logger().error(f"could not load bev config: {e}")
            exit(1)

        self._world = None
        try:
            self._world = WorldGT("Silverstone")
        except Exception as e:
            self.get_logger().warn(f"WorldGT unavailable: {e}")

        self._tf_buf      = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)
        self._image_msg   = None
        self._cv_bridge   = CvBridge()

        # Temporal smoothing
        self._left_line  = Line(n=SMOOTH_N)
        self._right_line = Line(n=SMOOTH_N)

        # Dynamic lane width tracking
        self._lane_width_px   = None
        self._lane_width_hist = []
        self._lane_width_n    = 30

        # Publishers
        self._render_pub = self.create_publisher(Image, '/lane_render',    10)
        self._bev_pub    = self.create_publisher(Image, '/lane_bev',       10)
        self._debug_pub  = self.create_publisher(Image, '/lane_bev_debug', 10)

        self.create_subscription(
            Image, "/camera/image_raw", self._on_image, 10
        )

    def _update_lane_width(self, left_fit, right_fit, y_bottom):
        left_x   = float(np.polyval(left_fit,  y_bottom))
        right_x  = float(np.polyval(right_fit, y_bottom))
        width_px = abs(right_x - left_x)
        if 50 < width_px < 700:
            self._lane_width_hist.append(width_px)
            if len(self._lane_width_hist) > self._lane_width_n:
                self._lane_width_hist.pop(0)
            self._lane_width_px = float(np.mean(self._lane_width_hist))

    def _position_is_consistent(self, fit, line_tracker, y_bottom):
        """
        Check 2 of Option 4: reject fits that jump too far from last frame.
        last_x_bottom is now the actual polyval result, not just C.
        """
        if not line_tracker.detected or line_tracker.last_x_bottom is None:
            return True
        new_x  = float(np.polyval(fit, y_bottom))
        last_x = line_tracker.last_x_bottom
        return abs(new_x - last_x) <= MAX_POSITION_JUMP

    def _on_image(self, msg) -> None:
        self._image_msg = msg
        if self._model is None:
            return

        image = self._cv_bridge.imgmsg_to_cv2(self._image_msg, "bgr8")

        src_pts   = np.float32(self._bev_cfg["src"])
        debug_img = image.copy()
        for pt in src_pts:
            cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)
        cv2.polylines(debug_img, [src_pts.astype(np.int32)],
                      isClosed=True, color=(0, 255, 0), thickness=2)
        self._debug_pub.publish(
            self._cv_bridge.cv2_to_imgmsg(debug_img, "bgr8")
        )

        mask = inference(self._model, image, self._dev)
        m    = mask.astype(np.uint8) * 255

        combine_fit_img, binary_BEV, left_fit, right_fit = \
            self.fit_poly_lanes(image, m)

        binary_BEV = np.pad(binary_BEV, ((0, 100), (0, 0)))
        binary_BEV = cv2.cvtColor(binary_BEV, cv2.COLOR_GRAY2BGR)

        XTE_str = "N/A"
        HE_str  = "N/A"

        if left_fit is not None and right_fit is not None:
            try:
                poly_px = (np.add(left_fit, right_fit) / 2)
                XTE, HE, camera_px, closest_px = self.compute_error(poly_px)

                ploty       = np.linspace(0, binary_BEV.shape[0]-1, binary_BEV.shape[0])
                left_fitx   = np.polyval(left_fit,  ploty)
                center_fitx = np.polyval(poly_px,   ploty)
                right_fitx  = np.polyval(right_fit, ploty)

                pts_left   = np.stack((left_fitx,   ploty), axis=1).astype(np.int32)
                pts_center = np.stack((center_fitx, ploty), axis=1).astype(np.int32)
                pts_right  = np.stack((right_fitx,  ploty), axis=1).astype(np.int32)

                cv2.polylines(binary_BEV, [pts_center], False, (0, 255, 255), 4)
                cv2.polylines(binary_BEV, [pts_left],   False, (255, 0, 0),   4)
                cv2.polylines(binary_BEV, [pts_right],  False, (0, 0, 255),   4)

                cv2.circle(binary_BEV,
                           (int(closest_px[0]), int(closest_px[1])), 8, (0,255,0), -1)
                cv2.line(binary_BEV,
                         (int(camera_px[0]), int(camera_px[1])),
                         (int(closest_px[0]), int(closest_px[1])), (0,255,0), 4)
                cv2.line(binary_BEV,
                         (int(camera_px[0]), int(camera_px[1])),
                         (int(camera_px[0]-20), int(camera_px[1]+20)), (255,0,255), 4)
                cv2.line(binary_BEV,
                         (int(camera_px[0]), int(camera_px[1])),
                         (int(camera_px[0]+20), int(camera_px[1]+20)), (255,0,255), 4)

                XTE_str = f"{XTE:.2f}"
                HE_str  = f"{np.degrees(HE):.2f}"
            except Exception as e:
                self.get_logger().debug(f"compute_error failed: {e}")

        w_str = f"{self._lane_width_px:.0f}px" if self._lane_width_px else "no_width_yet"
        print(f"EST XTE: {XTE_str} m  HE: {HE_str}°  lane_width: {w_str}")
        lane_error_msg = Float32MultiArray()
        lane_error_msg.data = [float(XTE), float(HE)]
        self._lane_error_pub.publish(lane_error_msg)
        if combine_fit_img is None:
            combine_fit_img = image

        self._render_pub.publish(
            self._cv_bridge.cv2_to_imgmsg(
                np.array(combine_fit_img, dtype=np.uint8), "bgr8")
        )
        self._bev_pub.publish(
            self._cv_bridge.cv2_to_imgmsg(
                np.array(binary_BEV, dtype=np.uint8), "bgr8")
        )

    def compute_error(self, poly_px):
        bev_height_m, bev_width_m = self._bev_cfg["bev_world_dim"]
        Sy, Sx = self._bev_cfg["unit_conversion_factor"]
        scale  = np.array([Sx, Sy])

        camera_m   = np.array([(bev_width_m / 2), bev_height_m])
        camera_px  = camera_m / scale
        closest_px = closest_point_on_polynomial(camera_px, poly_px)
        closest_m  = closest_px * scale

        XTE = np.linalg.norm(camera_m - closest_m)
        if camera_m[0] > closest_m[0]:
            XTE *= -1

        derivative  = np.polyder(poly_px)
        slope_px    = np.polyval(derivative, closest_px[1])
        slope_scale = scale[0] / scale[1]
        HE = np.arctan(slope_px * slope_scale)

        return float(XTE), float(HE), camera_px, closest_px

    def fit_poly_lanes(self, raw_img, binary_img):
        binary_warped, M, Minv = perspective_transform(
            binary_img, np.float32(self._bev_cfg["src"])
        )
        ret = lane_fit(binary_warped)

        if ret is None:
            return None, binary_warped, None, None

        left_fit_raw  = ret['left_fit']
        right_fit_raw = ret['right_fit']
        ploty         = ret['ploty']
        y_bottom      = float(ploty[-1])

        # ── Check 2: position consistency ─────────────────────────────────────
        if left_fit_raw is not None:
            if not self._position_is_consistent(left_fit_raw, self._left_line, y_bottom):
                self.get_logger().debug("Left fit jumped too far — discarding")
                left_fit_raw = None

        if right_fit_raw is not None:
            if not self._position_is_consistent(right_fit_raw, self._right_line, y_bottom):
                self.get_logger().debug("Right fit jumped too far — discarding")
                right_fit_raw = None

        # ── Both lanes reliable ───────────────────────────────────────────────
        if left_fit_raw is not None and right_fit_raw is not None:
            self._update_lane_width(left_fit_raw, right_fit_raw, y_bottom)

            left_fit  = self._left_line.add_fit(left_fit_raw)
            right_fit = self._right_line.add_fit(right_fit_raw)

            # FIX: update last_x_bottom with actual polyval result, not C coeff
            self._left_line.last_x_bottom  = float(np.polyval(left_fit,  y_bottom))
            self._right_line.last_x_bottom = float(np.polyval(right_fit, y_bottom))

            combine = final_viz(raw_img, left_fit, right_fit, Minv)
            return combine, binary_warped, left_fit, right_fit

        # ── One lane missing ──────────────────────────────────────────────────
        if self._lane_width_px is None:
            self.get_logger().debug("One lane missing, no width history yet")
            return None, binary_warped, None, None

        if left_fit_raw is None and right_fit_raw is not None:
            right_fit = self._right_line.add_fit(right_fit_raw)
            self._right_line.last_x_bottom = float(np.polyval(right_fit, y_bottom))
            left_fit  = right_fit.copy()
            left_fit[-1] -= self._lane_width_px
            self._left_line.last_x_bottom = float(np.polyval(left_fit, y_bottom))
            self.get_logger().debug(
                f"Left missing — extrapolating (width={self._lane_width_px:.0f}px)"
            )

        elif right_fit_raw is None and left_fit_raw is not None:
            left_fit  = self._left_line.add_fit(left_fit_raw)
            self._left_line.last_x_bottom = float(np.polyval(left_fit, y_bottom))
            right_fit = left_fit.copy()
            right_fit[-1] += self._lane_width_px
            self._right_line.last_x_bottom = float(np.polyval(right_fit, y_bottom))
            self.get_logger().debug(
                f"Right missing — extrapolating (width={self._lane_width_px:.0f}px)"
            )

        else:
            return None, binary_warped, None, None

        combine = final_viz(raw_img, left_fit, right_fit, Minv)
        return combine, binary_warped, left_fit, right_fit


def main(args=None):
    rclpy.init(args=args)
    node = LaneVisualizer()
    rclpy.spin(node)
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()