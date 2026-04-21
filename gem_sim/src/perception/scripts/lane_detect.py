import os

import torch
import json
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.parameter import Parameter

from worldgt import WorldGT
from line_fit import final_viz, perspective_transform
from model_utils import load_model, inference
import rich
import cv2
from scipy.spatial.transform import Rotation as R


def fit_left_lane(binary_warped, distance_power=1.0, min_pixels=50):
    img_height, img_width = binary_warped.shape
    car_y    = img_height
    car_x    = img_width // 2
    max_dist = np.sqrt(car_x**2 + car_y**2)

    def center_weights(ys, xs):
        dy           = ys - car_y
        dx           = xs - car_x
        dist         = np.sqrt(dx**2 + dy**2)
        radial_w     = 1.0 - np.clip(dist / max_dist, 0, 1)
        lateral_dist = np.abs(xs - car_x)
        center_w     = 1.0 - np.clip(lateral_dist / car_x, 0, 1)
        combined     = radial_w * 0.75 * center_w
        return combined ** distance_power

    def fit_poly(mask):
        ys, xs = np.where(mask > 0)
        if len(ys) < min_pixels:
            return None, None, None
        try:
            w  = center_weights(ys, xs)
            y  = ys.astype(np.float64)
            cy = float(car_y)

            X   = np.column_stack([y**4, y**3, y**2, y, np.ones_like(y)])
            W   = np.diag(w)
            XtW = X.T @ W

            SLOPE_PENALTY = 3000.0
            deriv_vec  = np.array([4*cy**3, 3*cy**2, 2*cy, 1.0, 0.0])
            slope_pen  = SLOPE_PENALTY * np.outer(deriv_vec, deriv_vec)

            B_PENALTY = 5000.0
            b_pen = np.zeros((5, 5))
            b_pen[1, 1] = B_PENALTY

            A_PENALTY = 50.0
            a_pen = np.zeros((5, 5))
            a_pen[0, 0] = A_PENALTY

            reg = np.eye(5) * 1e-4

            A_mat  = XtW @ X + slope_pen + b_pen + a_pen + reg
            b_vec  = XtW @ xs.astype(np.float64)
            coeffs = np.linalg.solve(A_mat, b_vec)

            return coeffs, float(ys.min()), float(ys.max())
        except Exception:
            return None, None, None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_warped, connectivity=8)

    BOTTOM_STRIP = 0.6
    best_mask    = None
    best_count   = 0

    for i in range(1, num_labels):
        component_mask = (labels == i).astype(np.uint8)
        pixel_count    = int(np.sum(component_mask))
        if pixel_count < 15:
            continue
        touches_bottom = np.any(component_mask[int(img_height * BOTTOM_STRIP):, :])
        if not touches_bottom:
            continue
        if pixel_count > best_count:
            best_count = pixel_count
            best_mask  = component_mask

    if best_mask is None:
        return None

    result = fit_poly(best_mask)
    if result[0] is None:
        return None

    coeffs, y_min, y_max = result

    ploty   = np.linspace(0, img_height - 1, img_height)
    nonzero = binary_warped.nonzero()

    return {
        'left_coeffs': coeffs,
        'left_fit':    lambda y: np.polyval(coeffs, y),
        'left_y_min':  y_min,
        'left_y_max':  y_max,
        'ploty':       ploty,
        'nonzerox':    np.array(nonzero[1]),
        'nonzeroy':    np.array(nonzero[0]),
    }


class LaneVisualizer(Node):
    def __init__(self):
        super().__init__("lane_visualizer")

        sim_time_param = Parameter('use_sim_time', Parameter.Type.BOOL, True)
        self.set_parameters([sim_time_param])

        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._model = load_model()
            if self._model is not None:
                self._model = self._model.to(self._dev)
                self._model = self._model.eval()
                rich.print("[green]loaded SimpleEnet :o")
            else:
                self.get_logger().error(f"could not load SimpleEnet model x_X: {e}")
                exit(1)
        except Exception as e:
            self.get_logger().error(f"could not load SimpleEnet model x_X: {e}")
            exit(1)

        try:
            with open(os.path.join("data", "bev_config.json")) as f:
                self._bev_cfg = json.load(f)
        except FileNotFoundError:
            self.get_logger().error(f"could not load bev config x_X: {e}")
            exit(1)

        self._world = WorldGT("Silverstone")
        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)

        self._image_msg = None
        self._cv_bridge  = CvBridge()

        self.create_subscription(
            Image,
            "/camera/image_raw",
            self._on_image,
            10
        )

        self._lane_error_pub = self.create_publisher(
            Float32MultiArray, '/lane_error', 10)

        self._lane_width    = None
        self.COLLAPSE_THRESHOLD = 50
        self._dilate_kernel = np.ones((3, 3), np.uint8)
        self._last_ret      = None

    def _on_image(self, msg) -> None:
        self._image_msg = msg
        if self._model is None:
            return

        image = self._cv_bridge.imgmsg_to_cv2(self._image_msg, "bgr8")
        mask  = inference(self._model, image, self._dev)
        m     = mask.astype(np.uint8) * 255

        cv2.imshow("raw_mask", m)

        combine_fit_img, binary_BEV, ret = self.fit_poly_lanes(image, m)

        binary_BEV = np.pad(binary_BEV, ((0, 100), (0, 0)))
        binary_BEV = cv2.cvtColor(binary_BEV, cv2.COLOR_GRAY2BGR)

        if ret:
            left_fit      = ret['left_fit']
            right_fit     = ret['right_fit']
            center_fit    = ret['center_fit']
            center_coeffs = ret['center_coeffs']

            XTE, HE, camera_px, closest_px = self.compute_error(center_coeffs)

            # Publish lane error for controller
            lane_error_msg      = Float32MultiArray()
            lane_error_msg.data = [float(XTE), float(HE)]
            self._lane_error_pub.publish(lane_error_msg)

            ploty       = ret['ploty']
            left_fitx   = np.array([float(left_fit(y))   for y in ploty])
            center_fitx = np.array([float(center_fit(y)) for y in ploty])
            right_fitx  = np.array([float(right_fit(y))  for y in ploty])

            pts_left   = np.stack((left_fitx,    ploty), axis=1).astype(np.int32)
            pts_center = np.stack((center_fitx,  ploty), axis=1).astype(np.int32)
            pts_right  = np.stack((right_fitx,   ploty), axis=1).astype(np.int32)

            cv2.polylines(binary_BEV, [pts_center], isClosed=False, color=(0, 255, 255), thickness=1)
            cv2.polylines(binary_BEV, [pts_left],   isClosed=False, color=(255, 0, 0),   thickness=1)
            cv2.polylines(binary_BEV, [pts_right],  isClosed=False, color=(0, 0, 255),   thickness=1)

            cv2.circle(binary_BEV,
                       (int(closest_px[0]), int(closest_px[1])), 8, (0, 255, 0), -1)
            cv2.line(binary_BEV,
                     (int(camera_px[0]),  int(camera_px[1])),
                     (int(closest_px[0]), int(closest_px[1])),
                     (0, 255, 0), 4)
            cv2.line(binary_BEV,
                     (int(camera_px[0]), int(camera_px[1])),
                     (int(camera_px[0] - 20), int(camera_px[1] + 20)),
                     (255, 0, 255), 4)
            cv2.line(binary_BEV,
                     (int(camera_px[0]), int(camera_px[1])),
                     (int(camera_px[0] + 20), int(camera_px[1] + 20)),
                     (255, 0, 255), 4)

            XTE = f"{XTE:.2f}"
            HE  = f"{np.degrees(HE):.2f}"
        else:
            XTE = "N/A"
            HE  = "N/A"

        try:
            trans = self._tf_buf.lookup_transform(
                "silverstone", "stereo_camera_link", msg.header.stamp)
            pos = trans.transform.translation
            q   = trans.transform.rotation
            rotation     = R.from_quat([q.x, q.y, q.z, q.w])
            euler_angles = rotation.as_euler('xyz', degrees=False)
            yaw  = euler_angles[2]
            lane, _, gt_XTE, gt_HE = self._world.get_metrics(pos.x, pos.y, yaw)
            gt_XTE = f"{gt_XTE:.2f}"
            gt_HE  = f"{np.degrees(gt_HE):.2f}"
        except Exception:
            lane   = "unknown"
            gt_XTE = "N/A"
            gt_HE  = "N/A"

        print(f"EST XTE: {XTE} m - HE: {HE}° -- GT XTE: {gt_XTE} m HE: {gt_HE}° - lane: {lane}")

        if combine_fit_img is None:
            combine_fit_img = image

        cv2.imshow("render_view", combine_fit_img)
        cv2.imshow("binary_BEV", binary_BEV)
        cv2.waitKey(1)

    def compute_error(self, center_coeffs):
        bev_height_m, bev_width_m = self._bev_cfg["bev_world_dim"]
        Sy, Sx = self._bev_cfg["unit_conversion_factor"]
        scale = np.array([Sx, Sy])

        camera_m  = np.array([(bev_width_m / 2), bev_height_m])
        camera_px = camera_m / scale

        car_y_px    = camera_px[1]
        center_x_px = float(np.polyval(center_coeffs, car_y_px))
        closest_px  = np.array([center_x_px, car_y_px])
        closest_m   = closest_px * scale

        dx  = closest_m[0] - camera_m[0]
        XTE = dx

        deriv    = np.polyder(center_coeffs)
        slope_px = float(np.polyval(deriv, car_y_px))
        slope_m  = slope_px * (Sx / Sy)

        f     = np.array([0.0, -1.0])
        t     = np.array([slope_m, -1.0])
        cross = f[0]*t[1] - f[1]*t[0]
        dot   = f[0]*t[0] + f[1]*t[1]
        HE    = np.arctan2(cross, dot)

        return XTE, HE, camera_px, closest_px

    def fit_poly_lanes(self, raw_img, binary_img):
        binary_warped, M, Minv = perspective_transform(
            binary_img, np.float32(self._bev_cfg["src"]))

        binary_warped = cv2.dilate(binary_warped, self._dilate_kernel, iterations=1)

        img_height = binary_warped.shape[0]
        img_width  = binary_warped.shape[1]

        ret = fit_left_lane(binary_warped)

        if ret is not None:
            left_coeffs = ret['left_coeffs']
            try:
                x_base = float(np.polyval(left_coeffs, img_height))
                if not (0 < x_base < img_width):
                    ret = None
            except Exception:
                ret = None

        if ret is None:
            if self._last_ret is not None:
                return final_viz(raw_img,
                                 self._last_ret['left_fit'],
                                 self._last_ret['right_fit'],
                                 Minv), binary_warped, self._last_ret
            return None, binary_warped, None

        left_coeffs = ret['left_coeffs']

        if self._lane_width is None:
            bev_height_m, bev_width_m = self._bev_cfg["bev_world_dim"]
            Sy, Sx = self._bev_cfg["unit_conversion_factor"]
            self._lane_width = 3.5 / Sx
            self.get_logger().info(f"Lane width initialized: {self._lane_width:.1f} px")

        right_coeffs         = left_coeffs.copy()
        right_coeffs[-1]    += self._lane_width
        center_coeffs        = (left_coeffs + right_coeffs) / 2.0

        ret['left_fit']      = lambda y: np.polyval(left_coeffs,   y)
        ret['right_fit']     = lambda y: np.polyval(right_coeffs,  y)
        ret['center_fit']    = lambda y: np.polyval(center_coeffs, y)
        ret['center_coeffs'] = center_coeffs
        ret['center_y_min']  = ret['left_y_min']
        ret['center_y_max']  = ret['left_y_max']

        self._last_ret = ret

        combine_fit_img = final_viz(raw_img, ret['left_fit'], ret['right_fit'], Minv)
        return combine_fit_img, binary_warped, ret


def main(args=None):
    rclpy.init(args=args)
    node = LaneVisualizer()
    rclpy.spin(node)
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()