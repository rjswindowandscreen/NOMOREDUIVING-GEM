import os

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
from line_fit import final_viz, perspective_transform, closest_point_on_polynomial
from model_utils import load_model, inference
import rich
import cv2
from scipy.spatial.transform import Rotation as R


def weighted_lane_fit(binary_warped, distance_power=3.0, min_pixels=20):
    img_height, img_width = binary_warped.shape
    midpoint = img_width  // 2
    car_y    = img_height
    car_x    = img_width  // 2
    max_dist = np.sqrt(car_x**2 + car_y**2)

    def center_weights(ys, xs):
        dy = ys - car_y
        dx = xs - car_x
        dist       = np.sqrt(dx**2 + dy**2)
        radial_w   = 1.0 - np.clip(dist / max_dist, 0, 1)
        lateral_dist = np.abs(xs - car_x)
        center_w   = 1.0 - np.clip(lateral_dist / car_x, 0, 1)
        combined   = radial_w * center_w
        return combined ** distance_power

    def fit_poly_weighted(mask):
        ys, xs = np.where(mask > 0)
        if len(ys) < min_pixels:
            return None
        try:
            w = center_weights(ys, xs)
            return np.polyfit(ys, xs, 2, w=w)
        except Exception:
            return None

    # Build left and right masks from connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_warped, connectivity=8)

    left_mask  = np.zeros_like(binary_warped)
    right_mask = np.zeros_like(binary_warped)

    BOTTOM_STRIP = 0.6

    for i in range(1, num_labels):
        component_mask = (labels == i).astype(np.uint8)
        touches_bottom = np.any(component_mask[int(img_height * BOTTOM_STRIP):, :])
        if not touches_bottom:
            continue
        bottom_pixels = component_mask[int(img_height * BOTTOM_STRIP):, :]
        bottom_cols   = np.where(bottom_pixels)[1]
        if len(bottom_cols) == 0:
            continue
        if np.mean(bottom_cols) < midpoint:
            left_mask  = cv2.bitwise_or(left_mask,  component_mask)
        else:
            right_mask = cv2.bitwise_or(right_mask, component_mask)

    left_fit  = fit_poly_weighted(left_mask)
    right_fit = fit_poly_weighted(right_mask)

    if left_fit is None and right_fit is None:
        return None

    ploty   = np.linspace(0, img_height - 1, img_height)
    nonzero = binary_warped.nonzero()

    return {
        'left_fit':        left_fit,
        'right_fit':       right_fit,
        'left_detected':   left_fit  is not None,
        'right_detected':  right_fit is not None,
        'ploty':           ploty,
        'nonzerox':        np.array(nonzero[1]),
        'nonzeroy':        np.array(nonzero[0]),
        'left_lane_inds':  np.where(left_mask.ravel()  > 0)[0],
        'right_lane_inds': np.where(right_mask.ravel() > 0)[0],
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

        # Lane width — set once from first clean detection, never updated
        self._lane_width  = None
        self.COLLAPSE_THRESHOLD = 50

        # Dilation kernel for thickening sparse lane lines
        self._dilate_kernel = np.ones((5, 5), np.uint8)

    def _assign_single_line(self, fit):
        """
        Determine if a detected line is left or right based on curvature direction.
        Positive A coefficient = curves right = left lane
        Negative A coefficient = curves left = right lane
        """
        A = fit[0]
        if A >= 0:
            # Curves right — this is the left lane
            return 'left'
        else:
            # Curves left — this is the right lane
            return 'right'

    def _infer_missing_line(self, visible_fit, visible_is_right):
        inferred = visible_fit.copy()
        if visible_is_right:
            inferred[2] -= self._lane_width
        else:
            inferred[2] += self._lane_width
        return inferred

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
            poly_px = (np.add(ret["left_fit"], ret["right_fit"]) / 2)
            XTE, HE, camera_px, closest_px = self.compute_error(poly_px)

            ploty       = ret['ploty']
            left_fitx   = np.polyval(ret["left_fit"],  ploty)
            center_fitx = np.polyval(poly_px,           ploty)
            right_fitx  = np.polyval(ret["right_fit"], ploty)

            pts_left   = np.stack((left_fitx,    ploty), axis=1).astype(np.int32)
            pts_center = np.stack((center_fitx,  ploty), axis=1).astype(np.int32)
            pts_right  = np.stack((right_fitx,   ploty), axis=1).astype(np.int32)

            cv2.polylines(binary_BEV, [pts_center], isClosed=False, color=(0, 255, 255), thickness=4)
            cv2.polylines(binary_BEV, [pts_left],   isClosed=False, color=(255, 0, 0),   thickness=4)
            cv2.polylines(binary_BEV, [pts_right],  isClosed=False, color=(0, 0, 255),   thickness=4)

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
                "highbay_testbed", "stereo_camera_link", msg.header.stamp)
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

    def compute_error(self, poly_px):
        bev_height_m, bev_width_m = self._bev_cfg["bev_world_dim"]
        Sy, Sx = self._bev_cfg["unit_conversion_factor"]
        scale = np.array([Sx, Sy])

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

        return XTE, HE, camera_px, closest_px

    def fit_poly_lanes(self, raw_img, binary_img):
        binary_warped, M, Minv = perspective_transform(
            binary_img, np.float32(self._bev_cfg["src"]))

        # Dilate to thicken sparse lane lines before BEV transform
        binary_warped = cv2.dilate(binary_warped, self._dilate_kernel, iterations=2)

        img_height = binary_warped.shape[0]
        img_center = binary_warped.shape[1] / 2

        ret = weighted_lane_fit(binary_warped)

        if ret is None:
            self.get_logger().debug("ret is None; returning None.")
            return None, binary_warped, None

        left_detected  = ret['left_detected']
        right_detected = ret['right_detected']
        left_fit       = ret['left_fit']
        right_fit      = ret['right_fit']

        if left_detected and right_detected:
            left_x_base    = np.polyval(left_fit,  img_height)
            right_x_base   = np.polyval(right_fit, img_height)
            measured_width = abs(right_x_base - left_x_base)

            if measured_width > self.COLLAPSE_THRESHOLD:
                # Record lane width from first clean detection only
                if self._lane_width is None:
                    self._lane_width = measured_width
                    self.get_logger().info(f"Lane width set: {self._lane_width:.1f} px")

        elif left_detected and not right_detected:
            # Only one component detected on left side — verify with curvature
            side = self._assign_single_line(left_fit)
            if side == 'left':
                # Confirmed left — infer right
                if self._lane_width is not None:
                    right_fit        = self._infer_missing_line(left_fit, visible_is_right=False)
                    ret['right_fit'] = right_fit
                else:
                    #self.get_logger().warn("Only left detected, no lane width yet")
                    return None, binary_warped, None
            else:
                # Curvature says this is actually the right lane
                if self._lane_width is not None:
                    right_fit        = left_fit
                    left_fit         = self._infer_missing_line(right_fit, visible_is_right=True)
                    ret['left_fit']  = left_fit
                    ret['right_fit'] = right_fit
                else:
                    #self.get_logger().warn("Misassigned line, no lane width yet")
                    return None, binary_warped, None

        elif right_detected and not left_detected:
            # Only one component detected on right side — verify with curvature
            side = self._assign_single_line(right_fit)
            if side == 'right':
                # Confirmed right — infer left
                if self._lane_width is not None:
                    left_fit        = self._infer_missing_line(right_fit, visible_is_right=True)
                    ret['left_fit'] = left_fit
                else:
                    #self.get_logger().warn("Only right detected, no lane width yet")
                    return None, binary_warped, None
            else:
                # Curvature says this is actually the left lane
                if self._lane_width is not None:
                    left_fit         = right_fit
                    right_fit        = self._infer_missing_line(left_fit, visible_is_right=False)
                    ret['left_fit']  = left_fit
                    ret['right_fit'] = right_fit
                else:
                    #self.get_logger().warn("Misassigned line, no lane width yet")
                    return None, binary_warped, None

        else:
            return None, binary_warped, None

        ret['left_fit']  = left_fit
        ret['right_fit'] = right_fit

        combine_fit_img = final_viz(raw_img, left_fit, right_fit, Minv)
        return combine_fit_img, binary_warped, ret


def main(args=None):
    rclpy.init(args=args)
    node = LaneVisualizer()
    rclpy.spin(node)
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()