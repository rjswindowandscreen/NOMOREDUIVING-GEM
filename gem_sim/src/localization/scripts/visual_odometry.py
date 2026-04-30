#!/usr/bin/env python3
"""
visual_odometry.py — Monocular visual odometry using ORB feature matching.

Estimates incremental (dx, dy, dyaw) in the camera/body frame between
consecutive frames. Scale is recovered using the IMU-integrated speed
passed in from LocalKF.

Inputs  : BGR image frame, current speed estimate (m/s), dt (s)
Outputs : (dx, dy, dyaw) body-frame displacement, or None if tracking fails
"""

import math
import numpy as np
import cv2


class VisualOdometry:
    """
    Sparse ORB + BFMatcher visual odometry for the GEM front camera.

    The camera is assumed to be forward-facing and roughly level.
    Only the x (forward) and yaw components are reliable for a monocular
    ground vehicle — vy is small and noisy, but included for completeness.
    """

    def __init__(self, n_features: int = 500):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self._prev_gray = None
        self._prev_kp   = None
        self._prev_des  = None

        # Ratio test threshold (Lowe's ratio test)
        self._ratio_thresh = 0.70

        # Minimum good matches required to trust the estimate
        self._min_matches = 20

        # Focal length placeholder — tune to your GEM camera calibration.
        # A rough default: fx ≈ image_width * (1 / tan(hfov/2))
        # For a 640×480 camera with ~90° HFOV: fx ≈ 320
        self._fx = 320.0
        self._fy = 320.0
        self._cx = 320.0   # principal point x (≈ image_width  / 2)
        self._cy = 240.0   # principal point y (≈ image_height / 2)

    # -----------------------------------------------------------------------
    def update(self, frame_bgr: np.ndarray, speed: float, dt: float):
        """
        Process a new frame and return incremental body-frame displacement.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Current BGR image from the GEM front camera.
        speed : float
            Current forward speed estimate from LocalKF (m/s). Used as
            scale reference for monocular VO.
        dt : float
            Time since last frame (seconds).

        Returns
        -------
        (dx, dy, dyaw) : tuple[float, float, float] or None
            Body-frame displacement in metres and heading change in radians.
            Returns None if tracking failed (too few matches, bad geometry).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self._prev_gray is None or des is None or self._prev_des is None:
            self._prev_gray = gray
            self._prev_kp   = kp
            self._prev_des  = des
            return None

        # --- match features --------------------------------------------------
        raw_matches = self.matcher.knnMatch(self._prev_des, des, k=2)
        good = [m for m, n in raw_matches if m.distance < self._ratio_thresh * n.distance]

        if len(good) < self._min_matches:
            # Not enough matches — keep previous frame, skip this estimate
            self._prev_gray = gray
            self._prev_kp   = kp
            self._prev_des  = des
            return None

        # --- extract matched point coordinates -------------------------------
        pts_prev = np.float32([self._prev_kp[m.queryIdx].pt for m in good])
        pts_curr = np.float32([kp[m.trainIdx].pt           for m in good])

        # --- estimate Essential Matrix and decompose rotation ----------------
        K = np.array([[self._fx,       0, self._cx],
                      [      0, self._fy, self._cy],
                      [      0,        0,       1]], dtype=np.float64)

        E, mask = cv2.findEssentialMat(pts_curr, pts_prev, K,
                                       method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)

        if E is None:
            self._prev_gray = gray
            self._prev_kp   = kp
            self._prev_des  = des
            return None

        _, R, t, inlier_mask = cv2.recoverPose(E, pts_curr, pts_prev, K, mask=mask)

        n_inliers = int(inlier_mask.sum())
        if n_inliers < self._min_matches:
            self._prev_gray = gray
            self._prev_kp   = kp
            self._prev_des  = des
            return None

        # --- scale recovery using IMU speed ----------------------------------
        # Monocular VO gives a unit translation vector.
        # Multiply by speed*dt to recover metric scale.
        scale = speed * dt
        tx = float(t[2, 0]) * scale    # forward  (camera z → body x)
        ty = float(-t[0, 0]) * scale   # lateral  (camera x → body y, flipped)

        # Yaw change from rotation matrix: R[1,0] ≈ sin(dyaw) for small angles
        dyaw = math.atan2(float(R[1, 0]), float(R[0, 0]))

        # --- update state ----------------------------------------------------
        self._prev_gray = gray
        self._prev_kp   = kp
        self._prev_des  = des

        return tx, ty, dyaw

    # -----------------------------------------------------------------------
    def set_camera_intrinsics(self, fx: float, fy: float,
                               cx: float, cy: float):
        """
        Override default camera intrinsics with calibrated values.
        Call this during node initialisation after loading calibration.
        """
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy

    def reset(self):
        """Clear stored keyframe (e.g., after a long pause or GPS jump)."""
        self._prev_gray = None
        self._prev_kp   = None
        self._prev_des  = None