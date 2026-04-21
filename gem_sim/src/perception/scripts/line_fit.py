import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


def viz1(binary_warped, ret, save_file=None):
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	out_img = ret['out_img']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	if save_file is None:
		plt.show()
	else:
		plt.savefig(save_file)
	plt.gcf().clear()


def bird_fit(binary_warped, ret, Minv):
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = np.polyval(left_fit, ploty)
	right_fitx = np.polyval(right_fit, ploty)
	margin = 100
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 0, 255))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	result = cv2.warpPerspective(result, Minv, (1280, 720))
	return result


def final_viz(undist, left_fit, right_fit, m_inv):
	ploty      = np.linspace(0, undist.shape[0]-1, undist.shape[0])
	left_fitx  = np.polyval(left_fit,  ploty)
	right_fitx = np.polyval(right_fit, ploty)
	color_warp = np.zeros((720, 1280, 3), dtype='uint8')
	pts_left   = np.array([np.transpose(np.vstack([left_fitx,  ploty]))])
	pts_right  = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
	undist  = np.array(undist,  dtype=np.uint8)
	newwarp = np.array(newwarp, dtype=np.uint8)
	return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


class Line():
	def __init__(self, n):
		self.n              = n
		self.detected       = False
		self.last_x_bottom  = None  # actual x at bottom of BEV, set externally
		self.A = []
		self.B = []
		self.C = []
		self.A_avg = 0.
		self.B_avg = 0.
		self.C_avg = 0.

	def get_fit(self):
		return np.array([self.A_avg, self.B_avg, self.C_avg])

	def add_fit(self, fit_coeffs):
		q_full = len(self.A) >= self.n
		self.A.append(fit_coeffs[0])
		self.B.append(fit_coeffs[1])
		self.C.append(fit_coeffs[2])
		if q_full:
			self.A.pop(0)
			self.B.pop(0)
			self.C.pop(0)
		self.A_avg    = np.mean(self.A)
		self.B_avg    = np.mean(self.B)
		self.C_avg    = np.mean(self.C)
		self.detected = True
		return np.array([self.A_avg, self.B_avg, self.C_avg])

	def reset(self):
		self.detected      = False
		self.last_x_bottom = None
		self.A = []
		self.B = []
		self.C = []
		self.A_avg = 0.
		self.B_avg = 0.
		self.C_avg = 0.


def lane_fit(binary_warped, nwindows=20, margin=50, minpix=10):
    """
    Sliding window lane detection.
    Returns dict with left_fit/right_fit (either may be None if not enough pixels).
    Returns None only if there are zero white pixels at all.
    """
    MIN_PIXELS = 500

    total_pixels = np.sum(binary_warped > 0)
    if total_pixels == 0:
        return None

    histogram   = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint    = int(histogram.shape[0] / 2)
    leftx_base  = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height   = int(binary_warped.shape[0] / nwindows)
    nonzero         = binary_warped.nonzero()
    nonzeroy        = np.array(nonzero[0])
    nonzerox        = np.array(nonzero[1])
    left_lane_inds  = []
    right_lane_inds = []
    left_x_current  = leftx_base
    right_x_current = rightx_base
    left_momentum   = 0
    right_momentum  = 0

    for window in range(nwindows):
        win_y_low  = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        separation = right_x_current - left_x_current
        if separation < (margin * 2):
            mid             = (left_x_current + right_x_current) / 2
            left_x_current  = int(mid - margin)
            right_x_current = int(mid + margin)

        good_left_inds = (
            (nonzerox >= left_x_current  - margin) &
            (nonzerox <= left_x_current  + margin) &
            (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzerox >= right_x_current - margin) &
            (nonzerox <= right_x_current + margin) &
            (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
        ).nonzero()[0]

        if len(good_left_inds) > minpix:
            new_x          = int(np.mean(nonzerox[good_left_inds]))
            left_momentum  = int(0.6*(new_x-left_x_current) + 0.4*left_momentum)
            left_x_current = new_x
            left_lane_inds.append(good_left_inds)
        else:
            left_x_current += left_momentum

        if len(good_right_inds) > minpix:
            new_x           = int(np.mean(nonzerox[good_right_inds]))
            right_momentum  = int(0.6*(new_x-right_x_current) + 0.4*right_momentum)
            right_x_current = new_x
            right_lane_inds.append(good_right_inds)
        else:
            right_x_current += right_momentum

    try:
        left_lane_inds  = np.concatenate(left_lane_inds)
    except ValueError:
        left_lane_inds  = np.array([], dtype=int)
    try:
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        right_lane_inds = np.array([], dtype=int)

    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Debug — remove once tuned
    print(f"[lane_fit] left_pixels={len(leftx)} right_pixels={len(righty)}")

    def fit_poly_safe(y, x):
        if len(y) < MIN_PIXELS:
            return None
        try:
            return np.polyfit(y, x, 2)
        except Exception:
            return None

    left_fit  = fit_poly_safe(lefty,  leftx)
    right_fit = fit_poly_safe(righty, rightx)

    if left_fit is None and right_fit is None:
        return None

    ploty      = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx  = np.polyval(left_fit,  ploty) if left_fit  is not None else None
    right_fitx = np.polyval(right_fit, ploty) if right_fit is not None else None

    return {
        'left_fit':        left_fit,
        'right_fit':       right_fit,
        'left_fitx':       left_fitx,
        'right_fitx':      right_fitx,
        'ploty':           ploty,
        'nonzerox':        nonzerox,
        'nonzeroy':        nonzeroy,
        'left_lane_inds':  left_lane_inds,
        'right_lane_inds': right_lane_inds,
        'left_pixel_count':  len(leftx),
        'right_pixel_count': len(righty),
    }


def perspective_transform(img, src):
	height, width = img.shape[:2]
	dst        = np.float32([(0,0), (0,height), (width,height), (width,0)])
	M          = cv2.getPerspectiveTransform(src, dst)
	Minv       = np.linalg.inv(M)
	warped_img = cv2.warpPerspective(img, M, (width, height))
	return warped_img, M, Minv


def closest_point_on_polynomial(point, coeffs):
	x0, y0       = point
	P            = np.poly1d(coeffs)
	P_deriv      = P.deriv()
	y_poly       = np.poly1d([1, 0])
	dist_deriv   = (P - x0) * P_deriv + (y_poly - y0)
	roots        = dist_deriv.roots
	real_roots   = roots[np.isreal(roots)].real
	if len(real_roots) == 0:
		# fallback — use midpoint of image height
		y_mid = y0
		return np.array([float(P(y_mid)), float(y_mid)])
	min_dist_sq  = float('inf')
	closest_x    = None
	closest_y    = None
	for y in real_roots:
		x       = P(y)
		dist_sq = (x - x0)**2 + (y - y0)**2
		if dist_sq < min_dist_sq:
			min_dist_sq = dist_sq
			closest_x   = x
			closest_y   = y
	return np.array([float(closest_x), float(closest_y)])