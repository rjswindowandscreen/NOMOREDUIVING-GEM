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
    left_fitx  = left_fit(ploty)
    right_fitx = right_fit(ploty)

    out_img[nonzeroy[left_lane_inds],  nonzerox[left_lane_inds]]  = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx,  ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
    plt.gcf().clear()


def final_viz(undist, left_fit, right_fit, m_inv):
    ploty      = np.linspace(0, undist.shape[0]-1, undist.shape[0])
    left_fitx  = np.array([float(left_fit(y))  for y in ploty])
    right_fitx = np.array([float(right_fit(y)) for y in ploty])

    color_warp = np.zeros((720, 1280, 3), dtype='uint8')

    pts_left  = np.array([np.transpose(np.vstack([left_fitx,  ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
    undist  = np.array(undist,  dtype=np.uint8)
    newwarp = np.array(newwarp, dtype=np.uint8)
    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


def closest_point_on_spline(point, spline, y_min, y_max):
    x0, y0 = point

    y_lo = y_max - (y_max - y_min) * 0.1
    y_hi = y_max
    y    = (y_lo + y_hi) / 2.0

    for _ in range(10):
        x   = float(spline(y))
        eps = 0.1
        x_p = float(spline(np.clip(y + eps, y_lo, y_hi)))
        x_m = float(spline(np.clip(y - eps, y_lo, y_hi)))
        dx  = (x_p - x_m) / (2 * eps)
        d2x = (x_p - 2*x + x_m) / (eps**2)

        f  = (x - x0) * dx + (y - y0)
        fp = dx**2 + (x - x0) * d2x + 1.0

        if abs(fp) < 1e-10:
            break

        y_new = y - f / fp
        y     = float(np.clip(y_new, y_lo, y_hi))

        if abs(y_new - y) < 1e-6:
            break

    return np.array([float(spline(y)), y])


def perspective_transform(img, src):
    height, width = img.shape[:2]
    dst = np.float32([(0, 0), (0, height), (width, height), (width, 0)])
    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = np.linalg.inv(M)
    warped_img = cv2.warpPerspective(img, M, (width, height))
    return warped_img, M, Minv


def closest_point_on_spline(point, spline, y_min, y_max):
    """
    Finds closest point on a callable curve x = S(y) using
    Newton's method in the bottom 10% of the y range.
    Works for both splines and lambda callables.
    """
    x0, y0 = point

    # Search only bottom 10% — car is always near y_max
    y_lo = y_max - (y_max - y_min) * 0.1
    y_hi = y_max
    y    = (y_lo + y_hi) / 2.0

    for _ in range(10):
        x   = float(spline(y))
        eps = 0.1
        x_p = float(spline(np.clip(y + eps, y_lo, y_hi)))
        x_m = float(spline(np.clip(y - eps, y_lo, y_hi)))
        dx  = (x_p - x_m) / (2 * eps)
        d2x = (x_p - 2 * x + x_m) / (eps ** 2)

        f  = (x - x0) * dx + (y - y0)
        fp = dx**2 + (x - x0) * d2x + 1.0

        if abs(fp) < 1e-10:
            break

        y_new = y - f / fp
        y     = float(np.clip(y_new, y_lo, y_hi))

        if abs(y_new - y) < 1e-6:
            break

    return np.array([float(spline(y)), y])


def closest_point_on_polynomial(point, coeffs):
    """
    Finds the closest point on a polynomial curve x = P(y) to a target point.
    Analytical solution using root finding.
    """
    x0, y0 = point

    P       = np.poly1d(coeffs)
    P_deriv = P.deriv()
    y_poly  = np.poly1d([1, 0])

    dist_deriv_poly = (P - x0) * P_deriv + y_poly - y0

    roots      = dist_deriv_poly.roots
    real_roots = roots[np.isreal(roots)].real

    min_dist_sq = float('inf')
    closest_x   = None
    closest_y   = None

    for y in real_roots:
        x       = P(y)
        dist_sq = (x - x0)**2 + (y - y0)**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_x   = x
            closest_y   = y

    return np.array([closest_x, closest_y])