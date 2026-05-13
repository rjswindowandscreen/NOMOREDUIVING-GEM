import torch
import cv2
import numpy as np

from dataset import CaptureDataset


def mask_by_hsv(image, target_hsv, tolerance):
    if isinstance(tolerance, int):
        tol_h, tol_s, tol_v = tolerance, tolerance, tolerance
    else:
        tol_h, tol_s, tol_v = tolerance
    h, s, v = target_hsv
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    s_min = max(s - tol_s, 0)
    s_max = min(s + tol_s, 255)
    v_min = max(v - tol_v, 0)
    v_max = min(v + tol_v, 255)

    if h - tol_h < 0:
        lower1 = np.array([0, s_min, v_min])
        upper1 = np.array([h + tol_h, s_max, v_max])

        lower2 = np.array([179 + (h - tol_h), s_min, v_min])
        upper2 = np.array([179, s_max, v_max])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif h + tol_h > 179:
        lower1 = np.array([h - tol_h, s_min, v_min])
        upper1 = np.array([179, s_max, v_max])

        lower2 = np.array([0, s_min, v_min])
        upper2 = np.array[(h + tol_h) - 179, s_max, v_max]

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower = np.array([h - tol_h, s_min, v_min])
        upper = np.array([h + tol_h, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
    return mask


if __name__ == "__main__":
    ds = CaptureDataset("data/capture")
    yellow_lane = [30, 255, 255]

    for i in range(len(ds)):
        image, _ = ds.read(i)
        thresh = mask_by_hsv(image, yellow_lane, [10, 100, 150])





        h, w = thresh.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        floodfill = thresh.copy()
        for x in range(w):
            if floodfill[0, x] == 255:
                cv2.floodFill(floodfill, mask, (x, 0), 128)
        top_region_mask = (floodfill == 128)
        thresh[top_region_mask] = 0

        orange_hsv = [15, 200, 200]   # orange
        white_hsv  = [0, 0, 255]      # white (low saturation, high value)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # ORANGE
        orange_lower = np.array([0, 20, 100])
        orange_upper = np.array([40, 255, 255])

        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)

        # WHITE
        white_lower = np.array([0, 0, 140])
        white_upper = np.array([179, 100, 255])

        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        # SHIFT ORANGE DOWNWARD
        shift_pixels = 30

        orange_shifted_down = np.zeros_like(orange_mask)
        orange_shifted_down[shift_pixels:, :] = orange_mask[:-shift_pixels, :]

        # EXPAND THE REGION BELOW ORANGE
        kernel = np.ones((60, 40), np.uint8)

        orange_below_region = cv2.dilate(
            orange_shifted_down,
            kernel,
            iterations=1
        )

        # KEEP ONLY WHITE UNDER ORANGE
        white_below_orange = cv2.bitwise_and(
            white_mask,
            orange_below_region
        )

        # COMBINE
        obstacle_thresh = cv2.bitwise_or(
            orange_mask,
            white_below_orange
        )

        kernel = np.ones((5, 5), np.uint8)
        lane_safe_mask = cv2.dilate(thresh, kernel, iterations=2)
        obstacle_thresh[lane_safe_mask > 0] = 0

        # cv2.imshow("image", image)
        # cv2.imshow("lane", thresh)
        # cv2.imshow("obstacle", obstacle_thresh)

        # cv2.waitKey(0)  # or 0 if you want to step manually

        red_lower1 = np.array([0, 60, 60])
        red_upper1 = np.array([15, 255, 255])

        red_lower2 = np.array([175, 60, 60])
        red_upper2 = np.array([179, 255, 255])

        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)

        sign_thresh = cv2.bitwise_or(red_mask1, red_mask2)

        
        final_mask = np.zeros((h, w), dtype=np.uint8) 
        # Lane = 1 
        final_mask[thresh > 0] = 255
        # Obstacle = 2 
        final_mask[obstacle_thresh > 0] = 100

        vis = np.zeros((h, w, 3), dtype=np.uint8)

        # Lane = yellow
        vis[thresh >0] = [255, 255, 255] 



        vis[sign_thresh >0] = [0, 0, 255]
        
        # Obstacle = red
        vis[obstacle_thresh >0] = [0, 255, 255] 


        ds.write_mask(vis, i)