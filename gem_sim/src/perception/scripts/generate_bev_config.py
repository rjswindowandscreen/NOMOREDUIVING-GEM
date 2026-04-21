import os
import json
import numpy as np


def main():
    # ── Camera intrinsics from /camera/camera_info ────────────────────────────
    K = np.array([
        [476.7014503479004, 0.0,               400.0],
        [0.0,               476.7014980316162, 300.0],
        [0.0,               0.0,               1.0  ]
    ])

    # ── Camera extrinsics from tf2_echo base_footprint stereo_camera_link ─────
    R = np.array([
        [0, -1,  0],
        [0,  0, -1],
        [1,  0,  0]
    ])
    t = np.array([0.394, 0.000, 1.630])

    # ── BEV image size ────────────────────────────────────────────────────────
    bev_img_height, bev_img_width = 600, 800

    # ── BEV rectangle (meters in robot frame: X=fwd, Y=left, Z=up) ───────────
    # near=4m is the closest we can get while keeping all 4 points in frame
    # Reduce bev_half_w if you want to zoom in on the lane
    bev_near   = 4.0   # meters ahead — near edge
    bev_far    = 15.0  # meters ahead — far edge
    bev_half_w = 3.0   # meters each side (6m total width)

    bev_height_m = bev_far - bev_near   # 11m
    bev_width_m  = bev_half_w * 2       # 6m

    unit_conversion_factor = (
        bev_height_m / bev_img_height,
        bev_width_m  / bev_img_width
    )

    # ── BEV corners in robot frame ────────────────────────────────────────────
    # NOTE: +Y = robot LEFT = image LEFT
    # Order must match dst in perspective_transform():
    #   dst = [(0,0), (0,H), (W,H), (W,0)]
    #       = [top-left, bottom-left, bottom-right, top-right]
    #       = [far-left, near-left,   near-right,   far-right ]
    bev_world_coords = np.float32([
        [bev_far,   bev_half_w, 0],   # far-left   → top-left     (robot LEFT  = image LEFT)
        [bev_near,  bev_half_w, 0],   # near-left  → bottom-left
        [bev_near, -bev_half_w, 0],   # near-right → bottom-right (robot RIGHT = image RIGHT)
        [bev_far,  -bev_half_w, 0],   # far-right  → top-right
    ])

    # ── Project to pixel coordinates ──────────────────────────────────────────
    src    = []
    labels = ['far_left', 'near_left', 'near_right', 'far_right']
    for i, pt in enumerate(bev_world_coords):
        cam_pt = R @ (pt - t)
        if cam_pt[2] <= 0:
            raise ValueError(
                f"{labels[i]} projects behind camera (Z={cam_pt[2]:.3f}). "
                "Increase bev_near."
            )
        bev_pt = K @ cam_pt
        u = float(bev_pt[0] / bev_pt[2])
        v = float(bev_pt[1] / bev_pt[2])
        in_frame = "✓" if 0 <= u <= bev_img_width and 0 <= v <= bev_img_height else "✗ OUT OF FRAME"
        side = "LEFT" if u < 400 else "RIGHT"
        print(f"  {labels[i]}: pixel ({u:.0f}, {v:.0f}) — {side} side {in_frame}")
        src.append((u, v))

    src = np.float32(src)

    output = {
        "bev_world_dim":          [bev_height_m, bev_width_m],
        "unit_conversion_factor": list(unit_conversion_factor),
        "src":                    src.tolist(),
    }

    save_fn = os.path.join("data", "bev_config.json")
    os.makedirs("data", exist_ok=True)

    if os.path.isfile(save_fn):
        if input("File already exists. Overwrite? (y/n): ").lower() != 'y':
            import sys; sys.exit()

    with open(save_fn, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {save_fn}")
    print(f"BEV covers {bev_near}m–{bev_far}m ahead, {bev_width_m}m wide")


if __name__ == "__main__":
    main()