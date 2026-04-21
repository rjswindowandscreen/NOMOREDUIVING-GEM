import os
import subprocess

DATA_DIR   = "data/capture"
masks_dir  = os.path.join(DATA_DIR, "masks")
images_dir = os.path.join(DATA_DIR, "images")

mask_files = sorted([
    f for f in os.listdir(masks_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
], key=lambda x: int(os.path.splitext(x)[0]))

print(f"Found {len(mask_files)} masks")
print("Close the editor window to move to the next image. Ctrl+C to stop.\n")

for i, fname in enumerate(mask_files):
    mask_path  = os.path.join(masks_dir,  fname)
    image_path = os.path.join(images_dir, fname)

    print(f"[{i+1}/{len(mask_files)}] Editing {fname}")
    print(f"  Reference image: {image_path}")

    # Open mask in editor — blocks until you close the window
    subprocess.run(["pinta", mask_path])  # or kolourpaint, gimp

print("Done!")