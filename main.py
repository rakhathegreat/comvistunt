# run_anthropometry.py
import cv2
import numpy as np
from comvistunt_II import process  # pastikan nama file core kamu ini
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to RGB image (.jpg/.png)")
parser.add_argument("--depth", help="Optional path to .npz depth file")
parser.add_argument("--age", type=int, required=True, help="Age in months (1–60)")
parser.add_argument("--sex", choices=["male", "female"], required=True, help="Sex")
parser.add_argument("--scale", type=float, help="Manual scale (cm per px), if needed")

args = parser.parse_args()

# Baca gambar RGB
img = cv2.imread(args.image)
if img is None:
    raise FileNotFoundError(f"Image file '{args.image}' not found or unreadable.")

# Jalankan proses
depth_file = args.depth if args.depth else None
result = process(img, depth_file, args.age, args.sex, manual_cm_per_px=args.scale)

# Tampilkan hasil
print(f"Height: {result.height_cm:.2f} cm")
print(f"Weight: {result.weight_kg:.2f} kg" if not np.isnan(result.weight_kg) else "Weight: not estimated")
print(f"HAZ: {result.haz:.2f} → {'Stunted' if result.stunted else 'Normal'}")
print("Debug info:", result.debug)
