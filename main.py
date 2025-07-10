from model.comvistunt import get_landmark, get_height, get_haz
from calibration import aruco, green_mat
from config_manager import get_config, set_config
import argparse
import sys


def analyze(image_path: str, gender: str, age: int):
    # Deteksi landmark
    lms = get_landmark(image_path)
    if not lms:
        print("Gagal mendeteksi pose dari gambar.")
        return 1

    # Ambil nilai konversi dari konfigurasi
    cm_per_px = get_config("CM_PER_PX")
    if not cm_per_px:
        print("Kalibrasi belum dilakukan. Jalankan dengan --calibrate terlebih dahulu.")
        return 1

    # Hitung tinggi dan HAZ
    height_cm = get_height(lms, cm_per_px)
    haz_score, haz_status = get_haz(height_cm, gender, age)

    # Tampilkan hasil
    print(f"Height: {height_cm:.2f} cm")
    print(f"HAZ: {haz_score:.2f} ({haz_status})")

    return 0


def calibration(image_path: str, method: str):
    if method == "aruco":
        cm_per_px = aruco(image_path)
    elif method == "green_mat":
        cm_per_px = green_mat(image_path)
    else:
        print("Metode kalibrasi tidak dikenali.")
        return 1

    if cm_per_px is None:
        print("Kalibrasi gagal.")
        return 1

    set_config("CM_PER_PX", cm_per_px)
    print(f"Calibration complete. 1 pixel = {cm_per_px:.4f} cm")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to RGB image (.jpg/.png)")
    parser.add_argument("--gender", default="male", choices=["male", "female"], help="Gender of the child")
    parser.add_argument("--age", default=0, type=int, help="Age of the child in months")
    parser.add_argument("--method", default="aruco", choices=["aruco", "green_mat"], help="Method for calibration")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate the model")
    parser.add_argument("--analyze", action="store_true", help="Analyze the image")
    args = parser.parse_args()

    # Validasi gambar
    if not args.image:
        print("Masukkan path gambar dengan --image path/to/image.jpg")
        sys.exit(1)

    if args.calibrate:
        return calibration(args.image, args.method)

    if args.analyze:
        return analyze(args.image, args.gender, args.age)

    print("Tambahkan flag --analyze atau --calibrate untuk memulai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
