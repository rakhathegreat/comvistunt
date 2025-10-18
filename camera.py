import cv2
import os

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def capture():
    # Membuka kamera (0 = kamera default)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Gagal mengakses kamera!")
        return

    # Membaca satu frame dari kamera
    ret, frame = cap.read()

    if ret:
        os.makedirs("capture", exist_ok=True)


        # Menyimpan gambar ke file
        cv2.imwrite("capture/captured.png", frame)
        print(f"Gambar berhasil disimpan!")
    else:
        print("Gagal mengambil gambar!")

    # Melepaskan kamera
    cap.release()