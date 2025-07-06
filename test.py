import cv2
import mediapipe as mp

# Inisialisasi pose detector
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

# Ganti path ini dengan gambar bayi kamu
image_path = 'baby3.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Proses gambar
results = pose.process(image_rgb)

# Gambar hasil pose
if results.pose_landmarks:
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
else:
    print("❌ Tidak ada pose terdeteksi.")

# Tampilkan gambar
cv2.imshow('Pose Estimation', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
