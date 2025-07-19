import cv2
import mediapipe as mp
import numpy as np
from model.comvistunt import get_landmark, get_haz

def landmark_to_px(landmark, shape):
    return (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
def pixel_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

img = cv2.imread("image/test/test7.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

res = pose.process(img_rgb)

if res.pose_landmarks:
    lm = res.pose_landmarks.landmark
    shape = img.shape

    # Landmark penting
    nose = landmark_to_px(lm[mp_pose.PoseLandmark.NOSE], shape)
    left_eye = landmark_to_px(lm[mp_pose.PoseLandmark.LEFT_EYE], shape)
    right_eye = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_EYE], shape)
    left_shoulder = landmark_to_px(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], shape)
    right_shoulder = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], shape)
    left_hip = landmark_to_px(lm[mp_pose.PoseLandmark.LEFT_HIP], shape)
    right_hip = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_HIP], shape)
    knee = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_KNEE], shape)
    ankle = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_ANKLE], shape)
    heel = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_HEEL], shape)

    shoulder_px = tuple(np.mean([left_shoulder, right_shoulder], axis=0).astype(int))
    hip_px = tuple(np.mean([left_hip, right_hip], axis=0).astype(int))
    eyes_px = tuple(np.mean([left_eye, right_eye], axis=0).astype(int))

    vec_direction = np.array(eyes_px) - np.array(nose)
    vec_norm = np.linalg.norm(vec_direction)
    head_peak = tuple((np.array(nose) + (vec_direction / vec_norm) * abs(shoulder_px[1] - hip_px[1]) / 1.5).astype(int))

    d1 = pixel_distance(nose, head_peak)
    d2 = pixel_distance(nose, hip_px)
    d3 = pixel_distance(right_hip, knee)
    d4 = pixel_distance(knee, ankle)
    d5 = pixel_distance(ankle, heel)

    result = (d1 + d2 + d3 + d4 + d5) * 0.38461538461538464
    print(result)
    print(f"Berat Badan: {22 * (result / 100) ** 2:.2f} kg")

    cv2.line(img, nose, head_peak, (0, 0, 255), 2)
    cv2.line(img, nose, hip_px, (0, 0, 255), 2)
    cv2.line(img, right_hip, knee, (0, 0, 255), 2)
    cv2.line(img, knee, ankle, (0, 0, 255), 2)
    cv2.line(img, ankle, heel, (0, 0, 255), 2)

    age = 5
    gender = "female"

    print(get_haz(result, gender, age))

    cv2.putText(img, f"Height: {result:.2f}", nose, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(img, f"Weight: {22 * (result / 100) ** 2:.2f}", head_peak, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(img, f"HAZ: {get_haz(result, gender, age)}", hip_px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    

while True:
    cv2.imshow("Pose Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()