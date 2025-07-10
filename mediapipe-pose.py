import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

img = cv2.imread("baby4.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = pose.process(img_rgb)

if results.pose_landmarks:
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

cv2.imshow("Pose Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
