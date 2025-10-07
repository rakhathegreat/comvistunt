import cv2
import mediapipe as mp
import numpy as np
from model.comvistunt import get_landmarks, draw_landmarks

img = "image/baby5.jpg"
lms = get_landmarks(img)
draw_landmarks(img, lms)
