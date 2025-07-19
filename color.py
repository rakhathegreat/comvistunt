import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_val = hsv[y, x]
        print(f"HSV at ({x}, {y}): {hsv_val}")

# Buka gambar
image = cv2.imread('image/test2.png')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow("Click to get HSV", image)
cv2.setMouseCallback("Click to get HSV", mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()
