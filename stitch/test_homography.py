import numpy as np
import cv2

img1 = cv2.imread("images/0/a.jpg", cv2.IMREAD_COLOR)
h, w, _ = img1.shape

H = np.array([[1.0, 0, 20], [0, 1.0, 0], [0, 0, 1.0]])

distance = -50
tem = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, distance], [0.0, 0.0, 1.0]])

H = np.dot(tem, H)

w1 = cv2.warpPerspective(img1, H, (w, h))
cv2.imwrite("images/0/a_.jpg", w1)