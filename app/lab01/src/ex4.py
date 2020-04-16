import numpy as np
import cv2

img = np.zeros((480, 640, 3), dtype=np.int8)
img_rows, img_cols, img_channels = img.shape

x1 = int(img_cols/2 - 100)
y1 = int(img_rows/2 - 100)
x2 = int(img_cols/2 + 100)
y2 = int(img_rows/2 + 100)

cx=int(img_cols/2)
cy=int(img_rows/2)

radius = int(cx - x1)

cv2.rectangle(img, (x1, y1), (x2,y2), (0, 255, 255), 4)
cv2.circle(img, (cx, cy), radius, (100, 100, 200), 4)
cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)

cv2.imshow("drawing", img)

cv2.waitKey(0)

cv2.destroyAllWindows()