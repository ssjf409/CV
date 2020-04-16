import numpy as np
import cv2

img = np.zeros((480, 640, 3), np.uint8)

img_rows, img_cols, img_channel = img.shape

print(img_rows, img_cols, img_channel)

crows, ccols = img_rows/2, img_cols/2

cv2.namedWindow("Hello, OpenCV", cv2.WINDOW_AUTOSIZE)

cv2.putText(img, "Hello, OpenCV", (0, int(crows)), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 4)

cv2.imshow("Hello, OpenCV", img)

cv2.waitKey(0)

cv2.destroyAllWindows()