#import numpy as np
import cv2

img = cv2.imread("../data/lena.png", cv2.IMREAD_COLOR)

gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_rows, img_cols, img_channels = img.shape

cv2.imshow("lena", img)
cv2.imshow("gray_lena", gray_img)

cv2.waitKey(0)

b = img.copy()
b[:,:,1]=0
b[:,:,2]=0
g = img.copy()
g[:,:,0]=0
g[:,:,2]=0
r = img.copy()
r[:,:,0]=0
r[:,:,1]=0

cv2.imshow("red", r)
cv2.imshow("green", g)
cv2.imshow("blue", b)

cv2.waitKey(0)

cv2.destroyAllWindows()