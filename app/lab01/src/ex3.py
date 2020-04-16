import cv2

img = cv2.imread("../data/lena.png", cv2.IMREAD_COLOR)

img_rows, img_cols, img_channels = img.shape

ccols=img_cols/2
crows=img_rows/2

cv2.putText(img, 'lena', (int(ccols), int(crows)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)

cv2.imshow("lena", img)

cv2.waitKey(0)

cv2.destroyAllWindows()