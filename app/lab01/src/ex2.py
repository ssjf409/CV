import cv2

img = cv2.imread("../data/lena.png", cv2.IMREAD_COLOR)

img_rows, img_cols, img_channels = img.shape

resized_img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
resized_img2 = cv2.resize(img, (int(img_rows*0.5), int(img_cols*0.5)), interpolation=cv2.INTER_CUBIC)

cv2.imshow("lena", img)
cv2.imshow("resized lena", resized_img)
cv2.imshow("resized lena 2", resized_img2)

cv2.imwrite("../data/lena_resized.png", resized_img)

cv2.waitKey(1000)

cv2.destroyAllWindows()