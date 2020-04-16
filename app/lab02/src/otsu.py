import cv2
import numpy as np
from random import randint

img = cv2.imread('../data/moon.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('noname', img)
#cv2.waitKey()

th=127
_, binary_img=cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
#
cv2.imshow("binary image", binary_img)
cv2.waitKey(0)

th, otsu=cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
print("Otsu's threshold={}".format(th))
cv2.imshow("Otsu's binary image", otsu)
cv2.waitKey(0)

#otsu=~otsu

#nlabels, labels = cv2.connectedComponents(otsu)
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(otsu)
print(stats)
#
colors=np.zeros((nlabels+1,3), dtype=np.uint8)
for i in range(1, nlabels+1):
    colors[i]=(randint(0, 255), randint(0, 255), randint(0, 255))
   
rows, cols = otsu.shape

label_image=np.zeros((rows, cols, 3), dtype=np.uint8)

for i in range(rows):
    for j in range(cols):
        label=labels[i][j]
        label_image[i][j]=colors[label]
        
cv2.imshow("Connected components", label_image)
cv2.waitKey(0)

cv2.destroyAllWindows()