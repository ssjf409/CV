import cv2
import numpy as np

image = cv2.imread('../data/fruits.png')
shifted = cv2.pyrMeanShiftFiltering(image, 30, 30)
shifted_list=shifted.tolist()

height, width, channels = image.shape

centers=[]

for m in range(0, height):
    #print(m, height)
    for n in range(0, width):
        if len(centers)==0:
            centers.append(shifted_list[m][n])
            continue
        if shifted_list[m][n] in centers:
            continue
            
        centers.append(shifted_list[m][n])
            
print(len(centers))
random_color=np.random.randint(0, 256, [len(centers), 3], np.uint8)

res_img=np.zeros(image.shape, np.uint8)

for m in range(0, height):
    for n in range(0, width):
        k=centers.index(shifted_list[m][n])
        res_img[m,n,:]=random_color[k,:]

cv2.imshow("Input", image)
cv2.imshow("Mean-shifted", shifted)
cv2.imshow("Random colored", res_img)
cv2.waitKey()

cv2.destroyAllWindows()