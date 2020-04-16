import numpy as np
import cv2

img = cv2.imread('../data/fruits.png')
x = img.reshape((-1,3))
#print(x.shape)
# convert to np.float32
x = np.float32(x)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 100, 1.0)
K = 8
ret,label,center=cv2.kmeans(x,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res_img = res.reshape((img.shape))

random_color = np.random.randint(0, 256, center.shape, np.uint8)
res = random_color[label.flatten()]
res_img2=res.reshape((img.shape))

cv2.imshow('Input', img)
cv2.imshow('Mean-colored',res_img)
cv2.imwrite('../data/k-means_rgb_mean_colored.png', res_img)
cv2.imshow('Randomly colored', res_img2)
cv2.imwrite('../data/k-means_rgb_random_colored.png', res_img2)
cv2.waitKey(0)

cv2.destroyAllWindows()