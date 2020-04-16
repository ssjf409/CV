import numpy as np
import cv2

dim=5
img = cv2.imread('../data/fruits.png')
height, width, channels=img.shape
xybgr=np.zeros([height*width, dim], np.float32)

k=0
for m in range(0, height):
    for n in range(0, width):
        xybgr[k,0]=img[m,n,0]/255.0
        xybgr[k,1]=img[m,n,1]/255.0
        xybgr[k,2]=img[m,n,2]/255.0
        xybgr[k,3]=n/width
        xybgr[k,4]=m/width
        k+=1
        
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 100, 1.0)
K = 8
ret,label,center=cv2.kmeans(xybgr,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center_color = np.uint8(255.0*center[:,0:3])
res = center_color[label.flatten()]
res_img = res.reshape((img.shape))

random_color = np.random.randint(0, 256, [K, 3], np.uint8)
res = random_color[label.flatten()]
res_img2=res.reshape((img.shape))

cv2.imshow('Input', img)
cv2.imshow('Mean-colored',res_img)
cv2.imwrite('../data/k-means_xyrgb_mean_colored.png', res_img)
cv2.imshow('Randomly colored', res_img2)
cv2.imwrite('../data/k-means_xyrgb_random_colored.png', res_img2)
cv2.waitKey(0)

cv2.destroyAllWindows()