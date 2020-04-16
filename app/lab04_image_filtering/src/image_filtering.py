import numpy as np
import cv2
from skimage.util import random_noise

img = cv2.imread('../data/cat.jpg', cv2.IMREAD_GRAYSCALE)
#noisy = random_noise(img, mode='s&p', amount=0.05)
noisy = random_noise(img, var=0.01)
noisy = np.array(noisy, dtype=np.float32)

bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)
gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
median = cv2.medianBlur(noisy, 5)

cv2.imshow('noisy', noisy)
cv2.imshow('bilateral', bilateral)
cv2.imshow('guassian', gaussian)
cv2.imshow('median', median)
cv2.waitKey()

cv2.destroyAllWindows()