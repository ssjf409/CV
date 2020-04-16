import cv2
#import numpy as np

img = cv2.imread('../data/number2.jpg')
original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
th = 127;
_, binary = cv2.threshold(original, th, 255, cv2.THRESH_BINARY)
binary=~binary
cv2.imshow('binary', binary)
cv2.waitKey(0)

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
eroded = cv2.erode(binary, element, iterations=1)
dilated = cv2.dilate(binary, element, iterations=1)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)

cv2.imshow('original',original)
cv2.imshow('eroded',eroded)
cv2.imshow('dilated',dilated)
cv2.imshow('closed', closed)
cv2.imshow('opened',opened)
cv2.waitKey(0)

cv2.destroyAllWindows()