import cv2
import numpy as np

def nothing(x):
    pass

img = np.zeros((240,320,3), np.uint8)
cv2.namedWindow('Trackbar')

cv2.createTrackbar('R','Trackbar',0,255,nothing)
cv2.createTrackbar('G','Trackbar',0,255,nothing)
cv2.createTrackbar('B','Trackbar',0,255,nothing)

switch = '0:OFF \ 1:ON'
cv2.createTrackbar(switch, 'Trackbar', 0,1, nothing)

while True:
    cv2.imshow('Trackbar', img)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    r = cv2.getTrackbarPos('R','Trackbar')
    g = cv2.getTrackbarPos('G','Trackbar')
    b = cv2.getTrackbarPos('B','Trackbar')
    s = cv2.getTrackbarPos(switch,'Trackbar')
    
    if s == 0:
        img[:,:,:] = 0
    else:
        img[:,:,:] = [b,g,r]

cv2.destroyAllWindows()
