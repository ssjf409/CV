import cv2

img1 = cv2.imread('../data/deer.jpg')
img2 = cv2.imread('../data/catdog.jpg')

#cv2.imshow("deer", img1)
#cv2.imshow("catdog", img2)
#cv2.waitKey(1000)

img1 = cv2.resize(img1, (320, 240))
img2 = cv2.resize(img2, (320, 240))

def nothing(x):
    pass

cv2.namedWindow('Dissolve')
cv2.createTrackbar('W', 'Dissolve', 0, 100, nothing)

while True:

    w = cv2.getTrackbarPos('W','Dissolve')
    print(w)
    dst = cv2.addWeighted(img1,float(100-w) * 0.01, img2,float(w) * 0.01,0)

    cv2.imshow('Dissolve', dst)

    if cv2.waitKey(1) & 0xFF == 27:
        break;

cv2.destroyAllWindows()