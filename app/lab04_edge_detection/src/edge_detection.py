import cv2

def nothing(x):
    pass

cv2.namedWindow('Canny')
cv2.createTrackbar('low threshold', 'Canny', 0, 1000, nothing)
cv2.createTrackbar('high threshold', 'Canny', 0, 1000, nothing)
cv2.setTrackbarPos('low threshold', 'Canny', 50)
cv2.setTrackbarPos('high threshold', 'Canny', 150)

img = cv2.imread('../data/cat.jpg', cv2.IMREAD_GRAYSCALE)
blurred=cv2.GaussianBlur(img, (3,3), 0)

while (1):
    low = cv2.getTrackbarPos('low threshold', 'Canny')
    high = cv2.getTrackbarPos('high threshold', 'Canny')
    img_canny = cv2.Canny(img, low, high)
    cv2.imshow('Canny', img_canny)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()