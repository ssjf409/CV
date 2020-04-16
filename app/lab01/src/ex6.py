import numpy as np
import cv2

img = cv2.imread("../data/lena.png")#, cv2.IMREAD_COLOR)

cv2.imshow("lena", img)
cv2.waitKey()

lin_img = img.copy()

img_rows, img_cols, img_channels = lin_img.shape
print(img_rows, img_cols, img_channels)

'''
for m in range(img_rows):
    for n in range(img_cols):
        for k in range(img_channels):
            lin_img[m,n,k]=int(255*pow(lin_img[m,n,k]/255.0, 2.2)+0.5)
            '''
            
'''
for k in range(img_channels):
    lin_img[:,:,k]=255*np.power(lin_img[:,:,k]/255.0, 2.2)
    '''
    
lin_img[:,:,:]=np.int8(255.*np.power(lin_img[:,:,:]/255.0, 2.2)+0.5)
            
cv2.imshow("sBGR", lin_img)

cv2.waitKey(0)

xyz=cv2.cvtColor(lin_img, cv2.COLOR_BGR2XYZ)
#lab=cv2.cvtColor(lin_img, cv2.COLOR_BGR2Lab)
#hsv=cv2.cvtColor(lin_img, cv2.COLOR_BGR2HSV)

cv2.imshow("XYZ", xyz)
#cv2.imshow("Lab", lab)
#cv2.imshow("HSV", hsv)

cv2.waitKey(0)

cv2.destroyAllWindows()