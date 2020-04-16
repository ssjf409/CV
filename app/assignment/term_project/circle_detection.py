from glob import glob
import cv2
import numpy as np

# write your image path
img_path = '../data/training/*.jpg'
img_names = glob(img_path)

for fn in img_names:
    # Open the image.
    img = cv2.imread(fn)
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Write your code for circle detection here.
    # Write your circle detect code here. 
    # We are going to copy and paste your code here.
    # Below is an example of using the "HoughCircles" function in OpenCV.
    # If you just use or optimize this function, your score will be multiplied by 0.5  
    gimg = cv2.medianBlur(gimg, 5)
    circles = cv2.HoughCircles(gimg, cv2.HOUGH_GRADIENT, 1, 50,
                            param1=20, param2=40, minRadius=0, maxRadius=70)

    # Please do not change the below, which saves the circle parameters and result images. 
    icircles = np.uint16(np.around(circles))

    # write your circles x,y radius
    fn1 = fn.split('\\')
    fn2 = fn1[-1].split('.')
    fn_ = fn2[0]
    #print(fn_)

    save_path='../result/{}.txt'.format(fn_)
    with open(save_path, 'w') as f:
        f.write("{0} {1} {2}\n".format(circles[0][0][0],circles[0][0][1],circles[0][0][2]))     
  
    i=icircles[0][0]
    cv2.circle(img, (i[0], i[1]), i[2] ,(0, 255, 0), 2)
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
                                                                                 
    # write your save_path
    save_path = '../result/{}.jpg'.format(fn_)
                                      
    # write your image your path                                
    cv2.imwrite(save_path, img)

    
