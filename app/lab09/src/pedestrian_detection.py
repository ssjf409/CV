import cv2

#Read Video & Classifier
cap1=cv2.VideoCapture("../data/mitsubishi_768x576.avi")
fullbody_name="../data/haarcascade_fullbody.xml"
fullbody=cv2.CascadeClassifier(fullbody_name)

if cap1.isOpened()==False:
    print("not read file \n")

cv2.namedWindow("video")

while(cap1.isOpened()):
    #read frame
    ret, frame1 = cap1.read()
    original=frame1.copy()

    #Converts to a gray image & Histogram equalization
    if frame1.shape[2]>1 :
        gray=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
        gray=cv2.equalizeHist(gray)

    else:
        gray=frame1

    #Detect pedestraians
    pedestraians=[]
    pedestraians=fullbody.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2,flags=0,minSize=(30,30),maxSize=(150,150))
    #print(pedestraians,'1')
    for i in range(len(pedestraians)):
        cx=int(pedestraians[i][0]+pedestraians[i][2]*0.5)
        cy=int(pedestraians[i][1]+pedestraians[i][3]*0.5)

        w=int(pedestraians[i][1]*0.5)
        h=int(pedestraians[i][3]*0.5)
#        print(cx, cy, w, h)
        cv2.ellipse(frame1,(cx,cy),(h,w),0,0,360,(0,0,255),4,8,0)
    cv2.imshow("video",frame1)
    if cv2.waitKey(20)==27:break

cap1.release()
cv2.destroyAllWindows()
        