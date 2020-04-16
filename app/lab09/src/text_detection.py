import cv2
import numpy as np

#Read Image & Classifier
img=cv2.imread("../data/cvImage.jpg")
src=img.copy()

channels=channels = cv2.text.computeNMChannels(src)

text_classifier_name1 = "../data/trained_classifierNM1.xml"
text_classifier_name2 = "../data/trained_classifierNM2.xml"
erGrouping = "../data/trained_classifier_erGrouping.xml"


for c in range(0,len(channels)-1):
    channels.append((255-channels[c]))

for channel in channels:

  erc1 = cv2.text.loadClassifierNM1(text_classifier_name1)
  er1 = cv2.text.createERFilterNM1(erc1,16,0.00015,0.13,0.2,True,0.1)

  erc2 = cv2.text.loadClassifierNM2(text_classifier_name2)
  er2 = cv2.text.createERFilterNM2(erc2,0.5)

  regions = cv2.text.detectRegions(channel,er1,er2)

  rects = cv2.text.erGrouping(img,channel,[r.tolist() for r in regions])
  rects = cv2.text.erGrouping(img,channel,[x.tolist() for x in regions], cv2.text.ERGROUPING_ORIENTATION_ANY,erGrouping,0.5)

  #Visualization
  for r in range(0,np.shape(rects)[0]):
    rect = rects[r]
    cv2.rectangle(src, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 255), 3)


#Visualization
cv2.imshow("Text detection result", src)
cv2.waitKey(0)

cv2.destroyAllWindows()