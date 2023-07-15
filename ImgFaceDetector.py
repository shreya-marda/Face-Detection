import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("test.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4)
#detectMultiScale(image, scaleFactor, minNeighbors)
#scalefactor: Parameter specifying how much the image size is reduces at each image scale
#minNeighbor: Parameter specifying how many neighbors each candidate recatngle should have to retain it
#more the value of scalefactor and minneighbors better the result

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()