import cv2
import os

cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0
while(True):
    ret, img = cam.read()
    faces = face_detector.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), 2)     
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("images/User_" + str(count) + ".jpg", img[y1:y2,x1:x2])  #Change "User to your name"
        cv2.imshow('image', img)
    k = cv2.waitKey(200) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 1: # Take 1 face sample and stop video
         break
cam.release()
cv2.destroyAllWindows()
