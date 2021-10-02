import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

faceCascade = cv.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
print(faceCascade)

while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for (x, y, w, h) in faces:
        print("x,y,w,h : ",x,y,w,h)
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_img = frame[y:y+h, x:x+w]

    cv.imshow('frame', frame)

    k = cv.waitKey(30)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
