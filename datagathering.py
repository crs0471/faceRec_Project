import cv2 as cv
import os

faceCasscade = cv.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

cam = cv.VideoCapture(0)
cam.set(3,800)
cam.set(4,500)
count = 0
idname = {}
faceId = input('\nenter userid and press <return> : ')
name = input('\nenter name and press <return> : ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

idname[int(faceId)] = name

while True:
    ret, frame = cam.read()
    frame = cv.flip(frame,1)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    faces = faceCasscade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20,20)
    )

    for (x,y,w,h) in faces:
        count+=1
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_frame = frame[y:y+h,x:x+w]
        roi_gray  = gray[y:y+h,x:x+w]
        cv.imwrite("./dataset/user."+str(faceId)+"."+str(count)+".jpg",roi_gray)

    cv.imshow('orignal',frame)
    #cv.imshow('gray',gray)

    k = cv.waitKey(100)
    if k == 27:
        break
    elif count >=30:
        break

cam.release()
cv.destroyAllWindows()


