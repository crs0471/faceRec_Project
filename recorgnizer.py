import  cv2 as cv
import os
import  numpy as np

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("./trainer/trainer.yml")

cascadePath = "./haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascadePath)

font =cv.FONT_HERSHEY_SIMPLEX

id =0
names = ['none','cheris','hitesh','mrunal','anandi','crs']

cam = cv.VideoCapture(0)
cam.set(3,1800)
cam.set(4,950)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read()
    img = cv.flip(img,1)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20,20)
    )

    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id, confidance = recognizer.predict(gray[y:y+h,x:x+w])

        if(confidance < 100):
            id = names[id]
            confidance = "{0}%".format(round(100-confidance))
        else:
            id="unknown"
            confidance = "{0}%".format(round(100-confidance))

        cv.putText(img,str(id),(x+5,y-50),font,1,(0,0,255),2)
        cv.putText(img,str(confidance),(x+5,y-100),font,1,(0,0,255),2)

        cv.imshow("live view",img)

    k = cv.waitKey(1)
    if k == 27:
        break

cam.release()
cv.destroyAllWindows()
