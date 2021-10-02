import  cv2 as cv

cam = cv.VideoCapture(0)
cam.set(3,640)
cam.set(4,460)

faceCascade = cv.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

while True:
    ret, frame = cam.read()
    frame = cv.flip(frame,1)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(frame,scaleFactor=1.2,minNeighbors=5,minSize=(20,20))

    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_frame = frame[y:y+h,x:x+w]

    cv.imshow('Live',frame)



    k = cv.waitKey(30)
    if k == 27:
        break

cam.release()
cv.destroyAllWindows()