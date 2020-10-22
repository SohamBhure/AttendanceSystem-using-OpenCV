import cv2
import numpy as np

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

cam=cv2.VideoCapture(0);

rec=cv2.createLBPHFaceRecognizer();
rec.load("recognizer\\trainingData.yml")
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,10,1,0,1)

while(1):
    ret,img=cam.read();                                                  #returns a variable and the captured image
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                            #converting coloured image to grayscale
    faces=facedetect.detectMultiScale(gray,1.3,5);                       
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0), 2)                  #rectangle to show face
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        
        cv2.cv.PutText(cv2.cv.fromarray(img), str(id), (x,y+h),font,255);
    cv2.imshow("Face",img);                                              #to show output img

    if(cv2.waitKey(1)==ord('e')):
        break;

cam.release()
cv2.destroyAllWindows()
    
