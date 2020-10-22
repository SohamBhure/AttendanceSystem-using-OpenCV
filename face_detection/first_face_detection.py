import cv2
import numpy as np

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

cam=cv2.VideoCapture(0);

while(1):

    ret,img=cam.read();                                                  #returns a variable (bool) and the captured image
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                            #converting coloured image to grayscale for the classifier to work
    faces=facedetect.detectMultiScale(gray,1.3,5);                       #method to search for face rectangular coordinates. 1.3 is the scale factor. Decreases the shape value by 5%

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0), 2)                  #rectangle to show face

    cv2.imshow("Face",img);                                              #to show output img. Window name is "Face".

    if(cv2.waitKey(1)==ord('q')):                                        #end the program when "q" is pressed
        break;

cam.release()

cv2.destroyALLWindows()                                                  #closes all windows
    
