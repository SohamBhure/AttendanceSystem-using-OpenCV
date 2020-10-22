import cv2
import numpy as np
import xlwt
import xlrd
from xlrd import open_workbook
from xlutils.copy import copy
import xlsxwriter
import datetime

rd=open_workbook("FromPython.xls")
wb=copy(rd)
s=wb.get_sheet(0)

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
rec=cv2.createLBPHFaceRecognizer();
rec.load("recognizer\\trainingData.yml")
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,1)

i=str(datetime.date.today())

ab=i[8:10:1]

while(1):
    boolval,img=cam.read();                                                  #returns a variable and the captured image
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                            #converting coloured image to grayscale
    faces=facedetect.detectMultiScale(gray,1.3,5);                       #list to store faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0), 2)                  #rectangle to show face
        id,conf=rec.predict(gray[y:y+h,x:x+w])

        for j in range(20):
            if id==j:
                s.write(j+1,int(ab)+2,int(1))
                wb.save("FromPython.xls")
               
        
        cv2.cv.PutText(cv2.cv.fromarray(img), str(id), (x,y+h),font,255);


    cv2.imshow("Face",img);                                              #to show output img
    if(cv2.waitKey(1)==ord('e')):
        break;


cam.release()
cv2.destroyAllWindows()


    
