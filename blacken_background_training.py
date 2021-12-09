import os
import numpy as np
import cv2  
from os import path
path1 = r"C:\Users\MyPC\Desktop\train_database"
dirs = os.listdir(path1)
color = (0,255,255)
j = 1
num = 1
for file in dirs:
    path2 = path1 + "\\" + file
    dirs2 = os.listdir(path2)
    if(j==num):
        for f in dirs2:
            img = cv2.imread(path2+"\\"+f)
            img2 = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
            image, contours, h = cv2.findContours(thresh,1,2)
            for i, cnt in enumerate (contours):
             
                approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
                if len(approx) > 10 or len(approx)==3 or len(approx)==8:
                   print("yes")
                   cv2.fillPoly(img2,[approx],color)
                   mask1 = cv2.inRange(img2,color,color)
                   img2 = cv2.bitwise_and(img2,img2,mask=mask1)
                   img2[np.where((img2==[0,255,255]).all(axis=2))] = [255,255,255]
                   img2 = cv2.bitwise_and(img2,img)
                   cv2.imwrite(path2+r"\\z"+f,img2)
    j+=1
               
                   
                    
           

