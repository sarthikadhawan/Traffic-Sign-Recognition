import numpy as np
import cv2
from PIL import Image, ImageEnhance
import sys
import scipy
from scipy import ndimage
import math
from matplotlib import pyplot as plt
path=""
def check_aspect_ratio(aspect_ratio):
    if(aspect_ratio<=1.9 and aspect_ratio>=1/1.9):
        return True
    return False
def check_dimension_within_range(img_height,img_width,crop_img_height,crop_img_width):
    if(img_height!=crop_img_height and img_width!=crop_img_width and min(crop_img_height,crop_img_width)>=min(img_width,img_height)/10.0):
        return True
    return False

def add_motion_blurr(img):
    size = 15
    #generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    output = cv2.filter2D(img, -1, kernel_motion_blur)
    return output

def enhance(img):
    kernel3 = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,-476,24,6],[4,16,24,16,4],[1,4,6,4,1]])
    kernel3 = kernel3/(-256)
    output = cv2.filter2D(img, -1, kernel3)
    return output
def sharpen(img):
    kernel2 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    output = cv2.filter2D(img,-1,kernel2)
    return output

def adapatative_histogram_equalisation(img):
    bgr=img
    #convert image from rgb to lab - l:lightness, a:green-red and b-blue yellow, numerical values
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    #split into l,a,b
    lab_planes = cv2.split(lab)
    #clahe on lightness
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    #merge with image
    lab = cv2.merge(lab_planes)
    #convert back to bgr
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

def read_image(title,type):
    directory = r"C:\Users\MyPC\Desktop\IA project Final\Images\CroppedImages"
    path = directory+title
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    img = cv2.imread(r'C:\Users\MyPC\Desktop\IA project Final\Images'+title+'.'+type)
    return img,path

def detect_red(img):
    img_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
    mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
    mask = cv2.bitwise_or(mask1, mask2 )
    output = cv2.bitwise_and(crop_img,crop_img,mask=mask)
    return output
def detect_blue(img):
    img_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, (90,127,127), (130,255,255))
    mask2 = cv2.inRange(img_hsv, (86,31,4), (220,88,50))
    output = cv2.bitwise_and(crop_img,crop_img,mask=mask+mask2)
    return output

def get_crop_image(img,x,y,w,h,wdth,ht):
   
    h1 = int(h+h*0.4)
    w1 = int(w+0.4*w)
    x1 = int(x-0.2*w)
    y1 = int(y-0.2*h)
    if(x1>=0):
        x = x1
    if(y1>=0):
        y = y1
    if(w<=wdth):
        w = w1
    if(h<=ht):
        h = h1
    crop_img = img[y:y+h, x:x+w]
    return crop_img






img,path = read_image(r"\icy_road","jpg")
img = enhance(img)
img = adapatative_histogram_equalisation(img)
img = denoise(img)

#covert to graysale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#do thresholding
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
image, contours, h = cv2.findContours(thresh,1,2)
number=1
for i, cnt in enumerate (contours):
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    
    x, y, w, h = cv2.boundingRect(cnt)
    wdth, ht, g = img.shape

    crop_img = get_crop_image(img,x,y,w,h,wdth,ht)
    output_red = detect_red(crop_img)
    crop_img_width = crop_img.shape[0]
    crop_img_height = crop_img.shape[1]
    img_width = img.shape[0]
    img_height = img.shape[1]

    crop_img_num_red_pixels = np.count_nonzero(output_red)
    crop_img_num_pixels = output_red.shape[0]*output_red.shape[1]

    
    aspect_ratio = float(crop_img_width/crop_img_height)
    color_ratio_red = float(crop_img_num_red_pixels/crop_img_num_pixels)
    

    if(not np.all(output_red==0) and color_ratio_red>0.1 and check_aspect_ratio(aspect_ratio) and check_dimension_within_range(img_height,img_width,crop_img_height,crop_img_width)):
       if len(approx)==3 or len(approx)==8:
           print(approx)
           break
           cv2.imwrite(path+r"\Crop_img_"+str(number)+".png",crop_img)
           number+=1
            
    output_blue = detect_blue(crop_img)
    crop_img_num_blue_pixels = np.count_nonzero(output_blue)
    crop_img_num_pixels = output_blue.shape[0]*output_blue.shape[1]
    color_ratio_blue = float(crop_img_num_blue_pixels/crop_img_num_pixels)

    if(not np.all(output_blue==0) and color_ratio_blue>0.1 and check_aspect_ratio(aspect_ratio) and check_dimension_within_range(img_height,img_width,crop_img_height,crop_img_width)):
       if len(approx) > 10 or len(approx)==3 or len(approx)==8:
            cv2.imwrite(path+r"\Crop_img_"+str(number)+".png",crop_img)
            number+=1
