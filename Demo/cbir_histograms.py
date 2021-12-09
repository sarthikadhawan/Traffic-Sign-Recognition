import numpy as np
import cv2
from PIL import Image, ImageEnhance
import sys
import scipy
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from scipy import ndimage
import math
from matplotlib import pyplot as plt
import glob
import pickle
import os


class RGBHistogram:

    def __init__(self, bins, which):
        self.bins = bins
        self.which = which

    def describe(self, image):
        if(self.which==1):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      
            hist = cv2.calcHist(images=[image], channels=[0, 1, 2], mask=None,
                                histSize=self.bins, ranges=[0, 256] * 3)
            hist = cv2.normalize(hist, dst=hist.shape)
            return hist.flatten()
        
        if(self.which==2):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
            hist = cv2.calcHist(images=[image], channels=[0], mask=None,
                                histSize=self.bins, ranges=[0, 256])
            hist = cv2.normalize(hist, dst=hist.shape)
            return hist.flatten()
        
        if(self.which==3):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
      
            hist = cv2.calcHist(images=[image], channels=[0, 1], mask=None,
                                histSize=self.bins, ranges=[0, 180, 0, 256])
            hist = cv2.normalize(hist, dst=hist.shape)
            return hist.flatten()

def feature_extraction(dataset):
    features = {}
    all_images = {}
##    descriptor = RGBHistogram(bins=[8, 8, 8],which=1)
##    descriptor = RGBHistogram(bins=[256],which=2)
    descriptor = RGBHistogram(bins=[180,256],which=3)
##    
    dirs = os.listdir(dataset)

    for file in dirs:
        dirs1 = os.listdir(dataset+"\\"+file)
        for f in dirs1:
            img_name = f.split('.')[0]
            path = dataset+"\\"+file+"\\"+f
            print(path)
            image = cv2.imread(path)
            all_images[img_name] = image
            image = cv2.resize(image, (100, 100)) 
            feature = descriptor.describe(image)
            features[img_name]=feature
    return features,all_images


class Searcher:
    def __init__(self, features):
        self.features = features

    def search(self, query):
        results = {}

        for name, feature in self.features.items():
            dist = euclidean(query, feature)
            results[name] = dist

        results = sorted([(d, n) for n, d in results.items()])
        return results

def save(obj, path):
    with open(path+"\\features", "wb") as myFile:
        pickle.dump(obj, myFile)

f,all_images = feature_extraction(r"C:\Users\MyPC\Desktop\IA project Final\\Training")

features = f



##descriptor = RGBHistogram(bins=[8, 8, 8],which=1)
##descriptor = RGBHistogram(bins=[256],which=2)
descriptor = RGBHistogram(bins=[180,256],which=3)



searcher = Searcher(features)

img = cv2.imread(r"C:\Users\MyPC\Desktop\IA project Final\Images\CroppedImages\keep_right_5\Crop_img_1.png")
img = resized_image = cv2.resize(img, (100, 100))

f = descriptor.describe(img)
results = searcher.search(f)


for result in results:
    dis = result[0]
    print( result[1] + "  " + str(dis))

w=10
h=10
fig = plt.figure(figsize=(8,8))
columns = 5
rows = 3
j = 0
for i in range(1,columns*rows+1):
    fig.add_subplot(rows,columns,i)
    img_name = results[j][1]
    print(img_name)
    plt.imshow(all_images[img_name]/255)
    j+=1

cv2.imshow("query img",img)
    
plt.show()


