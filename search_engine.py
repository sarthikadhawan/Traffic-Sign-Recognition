import numpy as np
import cv2
from PIL import Image, ImageEnhance
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    descriptor1 = RGBHistogram(bins=[8,8,8],which=1)
    descriptor2 = RGBHistogram(bins=[256],which=2)
    descriptor3 = RGBHistogram(bins=[180,256],which=3)
    
    dirs = os.listdir(dataset)

    for file in dirs:
        dirs1 = os.listdir(dataset+"\\"+file)
        for f in dirs1:
            img_name = f.split('.')[0]
            path = dataset+"\\"+file+"\\"+f
            print(path)
            image = cv2.imread(path)
            
            feature1 = descriptor1.describe(image)
            feature2 = descriptor2.describe(image)
            feature3 = descriptor3.describe(image)
            x=[]
            x.extend(feature1)
            x.extend(feature2)
            x.extend(feature3)
            features[img_name]=pca.fit_transform(x)
    return features
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







##feature_dataset = np.array(feature_dataset)
##feature_dataset = StandardScaler().fit_transform(feature_dataset)
##print(feature_dataset[0])
##pca = PCA(n_components = 55)
##principalComponents = pca.fit_transform(feature_dataset)
##std_deviation_of_principal_components = np.std(principalComponents,axis = 0)
##variance_of_principal_components = np.array([x**2 for x in std_deviation_of_principal_components])
##variance_of_principal_components = np.array(variance_of_principal_components)
##sum_variance_of_principal_components = np.sum(variance_of_principal_components)
##proportion_variance_principal_components = np.array([(float(x)/sum_variance_of_principal_components) for x in variance_of_principal_components])
##cumulative_scores = []
##number_of_components = []
##epsilon = 0.002
##score = 0
##num_principal_components = 0
##for i in range(55):
##	score+=proportion_variance_principal_components[i]
##	if proportion_variance_principal_components[i]<epsilon and num_principal_components==0:
##		num_principal_components=i+1
##	number_of_components.append(i+1)
##	cumulative_scores.append(score)
##plt.plot(number_of_components,cumulative_scores)
##
##plt.title("Plot to obtain optimal number of components")
##plt.ylabel("cumulative proportion of variance of components")
##plt.xlabel("Number of principal components")
##plt.show()
##print(num_principal_components)
num_principal_components=39
pca = PCA(n_components = num_principal_components)


##f = feature_extraction(r"C:\Users\MyPC\Desktop\IA project Final\\Training")
##print("yes")
##save(f,r"C:\Users\MyPC\Desktop\IA project Final")

features = {}
path = r"C:\Users\MyPC\Desktop\IA project Final"
with open(path + "\\features", "rb") as myFile:
	features = pickle.load(myFile)
feature_dataset=[]
for feature in features.items():
    feature_dataset.append(feature[1])

feature_dataset1 = pca.fit_transform(feature_dataset)

for i,feature in enumerate(features.items()):
    features[feature[0]] = feature_dataset1[i]

    
descriptor1 = RGBHistogram(bins=[8, 8, 8],which=1)
descriptor2 = RGBHistogram(bins=[256],which=2)
descriptor3 = RGBHistogram(bins=[180,256],which=3)

searcher = Searcher(features)
img = cv2.imread(r"C:\Users\MyPC\Desktop\IA project Final\Images\CroppedImages\keep_right_5\Crop_img_1.png")


f1 = descriptor1.describe(img)
f2 = descriptor2.describe(img)
f3 = descriptor3.describe(img)
f = []
f.extend(f1)
f.extend(f2)
f.extend(f3)
feature_dataset.append(f)
fs = pca.transform(feature_dataset)
ff = fs[-1]
results = searcher.search(ff)
print(results[0])

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
    path = r"C:\Users\MyPC\Desktop\IA project Final\Training\\"
    x = results[j][1].split("_")[0]
    for y in range(1,len(results[j][1].split("_"))):
        if(y<len(results[j][1].split("_"))-1):
           x+="_"+results[j][1].split("_")[y]

           
    path+= x +"\\" +  results[j][1] + ".png"
    path = os.path.normpath(path)
    print(path)
    j+=1
    img1 = plt.imread(path)
       
    fig.add_subplot(rows,columns,i)
    plt.imshow(img1)

cv2.imshow("query img",img)
    
plt.show()



    




