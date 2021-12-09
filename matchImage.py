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
    """
    Image descriptor using color histogram.

    :param bins: list
        Histogram size. 1-D list containing ideal values
        between 8 and 128; but you can go up till 0 - 256.

    Example:
        >>> histogram = RGBHistogram(bins=[32, 32, 32])
        >>> feature_vector = histogram.describe(image='folder/image.jpg')
        >>> print(feature_vector.shape)
    """

    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        """
        Color description of a given image

        compute a 3D histogram in the RGB color space,
        then normalize the histogram so that images
        with the same content, but either scaled larger
        or smaller will have (roughly) the same histogram

        :param image:
            Image to be described.
        :return: flattened 3-D histogram
            Flattened descriptor [feature vector].
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
        hist = cv2.calcHist(images=[image], channels=[0], mask=None,
                            histSize=self.bins, ranges=[0,256])
        hist = cv2.normalize(hist, dst=hist.shape)
        return hist.flatten()

def feature_extraction(dataset):
    features = {}
    descriptor = RGBHistogram(bins=[256])

    dirs = os.listdir(dataset)

    for file in dirs:
        dirs1 = os.listdir(dataset+"\\"+file)
        for f in dirs1:
            img_name = f.split('.')[0]
            path = dataset+"\\"+file+"\\"+f
            print(path)
            image = cv2.imread(path)
            
            feature = descriptor.describe(image)
            features[img_name]=feature
    return features
class Searcher:
    def __init__(self, features):
        self.features = features

    def search(self, query):
        results = {}

        for name, feature in self.features.items():
            dist = cosine(query, feature)
            results[name] = dist

        results = sorted([(d, n) for n, d in results.items()])
        return results

def save(obj, path):
    with open(path+"\\features", "wb") as myFile:
        pickle.dump(obj, myFile)

f = feature_extraction(r"C:\Users\MyPC\Desktop\IA project Final\\Training")
print(f)
save(f,r"C:\Users\MyPC\Desktop\IA project Final")

features = {}
path = r"C:\Users\MyPC\Desktop\IA project Final"
with open(path + "\\features", "rb") as myFile:
	features = pickle.load(myFile)

path = r"C:\Users\MyPC\Desktop\IA project Final\Images\\CroppedImages"

dirs = os.listdir(path)
descriptor = RGBHistogram(bins=[256])

searcher = Searcher(features)
for file in dirs:
    dirs_1 = os.listdir(path+"\\"+file)
    for f in dirs_1:
        img_name = f.split(".")[0]
        
        p = path+"\\"+file+"\\"+f
        image=cv2.imread(p)
        f1 = descriptor.describe(image)
        results = searcher.search(f1)
        print(file)
        for result in results:
            name = result[1]
            dis = result[0]
            name = name.split("_")[0]
            print( result[1] + "  " + str(dis))
        break
    break
    print("-------------------")






