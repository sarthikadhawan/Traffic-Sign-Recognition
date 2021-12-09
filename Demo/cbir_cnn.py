import numpy as np
from keras.layers import *
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from scipy.misc import imsave
import  numpy  as  np
from keras.layers import *
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input
from mapping import get_sign_name
import os
from keras.models import model_from_json
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

def build_model():
	# load model from json 
	json_file = open(r"C:\Users\MyPC\Desktop\IA project Final\cnn_feats.json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# load weights into model
	loaded_model.load_weights(r"C:\Users\MyPC\Desktop\IA project Final\cnn_feats.h5")
	model=loaded_model

	return model


def feature_extraction2(img):
	img_data = image.img_to_array(img)
	img_data = np.expand_dims(img_data, axis=0)
	img_data = preprocess_input(img_data)
        
	vgg_feature = model.predict(img_data)
	return vgg_feature

    
def feature_extraction(dataset):
    features = {}
    all_images = {}
   
    
    dirs = os.listdir(dataset)

    for file in dirs:
        dirs1 = os.listdir(dataset+"\\"+file)
        for f in dirs1:
            img_name = f.split('.')[0]
            path = dataset+"\\"+file+"\\"+f
            print(path)
            print(f)
            image = cv2.imread(path)
            all_images[img_name] = image
            image = cv2.resize(image, (100, 100)) 
            feature4 = feature_extraction2(image)
            features[img_name]=[]
            features[img_name].extend(feature4)
           
                
    return features,all_images



class Searcher:
    def __init__(self, features):
        self.features = features
     

    def search(self, query):
        results = {}

        for name, feature in self.features.items():
            query = np.array(query)
            feature = np.array(feature)
            dist = euclidean(query, feature)
            results[name] = dist

        results = sorted([(d, n) for n, d in results.items()])
        return results



def save(obj, obj2, path):
    with open(path+"\\features", "wb") as myFile:
        pickle.dump(obj, myFile)
    with open(path+"\\all_images", "wb") as myFile:
        pickle.dump(obj2, myFile)



model=build_model()
    

f,all_images = feature_extraction(r"C:\Users\MyPC\Desktop\IA project Final\\Training")
print("yes")
save(f,all_images,r"C:\Users\MyPC\Desktop\IA project Final")


features = {}
path = r"C:\Users\MyPC\Desktop\IA project Final"
with open(path + "\\features", "rb") as myFile:
	features = pickle.load(myFile)
all_images = {}
path = r"C:\Users\MyPC\Desktop\IA project Final"
with open(path + "\\all_images", "rb") as myFile:
	all_images = pickle.load(myFile)


	
searcher = Searcher(features)
img = cv2.imread(r"C:\Users\MyPC\Desktop\IA project Final\Images\CroppedImages\keep_right_5\Crop_img_1.png")
img = resized_image = cv2.resize(img, (100, 100)) 
f4 = feature_extraction2(img)
f = []
f.extend(f4)
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
    plt.imshow(all_images[img_name])
    j+=1

cv2.imshow("query img",img)
    
plt.show()



    




