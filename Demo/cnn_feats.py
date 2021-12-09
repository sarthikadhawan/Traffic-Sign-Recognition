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
import pickle

def build_model():
	model = Sequential()
	model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(100, 100, 3), padding='VALID'))
	model.add(Conv2D(64, kernel_size=(3, 3), padding='VALID'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Block 2
	model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
	model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
	model.add(AveragePooling2D(pool_size=(19, 19)))

	# set of FC => RELU layers
	model.add(Flatten())

	model_json = model.to_json()
	with open("cnn_feats.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("cnn_feats.h5")

	# load model from json 
	json_file = open("cnn_feats.json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# load weights into model
	loaded_model.load_weights("cnn_feats.h5")
	model=loaded_model

	#getting the summary of the model (architecture)
	# model.summary()	

	return model

model=build_model()

def feature_extraction(img_path):
	img = image.load_img(img_path, target_size=(100,100))
	img_data = image.img_to_array(img)
	img_data = np.expand_dims(img_data, axis=0)
	img_data = preprocess_input(img_data)

	vgg_feature = model.predict(img_data)
	return vgg_feature

def get_data():
	train_x=[]
	test_x=[]
	train_y=[]
	test_y=[]
	labelslist={}
	c=0
	labelslist={}
	for folder in os.listdir('../CBIR-master/test_database/'):

		if len(folder.split('.'))>1 and folder.split('.')[1]=='txt':
			continue
		dir=os.listdir('../CBIR-master/test_database/'+folder) 

		if get_sign_name(int(folder)) not in labelslist:
			labelslist[get_sign_name(int(folder)).lower()]=c
			c=c+1

	for folder in os.listdir('../CBIR-master/test_database/'):
		if len(folder.split('.'))>1 and folder.split('.')[1]=='txt':
			continue
		dir=os.listdir('../CBIR-master/test_database/'+folder) 

		for file in dir:
			if file.split('.')[1]=='csv':
				continue
			file='../CBIR-master/test_database/'+folder+'/'+file
			final_array=list(feature_extraction(file).reshape(-1,1))
			mainlist=[]
			for f in final_array:
				mainlist.append(f[0])
			train_x.append(mainlist)
			train_y.append(labelslist[get_sign_name(int(folder)).lower()])
		

	with open('train_x', 'wb') as fp:
    		pickle.dump(train_x, fp)
	with open('train_y', 'wb') as fp:
    		pickle.dump(train_y, fp)

	with open ('train_x', 'rb') as fp:
	    train_x = pickle.load(fp)
	with open ('train_y', 'rb') as fp:
	    train_y = pickle.load(fp)

	print 'train'

	for folder in os.listdir('Traffic-Sign-Recognition-master/Images/CroppedImages'):

		if len(folder.split('.'))>1 and folder.split('.')[1]=='txt':
			continue
		dir=os.listdir('Traffic-Sign-Recognition-master/Images/CroppedImages/'+folder)    
		for file in dir:
			file='Traffic-Sign-Recognition-master/Images/CroppedImages/'+folder+'/'+file
			final_array=list(feature_extraction(file).reshape(-1,1))
			mainlist=[]
			for f in final_array:
				mainlist.append(f[0])
			test_x.append(mainlist)
			label=' '.join(folder.split('_'))
			test_y.append(labelslist[label.lower()])

	with open('test_x', 'wb') as fp:
    		pickle.dump(test_x, fp)
	with open('test_y', 'wb') as fp:
    		pickle.dump(test_y, fp)

 	with open ('test_x', 'rb') as fp:
	    test_x = pickle.load(fp)
	with open ('test_y', 'rb') as fp:
	    test_y = pickle.load(fp)

	print 'test'

	
	return train_x,train_y,test_x,test_y



# with open ('outfile', 'rb') as fp:
#     itemlist = pickle.load(fp)