import os
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import librosa
import pandas
import joblib
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.manifold import TSNE
import cv2
from skimage.feature import hog
from mapping import get_sign_name
from cnn_feats import get_data
import sys

train_x=[]
test_x=[]
train_y=[]
test_y=[]
labelslist={}

# c=0
# for folder in os.listdir('../CBIR-master/test_database/'):

# 	if len(folder.split('.'))>1 and folder.split('.')[1]=='txt':
# 		continue
# 	dir=os.listdir('../CBIR-master/test_database/'+folder) 

# 	if get_sign_name(int(folder)) not in labelslist:
# 		labelslist[get_sign_name(int(folder)).lower()]=c
# 		c=c+1

# for folder in os.listdir('../CBIR-master/test_database/'):

# 	if len(folder.split('.'))>1 and folder.split('.')[1]=='txt':
# 		continue
# 	dir=os.listdir('../CBIR-master/test_database/'+folder) 

# 	for file in dir:
# 		if file.split('.')[1]=='csv':
# 			continue
# 		file='../CBIR-master/test_database/'+folder+'/'+file
# 		image= cv2.imread(file)
# 		image=cv2.resize(image,(100,100))
		
# 		# sift = cv2.xfeatures2d.SIFT_create()
# 		# descriptors = np.zeros((1,128)) #Matrix to hold the descriptors 
# 		# for i,img in enumerate(image):
# 		# 	kp, des     = sift.detectAndCompute(image,None)
# 		# 	descriptors = np.concatenate((descriptors,des),axis=0)

# 		# descriptors = descriptors[1:,:]

# 		fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, multichannel=True)
# 		train_x.append(fd)
# 		train_y.append(labelslist[get_sign_name(int(folder)).lower()])



# for folder in os.listdir('Traffic-Sign-Recognition-master/Images/CroppedImages'):

# 	if len(folder.split('.'))>1 and folder.split('.')[1]=='txt':
# 		continue
# 	dir=os.listdir('Traffic-Sign-Recognition-master/Images/CroppedImages/'+folder)    
# 	for file in dir:
# 		file='Traffic-Sign-Recognition-master/Images/CroppedImages/'+folder+'/'+file
# 		image= cv2.imread(file)
# 		image=cv2.resize(image,(100,100))

# 		# sift = cv2.xfeatures2d.SIFT_create()
# 		# descriptors = np.zeros((1,128)) #Matrix to hold the descriptors 
# 		# for i,img in enumerate(image):
# 		# 	kp, des     = sift.detectAndCompute(image,None)
# 		# 	descriptors = np.concatenate((descriptors,des),axis=0)

# 		# descriptors = descriptors[1:,:]
 
# 		fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, multichannel=True)      
# 		test_x.append(fd)
# 		label=' '.join(folder.split('_'))
# 		test_y.append(labelslist[label.lower()])

train_x,train_y,test_x,test_y=get_data()

# is_created=False
# for f in train_x:
#    if not is_created:
#        dataset_numpy = np.array(f)
#        is_created = True
#    elif is_created:
#        dataset_numpy = np.vstack((dataset_numpy, f))

# scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
# dataset_numpy = scaler.fit_transform(dataset_numpy)
# train_x= dataset_numpy

# is_created=False
# for f in test_x:
#    if not is_created:
#        dataset_numpy = np.array(f)
#        is_created = True
#    elif is_created:
#        dataset_numpy = np.vstack((dataset_numpy, f))

# scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
# dataset_numpy = scaler.fit_transform(dataset_numpy)
# test_x= dataset_numpy


# pca = PCA(n_components=12)
# train_x=pca.fit_transform(train_x)

# pca = PCA(n_components=12)
# test_x=pca.fit_transform(test_x)

# #Naive bayes
model = GaussianNB()
model.fit(test_x, test_y)
pickle.dump(model, open('GaussianNB', 'wb'))
# # with open('GaussianNB', 'rb') as pickle_file:
# #     model = pickle.load(pickle_file, encoding='latin1')
# #     print (accuracy_score(model.predict(test_x),test_y))
print (accuracy_score(model.predict(test_x),test_y))

# #KNN
# param_grid = {'n_neighbors':  [3,5,7,9], 'weights':['uniform','distance'], 'p':[1,2,3], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}
# model = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)model = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
model = KNeighborsClassifier()
model.fit(train_x, train_y)
pickle.dump(model, open('KNeighborsClassifier', 'wb'))
# # with open('KNeighborsClassifier', 'rb') as pickle_file:
# #     model = pickle.load(pickle_file, encoding='latin1')
# #     print (model.best_params_)
# #     print (accuracy_score(model.predict(test_x),test_y))
# print (model.best_params_)
print (accuracy_score(model.predict(test_x),test_y))

# #LDA
# param_grid = {'solver':  ['lsqr','eigen'], 'shrinkage':['auto',None]}
# model = GridSearchCV(LinearDiscriminantAnalysis(), param_grid, cv=5)
model = LinearDiscriminantAnalysis()
model.fit(train_x, train_y)
pickle.dump(model, open('LinearDiscriminantAnalysis', 'wb'))
# # model=pickle.load('LinearDiscriminantAnalysis')
# # with open('LinearDiscriminantAnalysis', 'rb') as pickle_file:
# #     model = pickle.load(pickle_file, encoding='latin1')
# #     print (model.best_params_)
# #     print (accuracy_score(model.predict(test_x),test_y))
# print (model.best_params_)
print (accuracy_score(model.predict(test_x),test_y))

#SVM
# param_grid = {'C':  [0.01, 0.1,1,10, 100], 'gamma':[0.08,0.1,10,100], 'kernel':['linear', 'rbf'], 'decision_function_shape':['ovo', 'ovr']}
# model = GridSearchCV(SVC(), param_grid, cv=5)
model=SVC()
model.fit(train_x, train_y)
pickle.dump(model, open('SVC', 'wb'))
# model=pickle.load('SVC')
# with open('SVC', 'rb') as pickle_file:
#     model = pickle.load(pickle_file, encoding='latin1')
#     print (model.best_params_)
#     print (accuracy_score(model.predict(test_x),test_y))
# print (model.best_params_)
print (accuracy_score(model.predict(test_x),test_y))

# #Logistic Regression
# param_grid = {'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
# model = GridSearchCV(LogisticRegression(), param_grid, cv=5)
model = LogisticRegression()
model.fit(train_x, train_y)
pickle.dump(model, open('LogisticRegression', 'wb'))
# # model=pickle.load('LogisticRegression')
# # with open('LogisticRegression', 'rb') as pickle_file:
# #     model = pickle.load(pickle_file, encoding='latin1')
# #     print (model.best_params_)
# #     print (accuracy_score(model.predict(test_x),test_y))
# print (model.best_params_)
print (accuracy_score(model.predict(test_x),test_y))

# #Decision Tree
model = tree.DecisionTreeClassifier()
model.fit(train_x, train_y)
pickle.dump(model, open('DecisionTreeClassifier', 'wb'))
# # model=pickle.load('DecisionTreeClassifier')
# # with open('DecisionTreeClassifier', 'rb') as pickle_file:
# #     model = pickle.load(pickle_file, encoding='latin1')
# #     print (accuracy_score(model.predict(test_x),test_y))
print (accuracy_score(model.predict(test_x),test_y))

