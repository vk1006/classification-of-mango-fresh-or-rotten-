#!/usr/bin/env python
# coding: utf-8

# In[42]:


# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve,auc,classification_report
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image
import matplotlib.pyplot as plt 
from PIL import Image 
import seaborn as sns
import pandas as pd
import os
import glob
import numpy as np
import cv2
import h5py
import os
import json
import datetime
import time


# In[43]:


# Deep learning architecture
model = VGG16(weights="imagenet", include_top=False)
for layer in vgg16.layers:
    layer.trainable = False


# In[44]:


train_path = "C:/Users/vikas/Downloads/dataset/train"
train_labels = os.listdir("C:/Users/vikas/Downloads/dataset/train")

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
le.fit([tl for tl in train_labels])

# variables to hold features and labels
features = []
labels   = []

# loop over all the labels in the folder
count = 1
for i, label in enumerate(train_labels):
  cur_path = train_path + "/" + label
  count = 1
  for image_path in glob.glob(cur_path + "/*"):
    img = image.load_img(image_path, target_size=(224,224,3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    flat = feature.flatten()
    features.append(flat)
    labels.append(label)
    print("[INFO] processed - " + str(count))
    count += 1
  print("[INFO] completed label - " + label)

# encode the labels using LabelEncoder
le = LabelEncoder()
le_labels = le.fit_transform(labels)


# In[45]:


from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

features = np.array(features)
labels = np.array(labels)

(x, X, y, Y) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=.3,
                                                                  random_state=9,
                                                                 shuffle = True)


# In[46]:


from sklearn.ensemble import RandomForestClassifier

k=RandomForestClassifier(n_estimators=50,random_state=25)
k.fit(x,y)

RandomForestClassifier(n_estimators=50, random_state=25)

y_predict =k.predict(X)

from sklearn.metrics import confusion_matrix

c=confusion_matrix(Y,y_predict)

sns.heatmap(c,annot=True)
fig = plt.gcf()
fig.set_size_inches(10, 8)

print(classification_report(Y,y_predict))
print("accuracy of  model is",accuracy_score(y_predict,Y)*100)


# In[49]:


from sklearn.tree import DecisionTreeClassifier
#xgboot
model_x=DecisionTreeClassifier()
model_x.fit(x,y)
y_predict = model_x.predict(X)

y_predict = model_x.predict(X)
print(classification_report(Y,y_predict))
print(accuracy_score(Y,y_predict)*100)

c=confusion_matrix(Y,y_predict)
sns.heatmap(c,annot=True)
fig = plt.gcf()
fig.set_size_inches(10, 8)


# In[50]:


from sklearn import svm
model_x=svm.SVC()
model_x.fit(x,y)
y_predict = model_x.predict(X)

y_predict = model_x.predict(X)
print(classification_report(Y,y_predict))
print(accuracy_score(Y,y_predict)*100)

c=confusion_matrix(Y,y_predict)
sns.heatmap(c,annot=True)
fig = plt.gcf()
fig.set_size_inches(10, 8)


# In[51]:


from sklearn.neighbors import KNeighborsClassifier
model_x=KNeighborsClassifier()
model_x.fit(x,y)
y_predict = model_x.predict(X)

y_predict = model_x.predict(X)
print(classification_report(Y,y_predict))
print(accuracy_score(Y,y_predict)*100)

c=confusion_matrix(Y,y_predict)
sns.heatmap(c,annot=True)
fig = plt.gcf()
fig.set_size_inches(10, 8)


# In[ ]:




