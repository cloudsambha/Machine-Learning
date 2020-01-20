#!/usr/bin/env python
# coding: utf-8

# In[214]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import tensorflow as tf

DATADIR = "/Users/sameerbhardwaj/Documents/samples/"
IMG_SIZE = 128
CATEGORIES = ["Button", "Input Field"]

for category in CATEGORIES:  # iterate for number of subfolders/classes.
    path = os.path.join(DATADIR,category)  # go to sub folder.
    print(path)
    for img in os.listdir(path):  # iterate over each image.
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        print(img_array.shape)
        print(img_array)
        #plt.imshow(img_array, cmap='gray')
        plt.imshow(new_array, cmap='gray')
        plt.show()  # display images
        break  # to check image
    break   


# In[215]:


data = []

def create_data():
    for category in CATEGORIES: 
        path = os.path.join(DATADIR,category) 
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)): 
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                #plt.imshow(new_array, cmap='gray')
                #plt.show()  # display images
                data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:
                print("Exception", e, os.path.join(path,img))

create_data()                


# In[216]:


#data=np.array(data)
#print(data)


# In[217]:


import random

random.shuffle(data)


# In[218]:


#print(data)


# In[219]:


X = []
y = []

for features,label in data:
    X.append(features)
    y.append(label)

train_pct_index = int(0.9 * len(X))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]

X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

#del(data) #free memory no need for list now.
print(X_train.shape)
print(X_test.shape)


# In[220]:


#convert 3d to 4d for modelling.
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(X_train.shape)

X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(X_test.shape)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255
X_test /= 255


# In[221]:


# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers

input_shape = (128, 128, 1)
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))


# In[222]:


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
history=model.fit(x=X_train,y=y_train, epochs=15)


# In[223]:


model.evaluate(X_test, y_test)


# In[224]:


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model accuracy')
plt.ylabel('Accuracy and loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[225]:


image_index = 6
#print(y_test[image_index])
plt.imshow(X_test[image_index].reshape(128, 128),cmap='Greys')
pred = model.predict(X_test[image_index].reshape(1, 128, 128, 1))
print(pred.argmax())
print(y_test)


# In[ ]:




