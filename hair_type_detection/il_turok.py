"""
Created on Fri Nov 20 15:41:33 2020

@author: HashiramaYaburi
"""

import cv2,os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.constraints import maxnorm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

data_path='dataset2/training'
test_path='dataset2/testing'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]
label_dict=dict(zip(categories,labels)) #empty dictionary
print(label_dict)
print(categories)
print(labels)

data=[]
test=[]
target=[]

img_size=28
for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
        
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)
        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
            resized=cv2.resize(gray,(img_size,img_size))
            #resizing the gray scale into 100x100, since we need a fixed common size for all the images in the dataset
            data.append(resized)
            target.append(label_dict[category])
        except Exception as e:
            print('Exception:',e)
            
img_names=os.listdir(test_path)
        
for img_name in img_names:
    img=cv2.imread(img_path)
    try:
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
        resized=cv2.resize(gray,(img_size,img_size))
        #resizing the gray scale into 100x100, since we need a fixed common size for all the images in the dataset
        test.append(resized)
    except Exception as e:
        print('Exception:',e)
            

data=np.array(data)
test=np.array(test)

target=np.array(target)
new_target=np_utils.to_categorical(target)

X_train, X_val, y_train, y_val = train_test_split(data, new_target, test_size=0.2, random_state=13)

print(X_train.shape)
print(y_train.shape)

X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 1)
X_train = X_train.astype('float32')
X_train /= 255

# Prepare the test images
X_test = test.reshape(test.shape[0], img_size, img_size, 1)
X_test = X_test.astype('float32')
X_test /= 255

# Prepare the validation images
X_val = X_val.reshape(X_val.shape[0], img_size, img_size, 1)
X_val = X_val.astype('float32')
X_val /= 255

input_shape = (img_size, img_size, 1)

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)


cnn3 = Sequential()
cnn3.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
#cnn3.add(BatchNormalization())
cnn3.add(MaxPooling2D((2, 2)))
cnn3.add(Dropout(0.5))

cnn3.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#cnn3.add(BatchNormalization())
cnn3.add(MaxPooling2D(pool_size=(2, 2)))
cnn3.add(Dropout(0.5))

cnn3.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#cnn3.add(BatchNormalization())
cnn3.add(Dropout(0.5))

cnn3.add(Flatten())

cnn3.add(Dense(128, activation='relu'))
#cnn3.add(BatchNormalization())
cnn3.add(Dropout(0.5))
cnn3.add(Dense(7, activation='softmax'))

cnn3.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history4 = cnn3.fit(X_train, y_train,
          batch_size=256,
          epochs=600,
          verbose=1,
          validation_data=(X_val, y_val))

