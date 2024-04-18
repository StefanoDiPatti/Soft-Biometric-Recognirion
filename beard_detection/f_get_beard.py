# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:47:16 2020

@author: HashiramaYaburi
"""

from keras.models import load_model
import cv2
import numpy as np

model = load_model('D:\\prova\\Face_info-master\\Face_info-master\\beard_detection\\model-005.model')
labels_dict={0:'Beard',1:'Clean Shaved'}



def get_beard(img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized=cv2.resize(gray, (100 , 100))
    normalized= resized/ 255.0
    reshaped=np.reshape(normalized, (1, 100, 100, 1))
    result=model.predict(reshaped)
        
    label=np.argmax(result, axis=1)[0]
    return labels_dict[label]