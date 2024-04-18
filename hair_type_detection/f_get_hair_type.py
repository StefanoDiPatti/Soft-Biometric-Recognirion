# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:31:33 2020

@author: HashiramaYaburi
"""

from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = load_model('D:\\prova\\Face_info-master\\Face_info-master\\hair_type_detection\\model-032.model')
labels_dict={0:'afro',1:'corti_uomo',2:'dread',3:'lisci',4:'mossi',5:'ricci',6:'treccia'}


def get_hair_type(img):  
    
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    resized=cv2.resize(gray, (100 , 100))
    normalized= resized/ 255.0
    reshaped=np.reshape(normalized, (1, 100, 100, 1))
    result=model.predict(reshaped)
        
    label=np.argmax(result, axis=1)[0]
    return labels_dict[label]
    