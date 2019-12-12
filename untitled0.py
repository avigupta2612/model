#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:32:39 2019

@author: chinmay
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib



model = tf.keras.models.load_model('tflite-env/model.h5')



def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [150,150])
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


p='can (339).jpg'
img=load_and_preprocess_image(p)
img=np.expand_dims(img,axis=0)
predict=model.predict(img)
label=predict.argmax(axis=-1)
print(label)
