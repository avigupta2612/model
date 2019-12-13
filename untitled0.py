

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib



model = tf.keras.models.load_model('model.h5')



def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [172,148])
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


p='can.jpg'
img=load_and_preprocess_image(p)
img=np.expand_dims(img,axis=0)
predict=model.predict(img)
label=predict.argmax(axis=-1)
print(label)
