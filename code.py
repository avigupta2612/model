
import tensorflow as tf

import numpy as np
model = tf.keras.models.load_model('/home/chinmay/model1.h5')
    
converter=tf.lite.TFLiteConverter.from_keras_model(model)
model_tflite=converter.convert()
    
interpreter = tf.lite.Interpreter(model_content=model_tflite)
interpreter.allocate_tensors()
    
input_index = interpreter.get_input_details()
output_index = interpreter.get_output_details()
label_names=['Metal cans', 'Plastic crockery', 'Plastic cup', 'crushed plastic bottle', 'plastic bags', 'plastic bottle', 'plastic wrapper', 'syringe', 'tetra pack', 'vegetable peels']
def func(img)
    img=load_and_preprocess_image(img)
    
    img=np.expand_dims(img,axis=0)
    
    interpreter.set_tensor(input_index[0]['index'],img)
    interpreter.invoke()
    
    predictions=interpreter.get_tensor(output_index[0]['index'])
    
    label=predictions.argmax(axis=-1)
    print(label_names[label[0]])


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [172,148])
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


