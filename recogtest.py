import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image
img = load_img('data/test/1.jpg', target_size=(150,150))

model = load_model('imgrec.HDF5')
model.summary()
test_datagen = ImageDataGenerator(rescale=1./255,)
test_generator = test_datagen.flow_from_directory(
        'data/test',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=16,
        class_mode='binary')
img_sources = str(input('Link a picture of a cat or dog:'))
img_path = tf.keras.utils.get_file('Animal', origin=img_sources)
img = load_img(img_path, target_size=(150,150))  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,150,150,3))
predictions = model.predict(x, batch_size = 16, verbose = 0)
if predictions > 0.5:
    print('is Dog')
else:
    print('is Cat')    
 