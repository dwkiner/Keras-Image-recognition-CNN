# Keras-Image-recognition-CNN
A CNN built with the keras API to detect two seperate classes of images. Included models are trained with Cats Vs. Dogs from Kaggle

Requires numpy,tensorflow (and tfds) ,keras,and pillow. pip script coming soon.


Right now it is designed to pull the Cats Vs Dogs dataset and train to tell the difference between cats and dogs but can be trained with any applicable data set with .

run modelgen.py after building and populating the folder structure in the same directory. It will take about 3-4 hours to train.

recogtest.py asks for a link to an image. Link an image of a cat or a dog, and it will return the prediction!
