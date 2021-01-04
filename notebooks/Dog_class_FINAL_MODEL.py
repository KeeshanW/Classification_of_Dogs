# Load all imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import itertools
#import imutils
import scipy.io
import shutil, os
import zipfile
import seaborn as sns
from tensorflow.keras.preprocessing import image
import datetime
import pickle

# define file paths for Train and Validate Image data sets
train_path = 'Train_images'
valid_path = 'Validate_images'

# get classes of images based on folder names in Train data set
Img_class = [f for f in os.listdir(train_path) if not f.startswith('.')]

# remove "Other" class for training
Img_class.remove('Other')

# Use first 10 classes for training
Img_class_set = Img_class[:10]
Img_class_set = [w.upper() for w in Img_class_set]

# Import images files and image classes for trainig and vaidating sets
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224, 224), classes=Img_class_set, batch_size=6)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), classes=Img_class_set, batch_size=6)

# Import VGG-16 model
vgg16_model = tf.keras.applications.vgg16.VGG16()

VGG16_Class_Model = Sequential()

# Remove last layer from VGG-16 model and add in new dense layer with softmaxx activation for training
for layer in vgg16_model.layers[:-1]:
  VGG16_Class_Model.add(layer)
for layer in VGG16_Class_Model.layers[:-1]:
  layer.trainable = False
num_classes = 10
VGG16_Class_Model.add(Dense(num_classes, activation='softmax'))

opt = tf.keras.optimizers.SGD(learning_rate=0.0001)

# Compile, train and save VGG-16 model
VGG16_Class_Model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

log_dir = "VVG-16_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
VGG16_Class_Model.fit(train_batches, validation_data=valid_batches, epochs=100, verbose=1, callbacks=[tensorboard_callback])
VGG16_Class_Model.save('Dir_Classifier_VGG16_model-R2.h5')

# Import VGG-19 model
vgg19_model = tf.keras.applications.vgg19.VGG19()

VVG19_Class_Model = Sequential()

# Remove last layer from VGG-19 model and add in new dense layer with softmaxx activation for training
for layer in vgg19_model.layers[:-1]:
    VVG19_Class_Model.add(layer)
for layer in VVG19_Class_Model.layers[:-1]:
    layer.trainable = False
num_classes = 10
VVG19_Class_Model.add(Dense(num_classes, activation='softmax'))

# Compile, train and save VGG-19 model
VVG19_Class_Model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

log_dir = "VVG-19_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
VVG19_Class_Model.fit(train_batches, validation_data=valid_batches, epochs=100, verbose=1, callbacks=[tensorboard_callback])
VVG19_Class_Model.save('Dir_Classifier_VGG19_model_R2.h5')

# define file paths for Train and Validate Image data sets
test_path = 'Test_images'

# Import images files and image classes for trainig and vaidating sets
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224), classes=Img_class_set, batch_size=331)

# Load trained model
VVG16_model = tf.keras.models.load_model('Dir_Classifier_VGG16_model-R2.h5')
VVG19_model = tf.keras.models.load_model('Dir_Classifier_VGG19_model_R2.h5')
test_imgs, test_labels = next(test_batches)

# Get prediction probobalities for each model
VVG16_pred = VVG16_model.predict(test_imgs)
VVG19_pred = VVG19_model.predict(test_imgs)

# Get label of each prediction for each model
y_test = []
y_pred_VVG16 = []
y_pred_VVG19 = []

for i in range(len(test_labels)):
    y_test.append(Img_class_set[np.argmax(test_labels[i])])
    y_pred_VVG16.append(Img_class_set[np.argmax(VVG16_pred[i])])
    y_pred_VVG19.append(Img_class_set[np.argmax(VVG19_pred[i])])

# Save all prediction parameters
save('y_pred_VVG19.npy',y_pred_VVG19)
save('y_pred_VVG16.npy',y_pred_VVG16)
save('y_test.npy',y_test)
save('test_imgs.npy',test_imgs)
save('test_labels.npy',test_labels)
save('VVG16_pred.npy', VVG16_pred)
save('VVG19_pred.npy', VVG19_pred)