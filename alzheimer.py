from zipfile import ZipFile
filename = 'alzh.zip'
with ZipFile(filename,'r') as Zip:
  Zip.extractall()
  print("done")
  
import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

TrainingImagePath='/content/archive/Alzheimer_s Dataset/train'
TestingImagePath='/content/archive/Alzheimer_s Dataset/test'
ValidationImagePath='/content/archive/Alzheimer_s Dataset/validation'

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

validation_datagen = ImageDataGenerator(rescale=1./255 )
training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(224,224),
        batch_size=64,
        class_mode='categorical',
        )
test_set = test_datagen.flow_from_directory(
        TestingImagePath,
        target_size=(224,224),
        batch_size=64,
        class_mode='categorical'
        )
validation_set = validation_datagen.flow_from_directory(
        ValidationImagePath,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical'
        )
import zipfile
TrainClasses=training_set.class_indices
ResultMap={0:'MildDemented',1:'ModerateDemented',2:'NonDemented',3:'VeryMildDemented'}
import pickle
with open("/content/Alzheimer_s Dataset.pkl", 'wb') as f:
    pickle.dump(ResultMap, f, pickle.HIGHEST_PROTOCOL)
OutputNeurons= len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf


classifier = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

#last_layer_output = classifier.layers[-1].output
#second_last_layer_output = classifier.layers[-2].output
#third_last_layer_output = classifier.layers[-3].output
#new_model = tf.keras.Model(inputs=classifier.input, outputs=[third_last_layer_output, second_last_layer_output, last_layer_output])
#new_model.summary()
classifier.summary()
classifier.trainable = False

from tensorflow.keras.layers import Input, Convolution2D, Dropout
from tensorflow.keras.models import Model



# First convolutional layer
x = Convolution2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(classifier.output)
x = Dropout(0.5)(x)

x = Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu')(classifier.output)
x = Dropout(0.5)(x)

x = Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(classifier.output)
x = Dropout(0.5)(x)

x = Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu')(classifier.output)
x = Dropout(0.5)(x)

x = Convolution2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu')(classifier.output)
x = Dropout(0.5)(x)


# Second convolutional layer



x= Flatten()(x)


x = Dense(4, activation='softmax')(x)


# Create the functional model
classifier1 = Model(inputs=classifier.input, outputs=x)

classifier1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
model_history = classifier1.fit_generator(
    training_set,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)//32,
    validation_data=validation_set,
    epochs=10,
    verbose=1
)
loss = model_history.history['loss']
accuracy = model_history.history['accuracy']

# Plot the loss
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(loss) + 1), loss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epoch')
plt.grid(True)
plt.show()

# Plot the accuracy
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(accuracy) + 1), accuracy, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs. Epoch')
plt.grid(True)
plt.show()

