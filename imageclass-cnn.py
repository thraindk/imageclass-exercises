# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 11:38:04 2022

@author: cedri

This program classifies images using keras and the CIFAR10 dataset

"""

# based on https://www.youtube.com/watch?v=iGWbqhdjf2s

# libraries

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight") # Pyplot style


# load data

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data() # 2 tupels train data /  test data 

# LOOK AT DATASET

# look at datatype of variables

print(type(x_train)) # all numpy.ndarray
print(type(y_train))
print(type(x_test))
print(type(y_test))


# get shape of arrays

print("x_train shape:", x_train.shape) # 4d array = 50k rows of 32x32 images with depth 3 (rgb!)
print("y_train shape:", y_train.shape) # 2d array = 50k rows + 1 col
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# take a look at the first image as array

print(x_train[0])


# show first image as picture

img = plt.imshow(x_train[0]) # frog

# get image label

print("the image label is:", y_train[0]) # 6 = classifaction frog (categorical)

# mapping categories

classification = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print("image class is:", classification[y_train[0][0]]) # only value [0] of 2d array entry

# PREPARE DATA FOR CNN

# convert the class labels into a set of 10 numbers to input into the neural network (categorical boolean is type)

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Print the new labels

print(y_train_one_hot)

# print the new label of current image

print("The one hot label is:", y_train_one_hot[0]) # frog out of classification

# normalize the pixels to be values between 0 ... 1

x_train = x_train / 255
x_test = x_test / 255

print(x_train[0])

# create MODELS architecture

model = Sequential()

# Add the first layer (convolution layer) -> extract (filter) features from the input image, 
# creates 32 5x5 relu feature maps

model.add( Conv2D(32, (5,5), activation="relu", input_shape=(32, 32, 3)) )

# 2nd layer, create pooling layer to get max element from the feature maps

model.add(MaxPooling2D(pool_size = (2,2))) # with 2x2 pixel filter

# Add another convolution layer
model.add( Conv2D(32, (5,5), activation="relu") )

# Add another pooling layer

model.add(MaxPooling2D(pool_size = (2,2)))

# Add flattening layer

model.add(Flatten())

# Add a layer with 1000 neurons

model.add( Dense(1000, activation="relu") )

# Add a drop out layer

model.add(Dropout(0.5))

# Add a another 500 neurons

model.add( Dense(500, activation="relu") )

# Add a drop out layer

model.add(Dropout(0.5))


# Add a another 250 neurons

model.add( Dense(250, activation="relu") )


# Add a another 10 neurons (classifications!)

model.add( Dense(10, activation="softmax") )

# compile the model

model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

# Train the model

hist = model.fit(x_train, y_train_one_hot,
                 batch_size = 256,
                 epochs = 10,
                 validation_split = 0.2)

# Evaluate model using test data set

model.evaluate(x_test, y_test_one_hot) [1] # 0.66 accurate

# Visualize the models accuracy

plt.clf()

plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend("Training", "Validation", loc="upper left")
plt.show()

plt.clf()

# Visualize models loss

plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend("Training", "Validation", loc="upper right")
plt.show()

plt.clf()

# Test the model with an example

new_image = plt.imread("cat.jpg") # licence-free pixabay image

from skimage.transform import resize

resized_image = resize(new_image, (32, 32, 3))

img = plt.imshow(resized_image)

# Get the models predictions

predictions = model.predict(np.array([resized_image]))

# show the predictions
print(predictions) # highest determines prediction

# sort predictions from least to greatest

list_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
x = predictions

for i in range(10):
    for j in range(10):
        if x[0][list_index[i]] > x[0][list_index[j]]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp
            
# show the sorted LABELS in order

print(list_index)

# print the first 5 most likely classes with prediction
print("prediction for custom image:")

for i in range(5):
    print(classification[list_index[i]], ":", round(predictions[0][list_index[i]] * 100, 2), "%")

