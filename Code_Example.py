#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:17:19 2019

@author: extremebeast
"""

## add lambda/2 to setup
## added validation section
## leaving prelu?
## Original & SPeckle are being loaded in a different matrix shape


## This code is a given as an example in oder to retrieve the horse muybridge video
## from the relative speckles.
## The other datasets can be obtained the same way by inserting instead of 'horse' the other
## datasets' keywords: 'Earth_R', 'Earth_G', 'Earth_B', 'Jupyter_R', 'Jupyter_G', 'Jupyter_B',
## 'cat', 'parrot, 'punch'.

import os

from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['axes.linewidth'] = 0

import numpy as np
import cv2

from keras.layers import (Input, Reshape, Dense)
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
#from keras_contrib.losses.dssim import DSSIMObjective


## The program needs to load ComplexNets.py
from ComplexNets import *

import subprocess
import imageio
from IPython.display import HTML

import h5py

class multimode(Dataset):
    def __init__(self,):
        

def load_dataset(file_location,what_to_load):
    hf = h5py.File( file_location , 'r')
    fl = hf[what_to_load]
    Rarray = np.array(fl)
    hf.close()
    return Rarray



image_dim = 120
orig_dim = 92
length_image = 50000


## Insert the DOI file location
file_location = 'D:/Code/trans/Data_1m.h5'

### TRAINING loading
##Random dataset
# Original Images
to_load = 'Training/Original_images/ImageNet'
train_original_image = load_dataset(file_location, to_load)
# Speckle Patterns
to_load = 'Training/Speckle_images/ImageNet'
train_speckle_image = load_dataset(file_location, to_load)


## TESTING loading
## horse dataset
# Original Images
to_load = 'Testing/Original_images/horse'
test_original_horse = load_dataset(file_location, to_load)
# Speckle Patterns
to_load = 'Testing/Speckle_images/horse'
test_speckle_horse = load_dataset(file_location, to_load)




###################### Neural Network Data

## Training data
# Speckle patterns
x_train = train_speckle_image[0:int(length_image/100.*90)]
x_train_ch = real_to_channels_np(x_train.astype('float32'))
# Original Images
y_train = train_original_image[0:int(length_image/100.*90)]
y_train = np.squeeze(y_train.reshape(-1, orig_dim*orig_dim, 1))



## Validation data
# Speckle patterns
x_validation = train_speckle_image[int(length_image/100.*90): length_image]
x_validation_ch = real_to_channels_np(x_validation.astype('float32'))
# Original Images
y_validation = train_original_image[int(length_image/100.*90): length_image]
y_validation = np.squeeze(y_validation.reshape(-1, orig_dim*orig_dim, 1))


## Testing data
# Speckle patterns
x_test = test_speckle_horse
x_test_ch = real_to_channels_np(x_test.astype('float32'))
# Original Images
y_test = test_original_horse
y_test = np.squeeze(y_test.reshape(-1, orig_dim*orig_dim, 1))



##################          MODEL
epochs = 850
lr = 1e-5
batch_size_n = 32
lamb = 0.1

model_name = 'ANN_MMF'


# Create Model
input_img = Input(shape=(image_dim*image_dim, 2))
l = input_img
l = ComplexDense(orig_dim*orig_dim, use_bias=False, kernel_regularizer=regularizers.l2(lamb))(l)
l = Amplitude()(l)
out_layer = l
model = Model(inputs=input_img, outputs=[out_layer])

model.compile(optimizer=SGD(lr=lr), loss='mse')
model.summary()


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=2, min_lr=lr/1e3,
                              verbose=1,)


# Train
model.fit(x_train_ch, 
          y_train,
          validation_data=(x_validation_ch, y_validation),
          epochs=epochs,
          batch_size=32,
          callbacks = [reduce_lr],
          shuffle=True,
         )



## Test Example
pred_test = model.predict(x_test_ch)**2
pred_test = pred_test.reshape(pred_test.shape[0], orig_dim, orig_dim)
