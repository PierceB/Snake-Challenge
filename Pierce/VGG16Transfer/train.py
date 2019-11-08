import numpy as np 
import os
import cv2
import random
import math
import keras
import pickle
from keras import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.applications import VGG16
from dataGenerator import *
from model import SnakeNet

model = SnakeNet((256, 256, 3), 2)
#model.load_weights('snakenet.hdf5')

# conv_base = VGG16(include_top=False, weights='imagenet',input_shape=(256,256,3))
# conv_base.trainable = False
# number_classes = 2

# model = models.Sequential()
# model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(256,activation='relu'))
# model.add(layers.Dense(number_classes,activation='softmax'))


model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['categorical_accuracy'])

train = SnakeDataGenerator(8, source='train')
validation = SnakeDataGenerator(8,  source='validate')

model_checkpoint = ModelCheckpoint('snakenet.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(train, steps_per_epoch=1000, epochs=200, callbacks=[model_checkpoint], max_queue_size=100, workers=8, validation_data=validation, validation_steps=100)

