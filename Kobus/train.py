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

from dataGenerator import *
from model import SnakeNet

model = SnakeNet((256, 256, 3), 4)
#model.load_weights('snakenet.hdf5')
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['categorical_accuracy'])

train = SnakeDataGenerator(8)
validation = SnakeDataGenerator(8,  validationFlag=True)

model_checkpoint = ModelCheckpoint('snakenet.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(train, steps_per_epoch=500, epochs=200, callbacks=[model_checkpoint], max_queue_size=100, workers=8, validation_data=validation, validation_steps=10)

