from keras.applications import VGG16
import sys

sys.path.append('../Data/')

from cfg import *

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
from dataPreprocess import *
from metrics import *
from model import VGG16Base

class LogCallback(Callback):
    def __init__(self, model, generator):
        self.model = model
        self.generator = generator
    def on_epoch_begin(self, batch, logs={}):
        with open('snakenetHistory', 'wb') as file_pi:
            pickle.dump(model.history.history, file_pi)

savePath = 'VGG16.hdf5'
datasetPath = getPath()
dp = DataPreprocessing(datasetRoot=datasetPath)

train_layers = 2
num_classes = getNumClasses()

dp.ClassList(num_classes)
dp.DataSplit()


vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=getImageSize())

for layer in vgg_conv.layers[:-train_layers]:
    layer.trainable = False

for layer in vgg_conv.layers:
    print(layer, layer.trainable)

model = models.Sequential()

model.add(vgg_conv)
model.add(layers.Flatten())
model.add(layers.Dense(1024,activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(num_classes,activation='sigmoid'))

# model = VGG16Base(getImageSize(),num_classes,train_layers=train_layers)
model.summary()
#model.load_weights(savePath)

model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['categorical_accuracy', recall, precision])

train = SnakeDataGenerator(getTrainBatch(), source='train', datasetRoot=datasetPath)
validation = SnakeDataGenerator(getValBatch(),  source='val', datasetRoot=datasetPath)

model_checkpoint = ModelCheckpoint(savePath, monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(train, steps_per_epoch=2000, epochs=getEpochs(), callbacks=[model_checkpoint], max_queue_size=100, workers=getWorkers(), validation_data=validation, validation_steps=200)

