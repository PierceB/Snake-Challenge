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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback

from dataGenerator import *
from dataPreprocess import *
from metrics import *
from model import SnakeNet

# Predicts on batch while training to see how the network is evolving
class LogCallback(Callback):
    def __init__(self, model, generator):
        self.model = model
        self.generator = generator
    def on_epoch_begin(self, batch, logs={}):
        with open('snakenetHistory', 'wb') as file_pi:
            pickle.dump(model.history.history, file_pi)



savePath = 'snakenet.hdf5'
datasetPath = getPath()
dp = DataPreprocessing(datasetRoot=datasetPath)


dp.ClassList(getNumClasses())
dp.DataSplit()

model = SnakeNet(getImageSize(), getNumClasses())
model.summary()
#model.load_weights(savePath)

model.compile(optimizer = Adam(lr = 1e-3), loss = 'categorical_crossentropy', metrics = ['categorical_accuracy', recall, precision])

train = SnakeDataGenerator(getTrainBatch(), source='train', datasetRoot=datasetPath)
validation = SnakeDataGenerator(getValBatch(),  source='val', datasetRoot=datasetPath)
logCallback = LogCallback(model, validation)

model_checkpoint = ModelCheckpoint(savePath, monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(train, epochs=getEpochs(), callbacks=[model_checkpoint, logCallback], max_queue_size=100, workers=getWorkers(), validation_data=validation, validation_steps=200)

