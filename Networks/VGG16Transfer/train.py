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
from model import SnakeNet

savePath = 'VGG16.hdf5'
datasetPath = getPath()
dp = DataPreprocessing(datasetRoot=datasetPath)


num_classes = getNumClasses()

dp.ClassList(num_classes)
dp.DataSplit()

model = VGG16Base(getImageSize(),num_classes)
model.summary()
#model.load_weights(savePath)

model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['categorical_accuracy', recall, precision])

train = SnakeDataGenerator(getTrainBatch(), source='train', datasetRoot=datasetPath)
validation = SnakeDataGenerator(getValBatch(),  source='val', datasetRoot=datasetPath)

model_checkpoint = ModelCheckpoint(savePath, monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(train, steps_per_epoch=2000, epochs=getEpochs(), callbacks=[model_checkpoint], max_queue_size=100, workers=getWorkers(), validation_data=validation, validation_steps=200)

