import sys
sys.path.append('../Data/')

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

datasetPath = '../../../train/'
savePath = 'drive/My Drive/Public/SnakeChallenge/snakenet.hdf5'
#datasetPath = 'E:/ML Dataset/Snake/train/'
dp = DataPreprocessing(datasetRoot=datasetPath)

dp.ClassList(45)
dp.DataSplit()

model = SnakeNet((512, 512, 3), 45)
model.summary()
#model.load_weights('snakenet.hdf5')

model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['categorical_accuracy', recall, precision])

train = SnakeDataGenerator(4, source='train', datasetRoot=datasetPath)
validation = SnakeDataGenerator(2,  source='val', datasetRoot=datasetPath)

model_checkpoint = ModelCheckpoint(savePath, monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(train, steps_per_epoch=20, epochs=200, callbacks=[model_checkpoint], max_queue_size=100, workers=8, validation_data=validation, validation_steps=200)

