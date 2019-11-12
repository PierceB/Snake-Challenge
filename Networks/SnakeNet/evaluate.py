
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
from model import SnakeNet

model = SnakeNet((256, 256, 3), 20)
#model.load_weights('snakenet.hdf5')
model.summary()
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['categorical_accuracy'])

test = SnakeDataGenerator(100, source='test')

image, label = test.__getitem__(0)

results = model.predict_on_batch(image)

count = 0
correct = 0
for i in range(len(results)):
    true = np.argmax(label[i])
    pred = np.argmax(results[i])
    print(str(i) + ' > ' +str(true) + ':' + str(pred))

    if (true == pred):
        correct += 1
    count += 1

print(correct / count)