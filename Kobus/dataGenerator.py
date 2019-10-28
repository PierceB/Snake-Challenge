from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import threading
from keras.utils import Sequence
import math
import random
import cv2
import albumentations

class SnakeDataGenerator(Sequence):
    def __init__(self, batch_size, source = 'train'):
        self.batch_size = batch_size
        self.source = source
        self.sizeX = 256
        self.sizeY = 256

        if self.source == 'validate':
            fileListFile = open("dataset/testList.txt", "r")
        elif self.source == 'test':
            fileListFile = open("dataset/valList.txt", "r")
        else:
            fileListFile = open("dataset/trainList.txt", "r")

        self.fileList = fileListFile.readlines()
        self.total = len(self.fileList)
        
        classListFile = open("dataset/classList.txt", "r")
        self.classList = classListFile.readlines()
        self.classCount = len(self.classList)

        self.path = 'E:/ML Dataset/Snake/train/'

        # self.composeAugmentation() We can use this later if we need to

        fileListFile.close()
        classListFile.close()

    def composeAugmentation(self):
        self.augment = albumentations.Compose(
            [
                albumentations.Rotate(10, always_apply=True),
                albumentations.RandomSizedCrop((128, 640), self.sizeY, self.sizeX, 1, always_apply=True),
                albumentations.GridDistortion(always_apply=False),
                albumentations.IAAAffine(rotate=2, shear=5, always_apply=False),
                albumentations.HorizontalFlip(),
                albumentations.OpticalDistortion(),
                albumentations.ElasticTransform(alpha=64, sigma=32, always_apply=True, alpha_affine=0),
                albumentations.RandomBrightnessContrast(0.2, 0.2, always_apply=True),
                albumentations.Blur(always_apply=False)
            ]
        )

    def __len__(self):
        return math.ceil(self.total / self.batch_size)

    def getLabel(self, sampleString):
        classString = sampleString.split('/')[0]
        

        label = np.zeros((self.classCount))

        for i in range(0, self.classCount):
            if (classString == self.classList[i].split('|')[0]):
                label[i] = 1
                return label

        print('Class label not found !')
        exit()

    def __getitem__(self, idx):
        imageResult = np.zeros((self.batch_size, self.sizeY, self.sizeX, 3))
        labelResult = np.zeros((self.batch_size, self.classCount))

        for b in range(0, self.batch_size):
            sample = random.randint(0, self.total-1)
            image = cv2.imread(self.path + self.fileList[sample].rstrip("\n\r"))

            # Some of the images are corrupted on my disk
            while (image is None):
                sample = random.randint(0, self.total-1)
                image = cv2.imread(self.path + self.fileList[sample].rstrip("\n\r"))

            image = cv2.resize(image, (imageResult.shape[1], imageResult.shape[2]))
            image = (image - image.mean()) / (image.std() + 1e-8)
            imageResult[b] = image

            labelResult[b] = self.getLabel(self.fileList[sample])
            #if self.validationFlag == False:
                #print(str(self.validationFlag) + ': ' +str(sample) + ' : ' + str(labelResult[b]))

        return (imageResult, labelResult)