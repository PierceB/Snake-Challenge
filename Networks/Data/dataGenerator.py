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
    def __init__(self, batch_size, source = 'train', datasetRoot='E:/ML Dataset/Snake/train/'):
        self.batch_size = batch_size
        self.source = source
        self.path = datasetRoot
        self.sizeX = 512
        self.sizeY = 512

        if (os.path.isdir(self.path) == False):
            print('DIRECTORY ERROR !! ___________________________')

        classListFile = open("../Data/dataset/classList.txt", "r")
        classList = classListFile.readlines()
        classListFile.close()

        self.classCount = len(classList)
        self.classList = []
        self.total = 0

        self.composeAugmentation()

        for i in range(0, len(classList)):
            className = classList[i].split('|')[0]
            classImagePathsFile = open("../Data/dataset/" + className + '_' + source + 'List.txt')
            lines = classImagePathsFile.readlines()
            classImagePathsFile.close()

            self.classList.append(lines)
            self.total += len(lines)            

    def composeAugmentation(self):
        self.augment = albumentations.Compose(
            [
                albumentations.HorizontalFlip()
                #albumentations.Rotate(10, always_apply=True),
                #albumentations.RandomSizedCrop((128, 640), self.sizeY, self.sizeX, 1, always_apply=True),
                #albumentations.GridDistortion(always_apply=False),
                #albumentations.IAAAffine(rotate=2, shear=5, always_apply=False),                
                #albumentations.OpticalDistortion(),
                #albumentations.ElasticTransform(alpha=64, sigma=32, always_apply=True, alpha_affine=0),
                #albumentations.RandomBrightnessContrast(0.2, 0.2, always_apply=True),
                #albumentations.Blur(always_apply=False)
            ]
        )

    def __len__(self):
        return math.ceil(self.total / self.batch_size)

    def __getitem__(self, idx):
        imageResult = np.zeros((self.batch_size, self.sizeY, self.sizeX, 3))
        labelResult = np.zeros((self.batch_size, self.classCount))

        for b in range(0, self.batch_size):            
            randomClass = random.randint(0, self.classCount-1)

            imageResult[b] = self.getImage(randomClass)
            labelResult[b] = self.getLabel(randomClass)

        return (imageResult, labelResult)

    
    def getImage(self, classNumber):
        # Some of the images are corrupted on my disk
        imageList = self.classList[classNumber]

        image = None

        while (image is None):
            imagePath = imageList[random.randint(0, len(imageList)-1)]
            image = cv2.imread(self.path + imagePath.rstrip("\n\r"))

        image = cv2.resize(image, (self.sizeY, self.sizeX))
        res = self.augment(image=image)
        image = res['image']

        image = (image - image.mean()) / (image.std() + 1e-8)

        return image

    def getLabel(self, classNumber):
        label = np.zeros(self.classCount)
        label[classNumber] = 1

        return label