import os
import math

from cfg import getPath


# Poin this to the folder containing all the classes of snakes

class DataPreprocessing:
    def __init__(self, datasetRoot='E:/ML Dataset/Snake/train/'):
        self.path = datasetRoot
        print('init_temp')
        if (os.path.isdir('../Data/dataset/') == False):
            os.mkdir('../Data/dataset/')

    # This method generates a list of the classes of snake we have avaliable and how many images we have
    def ClassList(self, classCount):
        classListFile = open("../Data/dataset/classList.txt", "w")
        imageListFile = open("../Data/dataset/imageList.txt", "w")

        confirmedCount = 0
        for i in range(1626):
            if confirmedCount >= classCount:
                break
            if os.path.isdir(self.path + 'class-' + str(i)):
                confirmedCount += 1
                imageList = os.listdir(self.path + 'class-' + str(i))
                classListFile.write('class-' + str(i) +'|'+ str(len(imageList)) + '\n')
                for imageIterator in range(len(imageList)):
                    imageListFile.write('class-' + str(i) + '/' + imageList[imageIterator] + '\n')

                print('Current class : ' + str(i))

        classListFile.close()
        imageListFile.close()

    # This method generates over/undersamples from each to create a proportional train/val/test set
    def DataSplit(self, trainSplit=0.7, validateSplit=0.2, testSplit=0.1):
        

        classListFile = open("../Data/dataset/classList.txt", "r")
        imageListFile = open("../Data/dataset/imageList.txt", "r")
        classListLines = classListFile.readlines()
        imageListLines = imageListFile.readlines()

        total = 0
        for i in range(len(classListLines)):        
            className = classListLines[i].split('|')[0]

            trainListFile = open("../Data/dataset/"+ className + "_trainList.txt", "w")
            valListFile = open("../Data/dataset/"+ className + "_valList.txt", "w")
            testListFile = open("../Data/dataset/"+ className + "_testList.txt", "w")
            
            classCount = int(classListLines[i].split('|')[1])
            trainCount = math.floor( classCount * trainSplit)
            valCount = math.floor( classCount * validateSplit)
            testCount = math.floor( classCount * testSplit)

            for imageIterator in range(total, total+trainCount, 1):
                trainListFile.write(imageListLines[imageIterator])
            
            for imageIterator in range(total+trainCount, total+trainCount+valCount, 1):
                valListFile.write(imageListLines[imageIterator])

            for imageIterator in range(total+trainCount+valCount, total+classCount, 1):
                testListFile.write(imageListLines[imageIterator])

            total += classCount

            trainListFile.close()
            valListFile.close()
            testListFile.close()

        classListFile.close()
        imageListFile.close()
