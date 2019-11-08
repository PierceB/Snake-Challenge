import os
import math

# Poin this to the folder containing all the classes of snakes
path = '/home/pierce/Desktop/CV Work/Snake Project/Snake-Challenge/Data/train/'


# This method generates a list of the classes of snake we have avaliable and how many images we have
def ClassList():
    classListFile = open("dataset/classList.txt", "w")
    imageListFile = open("dataset/imageList.txt", "w")
    for i in range(69):
        if os.path.isdir(path + 'class-' + str(i)):            
            imageList = os.listdir(path + 'class-' + str(i))
            classListFile.write('class-' + str(i) +'|'+ str(len(imageList)) + '\n')
            for imageIterator in range(len(imageList)):
                imageListFile.write('class-' + str(i) + '/' + imageList[imageIterator] + '\n')

            print('Current class : ' + str(i))

    classListFile.close()
    imageListFile.close()

# This method generates over/undersamples from each to create a proportional train/val/test set
def DataSplit():
    trainListFile = open("dataset/trainList.txt", "w")
    valListFile = open("dataset/valList.txt", "w")
    testListFile = open("dataset/testList.txt", "w")

    classListFile = open("dataset/classList.txt", "r")
    imageListFile = open("dataset/imageList.txt", "r")
    classListLines = classListFile.readlines()
    imageListLines = imageListFile.readlines()

    total = 0
    for i in range(len(classListLines)):        
        className = classListLines[i].split('|')[0]
        
        classCountOriginal = int(classListLines[i].split('|')[1])
        classCount = min(classCountOriginal, 1000) # This is shit. We need a better way to sample equally from all classes
        trainCount = math.floor( classCount * 0.7)        
        valCount = math.floor( classCount * 0.2)
        testCount = math.floor( classCount * 0.1)

        for imageIterator in range(total, total+trainCount, 1):
            trainListFile.write(imageListLines[imageIterator])
        
        for imageIterator in range(total+trainCount, total+trainCount+valCount, 1):
            valListFile.write(imageListLines[imageIterator])

        for imageIterator in range(total+trainCount+valCount, total+classCount, 1):
            testListFile.write(imageListLines[imageIterator])

        total += classCountOriginal
    
    classListFile.close()
    imageListFile.close()

    trainListFile.close()
    valListFile.close()
    testListFile.close()

ClassList()
DataSplit()