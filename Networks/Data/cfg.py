path = 'E:/ML Dataset/Snake/train/'

train_batch_size = 8
val_batch_size = 2

image_size = (256, 256, 3)
num_classes = 45

epochs = 200
workers = 1

def getPath():
	return path

def getTrainBatch():
	return train_batch_size

def getValBatch():
	return val_batch_size

def getImageSize():
	return image_size

def getNumClasses():
	return num_classes

def getWorkers():
	return workers

def getEpochs():
	return epochs
