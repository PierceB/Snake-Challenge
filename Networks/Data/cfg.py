path = '/home/pierce/Desktop/CV Work/Snake Project/Snake-Challenge/Data/train/'

train_batch_size = 1 
val_batch_size = 1

image_size = (256,256,3)
num_classes = 2

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
