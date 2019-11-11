from keras import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

def SnakeNet(img_shape, class_count):
    lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)

    inputs = Input(img_shape)
    # 512 * 512 * 3

    conv1 = Conv2D(64, 3, activation = lrelu)(inputs)
    conv1 = Conv2D(64, 3, activation = lrelu)(conv1)    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.2)(pool1)
    # 256 * 256 * 64

    conv2 = Conv2D(128, 3, activation = lrelu)(drop1)
    conv2 = Conv2D(128, 3, activation = lrelu)(conv2)    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.2)(pool2)
    # 128 * 128 * 128

    conv3 = Conv2D(256, 3, activation = lrelu)(drop2)
    conv3 = Conv2D(256, 3, activation = lrelu)(conv3)    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.2)(pool3)
    # 64 * 64 * 256

    conv4 = Conv2D(512, 3, activation = lrelu)(drop3)
    conv4 = Conv2D(512, 3, activation = lrelu)(conv4)    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop4 = Dropout(0.2)(pool4)
    # 32 * 32 * 512
    
    conv5 = Conv2D(1024, 3, activation = lrelu)(drop4)
    conv5 = Conv2D(1024, 3, activation = lrelu)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    drop5 = Dropout(0.5)(pool5)
    # 16 * 16 * 1024

    flatten = Flatten()(drop5)
    # 4096

    dense1 = Dense(2048, activation = lrelu)(flatten)
    drop6 = Dropout(0.5)(dense1)

    denseOut = Dense(class_count, activation = 'softmax')(drop6)

    model = Model(input = inputs, output = denseOut)

    return model