from keras import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

def SnakeNet(img_shape, class_count):
    # leaky relu. I'm paranoid about dying neurons
    lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)

    concat_axis = 3
    inputs = Input(img_shape)
    # 256 * 256 * 3

    conv1 = Conv2D(64, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    drop1 = Dropout(0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
    # 128 * 128 * 64

    conv2 = Conv2D(128, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    #merge2 = concatenate([pool1,conv2], axis = 3)
    drop2 = Dropout(0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    # 64 * 64 * 128

    conv3 = Conv2D(256, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #merge3 = concatenate([pool2,conv3], axis = 3)
    drop3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    # 32 * 32 * 256

    conv4 = Conv2D(512, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #merge4 = concatenate([pool3,conv4], axis = 3)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    # 16 * 16 * 512

    conv5 = Conv2D(512, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.2)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(drop5)
    # 8 * 8 * 256

    flatten = Flatten()(pool5)
    # 32768

    dense1 = Dense(2048, activation = 'relu')(flatten)
    # drop6 = Dropout(0.4)(dense1)

    dense2 = Dense(256, activation = 'relu')(dense1)
    #drop7 = Dropout(0.4)(dense2)

    denseOut = Dense(class_count, activation = 'sigmoid')(dense2)

    model = Model(input = inputs, output = denseOut)

    return model