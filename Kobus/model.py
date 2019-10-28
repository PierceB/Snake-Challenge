from keras import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

def SnakeNet(img_shape, class_count):
    lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)

    concat_axis = 3
    inputs = Input(img_shape)
    # 256
    conv1 = Conv2D(64, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 128
    conv2 = Conv2D(128, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    merge2 = concatenate([pool1,conv2], axis = 3)
    #drop2 = Dropout(0.2)(merge2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(merge2)
    # 64
    conv3 = Conv2D(256, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    merge3 = concatenate([pool2,conv3], axis = 3)
    #drop3 = Dropout(0.2)(merge3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(merge3)
    # 32
    conv4 = Conv2D(512, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    merge4 = concatenate([pool3,conv4], axis = 3)
    #drop4 = Dropout(0.2)(merge4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(merge4)
    # 16
    # Mid
    mid = Conv2D(1024, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    mid = Conv2D(1024, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(mid)
    dropMid = Dropout(0.1)(mid)
    # 16
    flatten = Flatten()(dropMid)
    dense1 = Dense(256, activation = 'relu')(flatten)
    dense1 = Dense(256, activation = 'relu')(dense1)

    denseOut = Dense(class_count, activation = 'sigmoid')(dense1)

    model = Model(input = inputs, output = denseOut)

    return model