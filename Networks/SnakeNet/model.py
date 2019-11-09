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
    drop1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
    # 128 * 128 * 64

    conv2 = Conv2D(128, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    drop2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    # 64 * 64 * 128

    conv3 = Conv2D(256, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    # 32 * 32 * 256

    conv4 = Conv2D(512, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    # 16 * 16 * 512

    conv5 = Conv2D(128, 1, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    drop5 = Dropout(0.2)(pool5)
    # 16 * 16 * 128

    flatten = Flatten()(drop5)
    # 4096

    dense1 = Dense(1024, activation = 'relu')(flatten)
    drop6 = Dropout(0.3)(dense1)

    dense2 = Dense(512, activation = 'relu')(drop6)
    drop7 = Dropout(0.3)(dense2)

    denseOut = Dense(class_count, activation = 'sigmoid')(drop7)

    model = Model(input = inputs, output = denseOut)

    return model