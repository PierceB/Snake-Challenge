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
    convc = Conv2D(1, 1, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(drop1)
    # 128 * 128 * 64

    conv2 = Conv2D(128, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(convc)
    conv2 = Conv2D(128, 3, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    drop2 = Dropout(0.2)(conv2)
    flatten = Flatten()(convc2)
    # 32768

    dense1 = Dense(1024, activation = 'relu')(flatten)
    drop2 = Dropout(0.4)(dense1)

    dense2 = Dense(1024, activation = 'relu')(drop2)
    drop3 = Dropout(0.4)(dense2)

    dense3 = Dense(1024, activation = 'relu')(drop3)
    drop4 = Dropout(0.4)(dense3)


    denseOut = Dense(class_count, activation = 'softmax')(drop4)

    model = Model(input = inputs, output = denseOut)

    return model