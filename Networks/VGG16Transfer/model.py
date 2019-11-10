from keras import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.applications import VGG16


def VGG16Base(img_shape, class_count):
    # leaky relu. I'm paranoid about dying neurons
    lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)

    concat_axis = 3
    inputs = Input(img_shape)
    # 256 * 256 * 3

    conv_base = VGG16(weights='imagenet', include_top=False,input_shape=img_shape)(inputs)
    conv_base.trainable = False
    flatten = Flatten()(conv_base)
    # 32768

    dense1 = Dense(1024, activation = 'relu')(flatten)
    drop6 = Dropout(0.4)(dense1)

    dense2 = Dense(512, activation = 'relu')(drop6)
    drop7 = Dropout(0.4)(dense2)


    denseOut = Dense(class_count, activation = 'sigmoid')(drop7)

    model = Model(input = inputs, output = denseOut)
    return model
