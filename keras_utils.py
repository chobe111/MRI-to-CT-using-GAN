import keras.layers
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import Dropout


def encoder_conv1(filter_size: int, input: keras.layers) -> keras.layers:
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.5)(conv1)

    return conv1


def encoder_conv2(filter_size: int, input: keras.layers) -> keras.layers:
    pass


def decoder_conv1():
    pass


def decoder_conv1():
    pass
