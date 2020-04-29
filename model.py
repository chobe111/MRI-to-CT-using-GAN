import keras.models
import keras.layers
import keras.optimizers
from keras_utils import *


class Discriminator(object):
    def __init__(self, input_size):
        self.input_size = input_size
        self.inputs = keras.models.Input(input_size)

    def _load_data(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class Generator(object):
    def __init__(self, input_size):
        # input_shape = (256, 256, 1)
        self.input_size = input_size
        self.inputs = keras.models.Input(input_size)

    def _encoder(self):
        conv1 = encoder_conv1(32, self.inputs)
        conv1 = keras.layers.MaxPooling2D(conv1)
        conv2 = encoder_conv1(64, conv1)
        


        pass

    def _decoder(self):
        pass

    def __call__(self, *args, **kwargs):
        pass
