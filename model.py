import keras.models
import keras.layers
import keras.optimizers
from keras_utils import *


class Discriminator(keras.models.Model):
    def __init__(self, input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.inputs = keras.models.Input(input_size)

    def _load_data(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class Generator(keras.models.Model):
    def __init__(self, input_size, *args, **kwargs):
        # input_shape = (256, 256, 1)
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.inputs = keras.models.Input(input_size)

    def _encoder(self):
        batch1, pool1 = encoder_conv(32, self.inputs)
        batch2, pool2 = encoder_conv(64, pool1)
        batch3, pool3 = encoder_conv(128, pool2)
        batch4, pool4 = encoder_conv(256, pool3)
        batch5, pool5 = encoder_conv(512, pool4)

        up1 = encoder_to_decoder_conv(1024, pool5)

        up2 = decoder_conv(512, up1, batch5)
        up3 = decoder_conv(256, up2, batch4)
        up4 = decoder_conv(128, up3, batch3)
        up5 = decoder_conv(64, up4, batch2)

        generator_final_layer(32)

        pass

    def _decoder(self):
        pass

    def __call__(self, *args, **kwargs):
        pass
