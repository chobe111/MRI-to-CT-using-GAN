import keras.models
import keras.layers
import keras.optimizers
from keras.models import Model
from keras_utils import *
from keras.optimizers import Adam
from keras.losses import mae


class Discriminator(keras.models.Model):
    """
    This Discriminator model is implementation of MRI_only_brain_radiotherapy Discriminator

    """

    def __init__(self, input_size):
        self.input_size = input_size
        self.inputs = keras.models.Input(input_size)

    def _networks(self):
        pass

    def _load_data(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class Generator(keras.models.Model):
    """
    This Generator model is implementation of MRI_only_brain_radiotherapy Generator
    written by Samaneh Kazemifar

    loss function is MI optimizer is Adam with learning_rate=0.0002, beta_1 = 0.5

    MI(Mutual Information) In probability theory and information theory,
    the mutual information (MI) of two random variables is a measure of the mutual dependence between the two variables.

    """

    def __init__(self, input_size, *args, **kwargs):
        inputs = keras.layers.Input(input_size)
        outputs = self._networks(inputs)

        super().__init__(
            inputs=inputs,
            outputs=outputs
        )
        # input_shape = (512, 512, 1)

        self.input_size = input_size

        # set adam optimizer

    @staticmethod
    def _networks(inputs):
        batch1, pool1 = encoder_conv(32, inputs)
        batch2, pool2 = encoder_conv(64, pool1)
        batch3, pool3 = encoder_conv(128, pool2)
        batch4, pool4 = encoder_conv(256, pool3)
        batch5, pool5 = encoder_conv(512, pool4)

        up1 = encoder_to_decoder_conv(1024, pool5)

        up2 = decoder_conv(512, up1, batch5)
        up3 = decoder_conv(256, up2, batch4)
        up4 = decoder_conv(128, up3, batch3)
        up5 = decoder_conv(64, up4, batch2)

        outputs = generator_final_layer(32, up5, batch1)
        return outputs
