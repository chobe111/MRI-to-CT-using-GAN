import keras.models
import keras.layers
import keras.optimizers
from keras.models import Model
from keras_utils import *
from keras.optimizers import Adam
from keras.losses import mae
from keras.layers import Flatten
import tensorflow as tf


class MriGAN:
    def __init__(self):

        return


class Discriminator(keras.models.Model):
    """
    This Discriminator model is implementation of MRI_only_brain_radiotherapy Discriminator
    written by Samaneh Kazemifar

    input size = 352 * 352 * 1

    output_size = real_number 0 ~ 1

    learning_rate = 0.00005
    beta_1 = 0.5

    filter_size = 2 ->4 ->8 ->16 ->32 -> 64
    6 convolution layers + 5 fully connected layers
    """

    def __init__(self, input_size):
        inputs = keras.layers.Input(input_size)
        outputs = self._networks(inputs)

        self.loss_obj = keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = Adam(lr=0.00005, beta_1=0.5)
        super().__init__(
            inputs=inputs,
            outputs=outputs
        )

    def discriminator_loss(self, real_image, fake_image):
        real_loss = self.loss_obj(tf.ones_like(real_image), real_image)
        fake_loss = self.loss_obj(tf.zeros_like(fake_image), fake_image)

        total_loss = 0.5 * (real_loss + fake_loss)

        return total_loss

    @classmethod
    def _networks(cls, inputs):
        conv1 = discriminator_conv(2, inputs)
        pool1 = MaxPooling2D((2, 2))(conv1)
        conv2 = discriminator_conv(4, pool1)
        conv3 = discriminator_conv(8, conv2)
        conv4 = discriminator_conv(16, conv3)
        conv5 = discriminator_conv(32, conv4)
        conv6 = discriminator_conv(64, conv5)

        flat = Flatten()(conv6)

        dense1 = discriminator_dense(8 * 8 * 64, flat)
        dense2 = discriminator_dense(4068, dense1)
        dense3 = discriminator_dense(2048, dense2)
        dense4 = discriminator_dense(1024, dense3)
        dense5 = discriminator_dense(512, dense4)

        final_layer = discriminator_final_layer(dense5)

        return final_layer

    def _load_data(self):
        return

    def __call__(self, *args, **kwargs):
        return self


class Generator(keras.models.Model):
    """
    input_size = (512,512,1) gray scale

    This Generator model is implementation of MRI_only_brain_radiotherapy Generator
    written by Samaneh Kazemifar

    loss function is MI optimizer is Adam with learning_rate=0.0002, beta_1 = 0.5

    MI(Mutual Information) In probability theory and information theory,
    the mutual information (MI) of two random variables is a measure of the mutual dependence between the two variables.

    """

    def __init__(self, input_size, *args, **kwargs):
        inputs = keras.layers.Input(input_size)
        outputs = self._networks(inputs)
        self.loss_obj = keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = Adam(lr=0.0002, beta_1=0.5)
        super().__init__(
            inputs=inputs,
            outputs=outputs
        )
        # input_shape = (512, 512, 1)
        # set adam optimizer
        self.compile(loss=self._mi_losses, optimizer=Adam(lr=0.0002, beta_1=0.5))

    def generator_loss(self, generated_image):
        generated_loss = self.loss_obj(tf.ones_like(generated_image), generated_image)
        # TODO : Use L1 or L2 loss
        return generated_loss

    @classmethod
    def _networks(cls, inputs):
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

    # TODO : Create Mutual Information loss What is image Distribution
    @classmethod
    def _mi_losses(cls, y_true: Tensor, y_pred: Tensor) -> float:
        """
        :param y_true: y_true is input CT
        :param y_pred: y_pred is synthesis CT from input MRI
        :return: MI Information
        """
        pass

    def __call__(self, *args, **kwargs):
        return self
