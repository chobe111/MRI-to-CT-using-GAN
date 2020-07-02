import tensorflow.keras.models
import tensorflow.keras.layers
import tensorflow.keras.optimizers
from tensorflow.keras.models import Model
from keras_utils import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mae
from tensorflow.keras.layers import Flatten
import tensorflow as tf

from tensorflow import keras
import tensorflow.keras.backend as K
from utils import GanLosses
from data_loader import DataLoader


class MriGAN:

    def __init__(self):
        self.img_shape = (256, 256, 1)
        self.discriminator_optimizer = Adam(lr=0.00005, beta_1=0.5)
        self.generator_optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.mutual_loss = GanLosses.mutual_information_2d
        self.loss_obj = tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)

    def _gan_discriminator_net(self):
        self.discriminator = Discriminator(self.img_shape)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.discriminator_optimizer,
                                   metrics=['accuracy'])

        self.generator = Generator(self.img_shape)
        self.generator.compile(loss=[self._generator_mi_losses])

        self.z = keras.Input(shape=self.img_shape)

        # generated_image
        self.img = self.generator(self.z)

        self.discriminator.trainable = False

        self.valid = self.discriminator(self.img)

        self.combined_model = Model(self.z, self.valid)

        self.combined_model.compile(optimizer='adam',
                                    loss='binary_crossentropy',
                                    metrics=['accuracy'])

    def _discriminator_loss(self, real_image, fake_image):
        real_loss = self.loss_obj(tf.ones_like(real_image), real_image)
        fake_loss = self.loss_obj(tf.zeros_like(fake_image), fake_image)

        total_loss = 0.5 * (real_loss + fake_loss)

        return total_loss

    def _generator_mi_losses(self, real_image, generated_image):
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(generated_image + eps) * real_image, axis=1))
        entropy = K.mean(- K.sum(K.log(real_image + eps) * real_image, axis=1))

        return conditional_entropy + entropy

    def _generator_loss(self, generated_image):
        gen_loss = self.loss_obj(tf.ones_like(generated_image), generated_image)

        return gen_loss

    def _train(self):
        data_reader = DataLoader(self.data_path, name='data', image_size=self.img_size,
                                 batch_size=self.flags.batch_size,
                                 is_train=self.flags.is_train)

        pass


class Discriminator(keras.models.Model):

    def __init__(self, input_size):
        inputs = keras.layers.Input(input_size)
        outputs = self._networks(inputs)

        self.optimizer = Adam(lr=0.00005, beta_1=0.5)
        super().__init__(
            inputs=inputs,
            outputs=outputs
        )

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

    def __call__(self, *args, **kwargs):
        return self
