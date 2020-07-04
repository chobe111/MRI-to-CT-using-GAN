import tensorflow.keras.models
import tensorflow.keras.layers
import tensorflow.keras.optimizers
from tensorflow.keras.models import Model
from keras_utils import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mae
from tensorflow.keras.layers import Flatten, MaxPooling2D
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from utils import GanLosses
from data_loader import DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec


class MriGAN:
    def __init__(self, sess, flags):

        self.flags = flags
        self.sess = sess
        self.img_shape = (256, 256, 1)
        self.img_size = (256, 256, 1)
        self.discriminator_optimizer = Adam(lr=0.00005, beta_1=0.5)
        self.generator_optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.mutual_loss = GanLosses.mutual_information_2d
        self.loss_obj = tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)
        self.save_iter = 500
        self.batch_size = flags.batch_size
        self.sample_image_output_path = "../tc2mResults"

    def _set_discriminator(self):
        self.discriminator = Discriminator(self.img_shape)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.discriminator_optimizer,
                                   metrics=['accuracy'])

    def _set_generator(self):
        self.generator = Generator(self.img_shape)
        self.generator.compile(loss=[self.mutual_loss],
                               optimizer=self.generator_optimizer,
                               metrics=['accuracy'])

    def _set_combined_model(self):
        self.combined_model = Model(self.input, self.valid)
        self.combined_model.compile(optimizer='adam',
                                    loss='binary_crossentropy',
                                    metrics=['accuracy'])

    def _combined_generator_discriminator(self):
        self.input = keras.Input(shape=self.img_shape)
        self.gen_img = self.generator(self.input)
        self.discriminator.trainable = False
        self.valid = self.discriminator(self.gen_img)

    def _gan_discriminator_net(self):
        self._set_discriminator()
        self._set_generator()
        self._combined_generator_discriminator()
        self._set_combined_model()

    def _discriminator_loss(self, real_image, fake_image):
        real_loss = self.loss_obj(tf.ones_like(real_image), real_image)
        fake_loss = self.loss_obj(tf.zeros_like(fake_image), fake_image)

        total_loss = 0.5 * (real_loss + fake_loss)

        return total_loss

    # def _generator_mi_losses(self, real_image, generated_image):
    #     eps = 1e-8
    #     conditional_entropy = K.mean(- K.sum(K.log(generated_image + eps) * real_image, axis=1))
    #     entropy = K.mean(- K.sum(K.log(real_image + eps) * real_image, axis=1))
    #
    #     return conditional_entropy + entropy

    def _generator_loss(self, generated_image):
        gen_loss = self.loss_obj(tf.ones_like(generated_image), generated_image)

        return gen_loss

    def train(self, dataset, iter_num):
        data_reader = DataLoader(dataset,
                                 name='data',
                                 image_size=self.img_size,
                                 batch_size=self.batch_size,
                                 is_train=self.is_train,
                                 min_queue_examples=1000)

        # return batch size 32

        batch_image_gen = data_reader.feed()

        # batch_image_gen returns x_img, y_img, x_img_ori, y_img_ori, image_name_buffer

        for epoch in range(iter_num):
            img_ct, img_mr, img_ct_ori, img_mr_ori, img_names = batch_image_gen.get_next()

            # img_ct is tensor shape is ? * 256 * 256 * 1 in this case ? is 32

            img_ct_np_arr = self.sess.run(img_ct)

            valid_tensor = tf.ones((self.batch_size, self.img_size[0], self.img_size[1], self.img_size[2]))
            fake_tensor = tf.zeros((self.batch_size, self.img_size[0], self.img_size[1], self.img_size[2]))
            #           train discriminator

            valid_np_arr = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], self.img_size[2]))
            fake_np_arr = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], self.img_size[2]))

            gen_ct = self.generator.predict(img_mr)

            # gen_ct is may be 32 * 256 * 256 * 1 numpy array

            # train on discriminator
            d_loss_real = self.discriminator.train_on_batch(img_ct, valid_tensor)
            d_loss_fake = self.discriminator.train_on_batch(gen_ct, fake_np_arr)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # train on generator

            g_mi_loss = self.generator.train_on_batch(gen_ct, img_ct)

            g_loss = self.combined_model.train_on_batch(gen_ct, valid_tensor)

            if epoch % self.save_iter == 0:
                print("%d epoch d_loss = %.2f g_mi_loss = %f g_loss = %f" % (epoch, d_loss[0], g_mi_loss, g_loss))
                self.sampling(epoch, img_mr, img_ct, gen_ct)

    def sampling_images(self, mri_batch_tensor, ct_batch_tensor, gen_ct_batch_tensor):
        mri_batch_image, ct_batch_image, gen_ct_batch_image = self.sess.run(
            [mri_batch_tensor, ct_batch_tensor, gen_ct_batch_tensor])

        # return batch image type is numpy array
        return [mri_batch_image, ct_batch_image, gen_ct_batch_image]

    def sampling(self, epoch, mri_batch_tensor, ct_batch_tensor, gen_ct_batch_tensor):
        images = self.sampling_images(mri_batch_tensor, ct_batch_tensor, gen_ct_batch_tensor)

        pass

    @staticmethod
    def plots(imgs, iter_time, image_size, save_file):
        scale, margin = 0.02, 0.02
        n_cols, n_rows = len(imgs), imgs[0].shape[0]
        cell_size_h, cell_size_w = imgs[0].shape[1] * scale, imgs[0].shape[2] * scale

        fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        gs.update(wspace=margin, hspace=margin)

        imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]

        # save more bigger image
        for col_index in range(n_cols):
            for row_index in range(n_rows):
                ax = plt.subplot(gs[row_index * n_cols + col_index])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow((imgs[col_index][row_index]).reshape(image_size[0], image_size[1]), cmap='Greys_r')

        plt.savefig(save_file + '/sample_{}.png'.format(str(iter_time).zfill(5)), bbox_inches='tight')
        plt.close(fig)

        pass

    @staticmethod
    def _save_model(mr_img, ct_img, gen_ct_img):

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
        with tf.variable_scope('discriminator') as scope:
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
        with tf.variable_scope('generator') as scope:
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
