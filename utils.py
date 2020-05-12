from keras.losses import BinaryCrossentropy
import tensorflow as tf


class GanLosses:
    def __init__(self):
        self.binary_loss_obj = BinaryCrossentropy(from_logits=True)

    def discriminator_loss(self, real_image, generated_image):
        real_loss = self.binary_loss_obj(tf.ones_like(real_image), real_image)
        generated_image = self.binary_loss_obj(tf.zeros_like(generated_image), generated_image)

        total_loss = 0.5 * (real_loss + generated_image)

        return total_loss

    def generator_loss(self):
        return

    def __call__(self, name):
        return
