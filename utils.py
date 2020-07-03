from keras.losses import BinaryCrossentropy
import tensorflow as tf
import numpy as np
from scipy import ndimage


class GanLosses:
    binary_loss_obj = BinaryCrossentropy(from_logits=True)

    @classmethod
    def discriminator_loss(cls, real, generated):
        real_loss = cls.binary_loss_obj(tf.ones_like(real), real)
        generated_image = cls.binary_loss_obj(tf.zeros_like(generated), generated)
        total_loss = 0.5 * (real_loss + generated_image)
        return total_loss

    @classmethod
    def generator_loss(cls, real_image, generated_image):
        generated_loss = cls.binary_loss_obj(tf.ones_like(generated_image), generated_image)
        # Regulation with MI_LOSS
        mi_loss = cls.mutual_information_2d(real_image.ravel(), generated_image.ravel())
        return mi_loss + generated_loss

    @classmethod
    def mutual_information_2d(cls, x, y, sigma=1, normalized=False):
        EPS = np.finfo(float).eps

        bins = (256, 256)
        jh = np.histogram2d(x, y, bins=bins)[0]
        # smooth the jh with a gaussian filter of given sigma
        ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                output=jh)
        # compute marginal histograms
        jh = jh + EPS
        sh = np.sum(jh)
        jh = jh / sh
        s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
        s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

        # Normalised Mutual Information of:
        # Studholme,  jhill & jhawkes (1998).
        # "A normalized entropy measure of 3-D medical image alignment".
        # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
        if normalized:
            mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                  / np.sum(jh * np.log(jh))) - 1
        else:
            mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
                  - np.sum(s2 * np.log(s2)))

        return mi

    def __call__(self, name):
        pass
