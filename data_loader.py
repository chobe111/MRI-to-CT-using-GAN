import numpy as np
import tensorflow as tf
import math
import time


class DataLoader:
    def __init__(self, dataset, image_size=(256, 256, 1),
                 batch_size=32,
                 is_train=True,
                 epoch=1000,
                 name='data'):

        self.ori_img_size = image_size
        self.pair_img_size = (self.ori_img_size[0], self.ori_img_size[1] * 2, self.ori_img_size[2])
        self.dataset = dataset
        self.resize_factor = 1.05
        self.rotate_angle = 5.
        self.is_train = is_train
        self.name = name
        self.batch_size = batch_size
        self.epoch = epoch
        self.min_queue_examples = len(dataset) * 2

        self.image_features = {
            'image/file_name': tf.io.FixedLenFeature([], tf.string),
            'image/encoded_image': tf.io.FixedLenFeature([], tf.string)
        }

    def feed(self):
        with tf.name_scope(self.name):
            parsed_image_dataset = self.dataset().map(self._parse_image_function)
            if self.is_train:
                # return iterator object
                train_image_batch_tensor_iterator = parsed_image_dataset \
                    .shuffle(self.min_queue_examples).repeat(self.epoch).batch(self.batch_size).make_one_shot_iterator()
                return train_image_batch_tensor_iterator
            else:
                # return iterator object
                test_image_batch_tensor_iterator = parsed_image_dataset \
                    .batch(self.batch_size).repeat(1).make_one_shot_iterator()

                return test_image_batch_tensor_iterator

    def _parse_image_function(self, serialized):
        features = tf.io.parse_single_example(serialized, self.image_features)
        image_buffer = features['image/encoded_image']
        image_name_buffer = features['image/file_name']

        image = tf.image.decode_jpeg(image_buffer, channels=self.ori_img_size[2])

        x_img, y_img, x_img_ori, y_img_ori = self._preprocess(image, is_train=self.is_train)

        return x_img, y_img, x_img_ori, y_img_ori, image_name_buffer

    def __call__(self, *args, **kwargs):
        pass

    def _preprocess(self, img, is_train):
        # Resize to 2D and split to left and right image
        img = tf.image.resize(img, size=(self.pair_img_size[0], self.pair_img_size[1]))
        x_img_ori, y_img_ori = tf.split(img, [self.ori_img_size[1], self.ori_img_size[1]], axis=1)
        # x = CT y = MR
        x_img, y_img = x_img_ori, y_img_ori
        # Data augmentation
        if is_train:
            random_seed = int(round(time.time()))

            # Make image bigger
            x_img = tf.image.resize(x_img_ori, size=(int(self.resize_factor * self.ori_img_size[0]),
                                                     int(self.resize_factor * self.ori_img_size[1])))

            y_img = tf.image.resize(y_img_ori, size=(int(self.resize_factor * self.ori_img_size[0]),
                                                     int(self.resize_factor * self.ori_img_size[1])))

            # Random crop
            x_img = tf.image.random_crop(x_img, size=self.ori_img_size, seed=random_seed)
            y_img = tf.image.random_crop(y_img, size=self.ori_img_size, seed=random_seed)

            # Random flip
            x_img = tf.image.random_flip_left_right(x_img, seed=random_seed)
            y_img = tf.image.random_flip_left_right(y_img, seed=random_seed)

            # Random rotate
            radian_min = -self.rotate_angle * math.pi / 180.
            radian_max = self.rotate_angle * math.pi / 180.
            random_angle = tf.random.uniform(shape=[1], minval=radian_min, maxval=radian_max, seed=random_seed)
            x_img = tf.contrib.image.rotate(x_img, angles=random_angle, interpolation='NEAREST')
            y_img = tf.contrib.image.rotate(y_img, angles=random_angle, interpolation='NEAREST')

        # Do resize and zero-centering
        x_img = self.basic_preprocess(x_img)
        y_img = self.basic_preprocess(y_img)
        x_img_ori = self.basic_preprocess(x_img_ori)
        y_img_ori = self.basic_preprocess(y_img_ori)

        return x_img, y_img, x_img_ori, y_img_ori

    def basic_preprocess(self, img):
        img = tf.image.resize(img, size=(self.ori_img_size[0], self.ori_img_size[1]))
        # zero - centering input image
        img = (tf.image.convert_image_dtype(img, dtype=tf.float32) / 127.5) - 1.0
        img.set_shape(self.ori_img_size)

        return img
