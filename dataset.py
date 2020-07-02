import os
import sys
import tensorflow as tf


class BrainM2C(object):
    def __init__(self):
        self.name = 'brainM2C'
        self.image_size = (256, 256, 1)
        self.num_tests = 244

        # tfrecords path

        self.train_tfrecords_path = "../tc2mData/train/tfrecords/train.tfrecords"
        self.test_tfrecords_path = "../tc2mData/test/tfrecords/test.tfrecords"

        self.raw_image_train_dataset = tf.data.TFRecordDataset(self.train_tfrecords_path)
        self.raw_image_test_dataset = tf.data.TFRecordDataset(self.test_tfrecords_path)

        print("Load brain tfrecords dataset complete!!")

    def __call__(self, is_train=True):
        if is_train:
            if not os.path.isfile(self.train_tfrecords_path):
                sys.exit('Train tfrecords file is not found....')
            return self.raw_image_train_dataset
        else:
            if not os.path.isfile(self.test_tfrecords_path):
                sys.exit('Test tfrecord file is not found....')

            return self.raw_image_test_dataset


