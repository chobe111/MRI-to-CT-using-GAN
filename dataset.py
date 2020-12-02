import os
import sys
import tensorflow as tf


class BrainM2C(object):

    def _set_train_data_len(self):
        cnt = 0
        for record in tf.python_io.tf_record_iterator(self.train_tf_records_path):
            cnt += 1

        self.train_data_len = cnt
        print(cnt)

    def __init__(self, flags):
        self.name = 'brainM2C'
        self.image_size = (256, 256, 1)

        self.train_tf_records_path = flags.train_dataset_path
        self.test_tf_records_path = flags.test_dataset_path

        self._set_train_data_len()
        self.raw_image_train_dataset = tf.data.TFRecordDataset(self.train_tf_records_path)
        self.raw_image_test_dataset = tf.data.TFRecordDataset(self.test_tf_records_path)

        print("Load brain tf records dataset complete!!")

    def __call__(self, is_train=True):
        if is_train:
            if not os.path.isfile(self.train_tf_records_path):
                sys.exit('Train tfrecords file is not found....')
            return self.raw_image_train_dataset
        else:
            if not os.path.isfile(self.test_tf_records_path):
                sys.exit('Test tfrecord file is not found....')

            return self.raw_image_test_dataset

    def __len__(self):
        return self.train_data_len


def dataset(flags):
    dataset_name = flags.dataset

    if dataset_name == 'brainM2C':
        return BrainM2C(flags)
    # TODO: Later add another dataset
