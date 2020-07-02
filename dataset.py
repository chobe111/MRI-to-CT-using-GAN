import os
import sys


class BrainM2C:
    def __init__(self):
        self.name = 'brain'
        self.image_size = (256, 256, 1)
        self.num_tests = 244

        # tfrecords path

        self.train_tfrecords_path = "../tc2mData/train/tfrecords/train.tfrecords"
        self.test_tfrecords_path = "../tc2mData/test/tfrecords/test.tfrecords"

        print("Load brain dataset complete!!")

    def __call__(self, is_train=True):
        if is_train:
            if not os.path.isfile(self.train_tfrecords_path):
                sys.exit('Train tfrecords file is not found....')
            return self.train_tfrecords_path
        else:
            if not os.path.isfile(self.test_tfrecords_path):
                sys.exit('Test tfrecord file is not found....')

            return self.test_tfrecords_path
