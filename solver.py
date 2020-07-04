import tensorflow as tf
import numpy as np
import datetime
import time
import os
import logging
from dataset import dataset
from model import MriGAN


class Solver:

    def _set_session(self):
        run_config = tf.compat.v1.ConfigProto()
        run_config.gpu_options.allow_growth = True

        self.sess = tf.compat.v1.Session(config=run_config)

    def __init__(self, flags):
        self.flags = flags
        self.is_train = self.flags.is_train
        self.batch_size = self.flags.batch_size
        self._set_session()
        self.model = MriGAN(self.sess, flags)
        self.dataset = dataset(flags)
        self.iter_num = flags.iter_num
        self.cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        self.set_needed_folder()

    def train(self):
        pass

    def test(self):
        pass

    def load_model(self):
        pass

    def set_needed_folder(self):
        self._set_sample_folder()
        self._set_logger_folder()

    @staticmethod
    def make_folder():
        pass

    def _set_sample_folder(self):
        self.sample_base_path = "../samples/{}/{}".format(self.flags.dataset, self.cur_time)

    def _set_logger_folder(self):
        pass
