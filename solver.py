import tensorflow as tf
import numpy as np
import datetime
import time
import os
import logging
from dataset import dataset
from model import MriGAN
import os


class SolverLogger:
    def __init__(self, output_log_path):
        self.solver_logger = logging.getLogger(__name__)
        self.solver_logger.setLevel(logging.INFO)
        self.solver_logger.propagate = False

        formatter = logging.Formatter("%(asctime)s-%(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.solver_logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(os.path.join(output_log_path, __name__ + ".log"))
        file_handler.setFormatter(formatter)
        self.solver_logger.addHandler(file_handler)

    def __call__(self, *args, **kwargs):
        return self.solver_logger


class Solver:

    def _set_session(self):
        run_config = tf.compat.v1.ConfigProto()
        run_config.gpu_options.allow_growth = True

        self.sess = tf.compat.v1.Session(config=run_config)

    def _init_logger(self):
        solver_logger = SolverLogger(self.logger_base_path)
        self.logger = solver_logger()

        self.logger.info("is_train = {}".format(self.flags.is_train))
        self.logger.info("dataset name = {}".format(self.flags.dataset))
        self.logger.info("batch size = {}".format(self.flags.batch_size))
        self.logger.info("iter number = {}".format(self.flags.iter_num))
        self.logger.info("mode = {}".format(self.flags.mode))
        self.logger.info("test data path = {}".format(self.flags.test_dataset_path))
        self.logger.info("train data path = {}".format(self.flags.train_dataset_path))

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
        self._init_logger()

    def train(self):
        pass

    def test(self):
        pass

    def load_model(self):
        pass

    def set_needed_folder(self):
        self._set_sample_folder()
        self._set_logger_folder()

    def _set_sample_folder(self):
        self.sample_base_path = "../samples/{}/{}".format(self.flags.dataset, self.cur_time)
        if not os.path.isdir(self.sample_base_path):
            os.makedirs(self.sample_base_path)

    def _set_logger_folder(self):
        self.logger_base_path = "../logging/{}/{}".format(self.flags.dataset, self.cur_time)
        if not os.path.isdir(self.logger_base_path):
            os.makedirs(self.logger_base_path)
