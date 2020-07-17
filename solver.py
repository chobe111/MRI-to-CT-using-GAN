import tensorflow as tf
import datetime
import logging
from dataset import dataset
from model import MriGAN
import os
from utils import maybe_mkdirs,inverse_transform
from data_loader import DataLoader
import matplotlib.pyplot as plt
from matplotlib import gridspec


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
        self.logger.info("mode = {}".format(self.flags.mode))
        self.logger.info("test data path = {}".format(self.flags.test_dataset_path))
        self.logger.info("train data path = {}".format(self.flags.train_dataset_path))

    def _set_batch_image_generator(self):
        data_reader = DataLoader(
            self.dataset,
            name='data',
            image_size=(256, 256, 1),
            batch_size=self.flags.batch_size,
            is_train=self.flags.is_train,
            epoch=self.flags.epoch
        )

        self.batch_image_generator = data_reader.feed()

    def __init__(self, flags):
        self.flags = flags
        self.is_train = self.flags.is_train
        self.batch_size = self.flags.batch_size
        self.epochs = self.flags.epoch
        self._set_session()
        self.model = MriGAN(self.sess, flags)

        self.dataset = dataset(flags)
        self._set_batch_image_generator()

        self.cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.set_needed_folder()
        self._init_logger()

        self.sess.run(tf.global_variables_initializer())

    def train(self):
        steps_per_epoch = len(self.dataset) / self.batch_size
        for epoch in range(self.epochs):
            self.logger.info("{} epochs start..".format(epoch))
            images = self.model.train_steps(epoch, int(steps_per_epoch), self.batch_image_generator)
            self.plots(images, epoch, (256, 256, 1), self.sample_base_path)
            self.logger.info("{} epochs end..".format(epoch))

    @staticmethod
    def plots(imgs, iter_time, image_size, save_file):

        scale, margin = 0.02, 0.02
        n_cols, n_rows = len(imgs), imgs[0].shape[0]
        cell_size_h, cell_size_w = imgs[0].shape[1] * scale, imgs[0].shape[2] * scale

        fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        gs.update(wspace=margin, hspace=margin)

        imgs = [inverse_transform(imgs[idx]) for idx in range(len(imgs))]

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

    def test(self):
        pass

    def load_model(self):
        # TODO : MAKE LATER
        pass

    def set_needed_folder(self):

        if self.flags.model_output_path is None:
            self._set_sample_folder()
            self._set_logger_folder()

        elif self.flags.model_output_path is not None:
            pass

    def _set_sample_folder(self):
        self.sample_base_path = "../{}/samples/{}".format(self.flags.dataset, self.cur_time)
        maybe_mkdirs(self.sample_base_path)

    def _set_logger_folder(self):
        self.logger_base_path = "../{}/logging/{}".format(self.flags.dataset, self.cur_time)
        maybe_mkdirs(self.logger_base_path)

    def _set_models_folder(self):
        self.model_base_path = "../{}/models/{}"
        maybe_mkdirs(self.model_base_path)