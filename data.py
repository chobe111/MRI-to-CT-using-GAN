import os
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _init_logger(log_path):
    formatter = logging.Formatter("%(asctime)s:%(name):s%(message)s")

    # define file handler
    file_handler = logging.FileHandler(os.path.join(log_path, 'dataset.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # define stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


class Brain01:
    """
    Brain01 is paired CT, MRI Data from Polestar care Corporation
    """

    """
        Brain C2M data was provided by Pole star care

        include 3500 raw data in raw data
        """

    def __init__(self, flags):
        self.flags = flags
        self.name = 'brain01'
        self.image_size = (256, 256, 1)
        self.num_tests = 346

        # tfrecord path
        self.train_tfpath = "../../Data/brain01/tfrecords/train.tfrecords"
        self.test_tfpath = "../../Data/brain01/tfrecords/test.tfrecords"

        logger.info('Initialize {} dataset SUCCESS!'.format(self.flags.dataset))
        logger.info('Img size: {}'.format(self.image_size))

    def __call__(self, is_train='True'):
        if is_train:
            if not os.path.isfile(self.train_tfpath):
                sys.exit(' [!] Train tfrecord file is not found...')
            return self.train_tfpath
        else:
            if not os.path.isfile(self.test_tfpath):
                sys.exit(' [!] Test tfrecord file is not found...')
            return self.test_tfpath


class Brain02:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


def data(data_name, log_path, is_train):
    if is_train:
        _init_logger(log_path)
    if data_name == 'Brain01':
        return Brain01()
    elif data_name == 'Brain02':
        return Brain02()

    pass
