import numpy as np
import tensorflow as tf


class DataLoader:
    def __init__(self, data_path, image_size=(512, 512, 1), name='', batch_size=32, is_train=True):
        self.ori_img_size = image_size
        self.pair_img_size = (image_size.shape[0], image_size.shape[1] * 2, image_size.shape[2])

        pass

    def __call__(self, *args, **kwargs):
        pass
