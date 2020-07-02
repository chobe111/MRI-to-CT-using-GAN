import numpy as np
from PIL import Image
from model import MriGAN
from dataset import BrainM2C

is_train = True

if is_train:
    dataset = BrainM2C()
    gan_model = MriGAN()

    gan_model.train(dataset(True))
