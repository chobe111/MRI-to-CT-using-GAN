import numpy as np
from PIL import Image
from model import MriGAN
from dataset import BrainM2C

is_train = True

dataset = BrainM2C()
gan_model = MriGAN()
if is_train:
    gan_model.train(dataset(True))
else:
    gan_model.train(dataset(False))
