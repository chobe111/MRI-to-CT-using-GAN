from model import Generator
from keras.utils import plot_model
from keras.models import Model
from keras.applications import VGG16

generator = Generator((256, 256, 1))
plot_model(generator, to_file="model3.png", show_shapes=True)
