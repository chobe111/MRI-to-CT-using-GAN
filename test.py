from model import Generator, Discriminator
from keras.utils import plot_model
from keras.models import Model
from keras.applications import VGG16

generator = Generator((512, 512, 1))
discriminator = Discriminator((352, 352, 1))

plot_model(generator, to_file="model4.png", show_shapes=True)
# plot_model(discriminator, to_file="dis_model.png", show_shapes=True)
