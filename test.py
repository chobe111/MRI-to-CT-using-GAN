from model import Discriminator
from tensorflow.keras.layers import Input
import tensorflow as tf

model = Discriminator((256, 256, 1))

inputs = Input(shape=(256, 256, 1))

outputs = model(inputs)

print(outputs)