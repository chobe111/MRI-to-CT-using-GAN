from model import Discriminator

model = Discriminator((256, 256, 1))
print(model.summary())
