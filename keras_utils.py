import keras.layers
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import concatenate


class tensor:
    def __init__(self):
        pass


def encoder_conv(filter_size: int, input: tensor) -> tuple[tensor, tensor]:
    conv1 = Conv2D(filter_size, (3, 3), padding='same', activation='relu')(input)
    batch1 = BatchNormalization()(conv1)
    drop1 = Dropout(0.5)(batch1)

    conv2 = Conv2D(filter_size, (3, 3), padding='same', activation='relu')(drop1)
    batch2 = BatchNormalization()(conv2)

    conv3 = Conv2D(filter_size, (3, 3), padding='same', activation='relu')(batch2)
    batch3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(batch3)

    return batch3, pool3


def encoder_to_decoder_conv(filter_size: int, input: tensor) -> tensor:
    conv1 = Conv2D(filter_size, (3, 3), padding='same', activation='relu')(input)
    batch1 = BatchNormalization()(conv1)
    drop1 = Dropout(0.5)(batch1)

    conv2 = Conv2D(filter_size, (3, 3), padding='same', activation='relu')(drop1)
    batch2 = BatchNormalization()(conv2)

    up3 = Conv2DTranspose(int(filter_size / 2), (2, 2), padding='same', activation='relu')(batch2)

    return up3


def decoder_conv(filter_size: int, input: tensor, merge_input: tensor) -> tensor:
    merge1 = concatenate([input, merge_input], axis=3)

    conv1 = Conv2D(filter_size, (3, 3), padding='same', activation='relu')(merge1)
    batch1 = BatchNormalization()(conv1)
    drop1 = Dropout(0.5)(batch1)

    conv2 = Conv2D(filter_size, (3, 3), padding='same', activation='relu')(drop1)
    batch2 = BatchNormalization()(conv2)

    up3 = Conv2DTranspose(int(filter_size / 2), (2, 2), padding='same', activation='relu')(batch2)

    return up3


def generator_final_layer(filter_size, input: tensor, merge_input: tensor) -> tensor:
    merge1 = concatenate([input, merge_input], axis=3)

    conv1 = Conv2D(filter_size, (3, 3), padding='same', activation='relu')(merge1)
    batch1 = BatchNormalization()(conv1)

    conv2 = Conv2D(filter_size, (3, 3), padding='same', activation='relu')(batch1)
    batch2 = Conv2D(filter_size, (3, 3), padding='same', activation='relu')(conv2)

    conv3 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(batch2)

    return conv3

    pass
