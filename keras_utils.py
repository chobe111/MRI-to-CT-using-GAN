import keras.layers
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.layers import Activation
from keras.layers import Dense
from tensorflow.python.framework.ops import Tensor


def discriminator_final_layer(input: Tensor) -> Tensor:
    final_layer = Dense(1, activation='sigmoid')(input)
    return final_layer


def discriminator_dense(output_nodes: int, input: Tensor) -> Tensor:
    dense1 = Dense(output_nodes, activation='relu')(input)
    drop1 = Dropout(0.5)(dense1)

    return drop1


def discriminator_conv(filter_size: int, input: Tensor) -> Tensor:
    conv = Conv2D(filter_size, (3, 3), padding='same', strides=(2, 2))(input)
    batch = BatchNormalization()(conv)
    activate = Activation('relu')(batch)

    return conv


def base_conv(filter_size: int, input: Tensor) -> Tensor:
    conv = Conv2D(filter_size, (3, 3), padding='same')(input)
    # conv = Conv2D(filter_size, (3, 3), padding='valid')(input)
    batch = BatchNormalization()(conv)
    activate = Activation('relu')(batch)

    return activate


def dropout_conv(filter_size: int, input: Tensor) -> Tensor:
    conv = base_conv(filter_size, input)
    drop = Dropout(0.5)(conv)

    return drop


def encoder_conv(filter_size: int, input: Tensor) -> (Tensor, Tensor):
    conv1 = dropout_conv(filter_size, input)
    conv2 = base_conv(filter_size, conv1)
    conv3 = Conv2D(filter_size, (3, 3), padding='same')(conv2)
    # conv3 = Conv2D(filter_size, (3, 3), padding='valid')(conv2)
    batch3 = BatchNormalization()(conv3)
    activate3 = Activation('relu')(batch3)
    pool3 = MaxPooling2D((2, 2))(activate3)

    return batch3, pool3


def encoder_to_decoder_conv(filter_size: int, input: Tensor) -> Tensor:
    conv1 = dropout_conv(filter_size, input)
    conv2 = base_conv(filter_size, conv1)

    up3 = Conv2DTranspose(int(filter_size / 2), (2, 2), strides=(2, 2),
                          padding='same', activation='relu')(conv2)

    # up3 = Conv2DTranspose(int(filter_size / 2), (2, 2), strides=(2, 2),
    #                       padding='valid', activation='relu')(conv2)

    return up3


def decoder_conv(filter_size: int, input: Tensor, merge_input: Tensor) -> Tensor:
    merge1 = concatenate([input, merge_input], axis=3)

    conv1 = dropout_conv(filter_size, merge1)
    conv2 = base_conv(filter_size, conv1)

    up3 = Conv2DTranspose(int(filter_size / 2), (2, 2), strides=(2, 2),
                          padding='same', activation='relu')(conv2)
    # up3 = Conv2DTranspose(int(filter_size / 2), (2, 2), strides=(2, 2),
    #                       padding='valid', activation='relu')(conv2)

    return up3


def generator_final_layer(filter_size, input: Tensor, merge_input: Tensor) -> Tensor:
    merge1 = concatenate([input, merge_input], axis=3)

    conv1 = base_conv(filter_size, merge1)
    conv2 = base_conv(filter_size, conv1)
    conv3 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(conv2)
    # conv3 = Conv2D(1, (1, 1), padding='valid', activation='sigmoid')(conv2)

    return conv3
