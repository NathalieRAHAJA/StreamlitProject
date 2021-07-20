import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose


class ConvAE(Model):
    def __init__(self, inputDim, latentDim, name):
        """
        Autoencoder model composed of:
          - 5 blocks(one convolution layer + one batch normalization + one relu activation) as encoder
          - one latent block
          - 5 blocks(one convolution layer + one batch normalization + one relu activation) as decoder

        :param inputDim: the size of the input and output layer
        :param latentDim: the size of the latent layer
        :param name: name of the model
        """
        super().__init__(name)

        self.encoder = tf.keras.Sequential([
            Input(shape=(inputDim[0], inputDim[1], 1)),                     # 32x128x1
            BatchNormalization(),
            ReLU(),

            Conv2D(32, (5, 5), strides=(1, 2), padding='same'),             # 32x64x32
            BatchNormalization(),
            ReLU(),

            Conv2D(64, (5, 5), strides=(1, 2), padding='same'),             # 32x32x64
            BatchNormalization(),
            ReLU(),

            Conv2D(128, (5, 5), strides=(2, 2), padding='same'),            # 16x16x128
            BatchNormalization(),
            ReLU(),

            Conv2D(256, (3, 3), strides=(2, 2), padding='same'),            # 8x8x256
            BatchNormalization(),
            ReLU(),

            Conv2D(512, (3, 3), strides=(2, 2), padding='same'),            # 4x4x512
            BatchNormalization(),
            ReLU(),

            Conv2D(latentDim, (4, 4), strides=(1, 1), padding='valid'),     # 40
            Flatten()])

        self.decoder = tf.keras.Sequential([
            Dense(4 * 4 * 512),
            Reshape((4, 4, 512)),                                           # 4x4x512

            Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same'),   # 8x8x256
            BatchNormalization(),
            ReLU(),

            Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'),   # 16x16x128
            BatchNormalization(),
            ReLU(),

            Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'),    # 32x32x64
            BatchNormalization(),
            ReLU(),

            Conv2DTranspose(32, (5, 5), strides=(1, 2), padding='same'),    # 32x64x32
            BatchNormalization(),
            ReLU(),

            Conv2DTranspose(1, (5, 5), strides=(1, 2), padding='same')])    # 32x128x1

    def call(self, inputs):
        """
        Call the model on new inputs.

        :param inputs: tensor input
        :return: tensor the same size than input
        """
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
