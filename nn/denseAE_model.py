from abc import ABC
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization


class DenseAE(Model, ABC):
    def __init__(self, input_dim, name):
        """
        Autoencoder model composed of:
          - 4 blocks(one dense layer + one batch normalization + one relu activation) as encoder
          - one latent block
          - 4 blocks(one dense layer + one batch normalization + one relu activation) as decoder

        :param input_dim: the size of the input and output layer
        :param name: name of the model
        """
        super().__init__(name)

        self.dense_1 = Dense(512)
        self.batch_1 = BatchNormalization()
        self.activation_1 = Activation("relu")

        self.dense_2 = Dense(384)
        self.batch_2 = BatchNormalization()
        self.activation_2 = Activation("relu")

        self.dense_3 = Dense(256)
        self.batch_3 = BatchNormalization()
        self.activation_3 = Activation("relu")

        self.dense_4 = Dense(128)
        self.batch_4 = BatchNormalization()
        self.activation_4 = Activation("relu")

        self.dense_5 = Dense(8)
        self.batch_5 = BatchNormalization()
        self.activation_5 = Activation("relu")

        self.dense_6 = Dense(128)
        self.batch_6 = BatchNormalization()
        self.activation_6 = Activation("relu")

        self.dense_7 = Dense(256)
        self.batch_7 = BatchNormalization()
        self.activation_7 = Activation("relu")

        self.dense_8 = Dense(384)
        self.batch_8 = BatchNormalization()
        self.activation_8 = Activation("relu")

        self.dense_9 = Dense(512)
        self.batch_9 = BatchNormalization()
        self.activation_9 = Activation("relu")

        self.dense_10 = Dense(input_dim)

    def call(self, inputs):
        """
        Calls the model on new inputs.

        :param inputs: tensor input
        :return: tensor the same size than input
        """
        x = self.dense_1(inputs)
        x = self.batch_1(x)
        x = self.activation_1(x)
        x = self.dense_2(x)
        x = self.batch_2(x)
        x = self.activation_2(x)
        x = self.dense_3(x)
        x = self.batch_3(x)
        x = self.activation_3(x)
        x = self.dense_4(x)
        x = self.batch_4(x)
        x = self.activation_4(x)
        x = self.dense_5(x)
        x = self.batch_5(x)
        x = self.activation_5(x)
        x = self.dense_6(x)
        x = self.batch_6(x)
        x = self.activation_6(x)
        x = self.dense_7(x)
        x = self.batch_7(x)
        x = self.activation_7(x)
        x = self.dense_8(x)
        x = self.batch_8(x)
        x = self.activation_8(x)
        x = self.dense_9(x)
        x = self.batch_9(x)
        x = self.activation_9(x)

        return self.dense_10(x)
