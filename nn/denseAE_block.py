
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer


class DenseAEBlock(Layer):
    def __init__(self, units=512):
        """
        a block of layers which include:
         - one dense layer
         - one batch normalization
         - one relu activation

        :param units: number of neurone in the dense layer
        """
        super().__init__()

        self.dense = Dense(units)
        self.batch = BatchNormalization()
        self.activation = Activation("relu")

    def build(self, inputs):
        """
        Creates the variables of the layer

        :param inputs: tensor
        :return:
        """
        x = self.dense(inputs)
        x = self.batch(x)

        return self.activation(x)
