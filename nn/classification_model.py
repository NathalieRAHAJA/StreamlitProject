from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization, Flatten
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, MaxPooling2D, Dropout
from tensorflow.keras.regularizers import l1_l2


class Classification(Model):
    def __init__(self, output_dim, name):
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

        self.conv_1 = Conv2D(64, 7, activation="relu", padding='same')
        self.batch_1 = BatchNormalization()

        self.conv_2 = Conv2D(32, 5, activation="relu", padding='same')
        self.conv_3 = Conv2D(6, 5, activation="relu", padding='same')
        self.max_pool_1 = MaxPooling2D(4)
        # self.conv_4 = Conv2D(32, 5, strides=(2, 2), activation="relu", padding='same')
        # self.conv_5 = Conv2D(32, 5, strides=(2, 2), activation="relu", padding='same')

        self.flatten = Flatten()
        self.dense_1 = Dense(2000, activation="relu", activity_regularizer=l1_l2(0.001))
        self.batch_2 = BatchNormalization()
        self.dense_2 = Dense(128, activation="relu", activity_regularizer=l1_l2(0.001))
        self.drop_1 = Dropout(0.2)
        self.dense_3 = Dense(output_dim, activation="softmax")

    def call(self, inputs):
        """
        Call the model on new inputs.

        :param inputs: tensor input
        :return: tensor the same size than input
        """
        x = self.conv_1(inputs)
        x = self.batch_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.max_pool_1(x)
        # x = self.conv_4(x)
        # x = self.conv_5(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.batch_2(x)
        x = self.dense_2(x)
        x = self.drop_1(x)

        return self.dense_3(x)


# from tensorflow.keras import Model
# from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, Flatten, \
#     GlobalAveragePooling1D
# from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, MaxPooling2D, Dropout
# from tensorflow.keras.regularizers import l1_l2
#
#
# class Classification(Model):
#     def __init__(self, output_dim, name):
#         """
#         Autoencoder model composed of:
#           - 5 blocks(one convolution layer + one batch normalization + one relu activation) as encoder
#           - one latent block
#           - 5 blocks(one convolution layer + one batch normalization + one relu activation) as decoder
#
#         :param inputDim: the size of the input and output layer
#         :param latentDim: the size of the latent layer
#         :param name: name of the model
#         """
#         super().__init__(name)
#
#         self.conv_1 = Conv2D(64, 7, activation="relu", padding='same')
#         self.batch_1 = BatchNormalization()
#
#         self.conv_2 = Conv2D(32, 5, activation="relu", padding='same')
#         self.conv_3 = Conv2D(6, 5, activation="relu", padding='same')
#         # self.max_pool_1 = MaxPooling2D(2)
#         self.batch_2 = BatchNormalization()
#         self.conv_4 = Conv2D(1, (128, 1), strides=(1, 1), activation="relu")
#         self.batch_3 = BatchNormalization()
#         self.shap_1 = Reshape((128, 1))
#         self.conv_5 = Conv1D(348, 5, activation="relu", padding='same')
#         self.batch_4 = BatchNormalization()
#         self.conv_6 = Conv1D(348, 3, activation="relu", padding='same')
#         self.batch_5 = BatchNormalization()
#         self.conv_7 = Conv1D(1500, 3, activation="relu", padding='same')
#
#         self.aver_1 = GlobalAveragePooling1D()
#
#         self.flatten = Flatten()
#         self.dense_1 = Dense(3000, activation="relu", activity_regularizer=l1_l2(0.001))
#         self.batch_6 = BatchNormalization()
#         self.dense_2 = Dense(128, activation="relu", activity_regularizer=l1_l2(0.001))
#         self.drop_1 = Dropout(0.2)
#         self.dense_3 = Dense(output_dim, activation="softmax")
#
#     def call(self, inputs):
#         """
#         Call the model on new inputs.
#
#         :param inputs: tensor input
#         :return: tensor the same size than input
#         """
#         x = self.conv_1(inputs)
#         x = self.batch_1(x)
#         x = self.conv_2(x)
#         x = self.conv_3(x)
#         # x = self.max_pool_1(x)
#         x = self.batch_2(x)
#         x = self.conv_4(x)
#         x = self.batch_3(x)
#         x = self.shap_1(x)
#         # x = self.conv_5(x)
#         # x = self.batch_4(x)
#         x = self.conv_6(x)
#         x = self.batch_5(x)
#         x = self.conv_7(x)
#         x = self.aver_1(x)
#         x = self.flatten(x)
#         x = self.dense_1(x)
#         x = self.batch_6(x)
#         x = self.dense_2(x)
#         x = self.drop_1(x)
#
#         return self.dense_3(x)
