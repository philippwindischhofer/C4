import keras.backend as K
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error
from keras.regularizers import l2
import os
from config import ModelConfig

class C4Model:

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def save(self, folder = 'weights', filename = 'temp.tar'):
        path = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)

        self.model.save_weights(path)

    def load(self, folder = 'weights', filename = 'temp.tar'):
        path = os.path.join(folder, filename)
        self.model.load_weights(path)

    def build(self):
        # build the first layer that directly connects to the input
        in_layer = x = Input((2, self.config.num_rows, self.config.num_cols))
        x = Conv2D(filters = self.config.num_cnn_filters, kernel_size = self.config.size_cnn_filter, padding = "same",
                   data_format = "channels_first", kernel_regularizer = l2(self.config.l2_reg))(x)
        x = BatchNormalization(axis = 1)(x)
        x = Activation("relu")(x)

        # then build the following (hidden) convolutional layers of the network
        for layer in range(self.config.num_conv_layers):
            x = self._build_conv_block(x)

        # now split the network to give separate outputs for the value (i.e. estimated winning probability) and the projected best move on the network-level (i.e. without tree-improvement)
        out_intermediate = x

        # build the value output chain
        x = Conv2D(filters = 1, kernel_size = 1, data_format = "channels_first",
                   kernel_regularizer = l2(self.config.l2_reg))(out_intermediate) # cut the dimensionality down to 1 now
        x = BatchNormalization(axis = 1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        # add a dense layer as the second-to-last one in the network for the value path
        x = Dense(self.config.value_dense_units, kernel_regularizer = l2(self.config.l2_reg), activation = "relu")(x)
        # the last one reduces the dimension down to 1
        out_layer_value = Dense(1, kernel_regularizer = l2(self.config.l2_reg), activation = "tanh", name = "out_layer_value")(x)

        # build the move output chain
        x = Conv2D(filters = 2, kernel_size = 1, data_format = "channels_first", kernel_regularizer = l2(self.config.l2_reg))(out_intermediate)
        x = BatchNormalization(axis = 1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        out_layer_move = Dense(self.config.num_cols, kernel_regularizer = l2(self.config.l2_reg), activation = "softmax", name = "out_layer_move")(x)

        self.model = Model(in_layer, [out_layer_move, out_layer_value], name = "c4_model")
        
    def _build_conv_block(self, x):
        # this is the local (!!) input layer
        in_layer = x

        x = Conv2D(filters = self.config.num_cnn_filters, kernel_size = self.config.size_cnn_filter, padding = "same",
                   data_format = "channels_first", kernel_regularizer = l2(self.config.l2_reg))(x)
        x = BatchNormalization(axis = 1)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters = self.config.num_cnn_filters, kernel_size = self.config.size_cnn_filter, padding = "same",
                   data_format = "channels_first", kernel_regularizer = l2(self.config.l2_reg))(x)
        x = BatchNormalization(axis = 1)(x)
        x = Add()([in_layer, x]) # connect the local input layer directly as well
        x = Activation("relu")(x)

        return x

def prob_cost(y_true, y_pred):
    # use cross entropy here
    return K.sum(-y_true * K.log(y_pred + K.epsilon()), axis = -1)

def value_cost(y_true, y_pred):
    # use normal sum of squares
    return mean_squared_error(y_true, y_pred)
