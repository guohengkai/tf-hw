import sys
import math
import tensorflow as tf

sys.path.append("..")
from hw3.deep_mlp_model import DeepMLPModel

class StackLSTMModel(DeepMLPModel):
    def __init__(self, lstm_num_list, hidden_num_list):
        DeepMLPModel.__init__(self, hidden_num_list)
        self.__lstm_num_list = lstm_num_list

    def get_model(self, images, is_test):
        # adapt the image shape 
        images = tf.reshape(images, [-1, self.image_size, self.image_size])
        images = tf.transpose(images, [2, 0, 1])
        images = tf.reshape(images, [-1, self.image_size])
        images = tf.split(0, self.image_size, images)

        # stack LSTM
        lstm_cells = []
        for num in self.__lstm_num_list:
            lstm_cells.append(tf.nn.rnn_cell.BasicLSTMCell(num, state_is_tuple=True))
        stack_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells, state_is_tuple=True)
        lstm_state, _ = tf.nn.rnn(stack_lstm, images, dtype=tf.float32)
        lstm_state = tf.reshape(lstm_state[-1], [-1, self.__lstm_num_list[-1]])

        # fully connected layer
        hidden, last_dim = self._get_hidden_layers(self.__lstm_num_list[-1], lstm_state, is_test)
        if not is_test:
            hidden = tf.nn.dropout(hidden, 0.5)
        return self._get_softmax_model(hidden, last_dim, is_test)
