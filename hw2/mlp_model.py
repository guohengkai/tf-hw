import sys
import tensorflow as tf

sys.path.append("..")
from hw2.lr_model import LRModel

class MLPModel(LRModel):
    def __init__(self, hidden_num):
        LRModel.__init__(self)
        self.__hidden_num = hidden_num

    def get_model(self, images, is_test=False):
        with tf.variable_scope("hidden"):
            weights = tf.Variable(
                    tf.truncated_normal([self.image_pixel,
                        self.__hidden_num]),
                    name="weights")
            biases = tf.Variable(tf.zeros([self.__hidden_num]),
                    name="biases")
            if not is_test:
                tf.histogram_summary("hidden/weights", weights)
                tf.histogram_summary("hidden/biases", biases)
            hidden = tf.nn.relu(tf.matmul(images, weights) + biases)
        return self._get_softmax_model(hidden, self.__hidden_num, is_test)
