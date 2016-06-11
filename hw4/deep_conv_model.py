import sys
import math
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.ops.batch_norm_ops import batch_normalize

sys.path.append("..")
from hw3.deep_mlp_model import DeepMLPModel

class DeepConvModel(DeepMLPModel):
    def __init__(self, conv_num_list, hidden_num_list):
        DeepMLPModel.__init__(self, hidden_num_list)
        self.__conv_num_list = conv_num_list
        self.is_flatten = False

    def _get_conv_layers(self, in_channel, conv):
        for stage_num, stage_param in enumerate(self.__conv_num_list):
            stage_name = "stage%d_" % stage_num
            filter_size = stage_param[0]
            layer_count = stage_param[1]
            for layer_num in xrange(layer_count):
                out_channel = stage_param[layer_num + 2]
                with tf.variable_scope(stage_name + "conv%d" % layer_num):
                    weights = tf.get_variable("weights",
                            initializer=tf.truncated_normal([filter_size, filter_size,
                                in_channel, out_channel],
                                stddev=math.sqrt(2.0 / (filter_size ** 2 * in_channel))))
                    biases = tf.get_variable("biases",
                            initializer=tf.zeros([out_channel]))
                    conv = tf.nn.conv2d(conv, weights,
                            strides=[1, 1, 1, 1], padding="SAME")
                    conv = tf.nn.bias_add(conv, biases)
                with tf.variable_scope(stage_name + "bn%d" % layer_num):
                    bn = batch_normalize(conv, convnet=True)
                with tf.variable_scope(stage_name + "relu%d" % layer_num):
                    relu = tf.nn.relu(bn)
                in_channel = out_channel
                conv = relu
            with tf.variable_scope(stage_name + "pool"):
                pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding="SAME")
            conv = pool
        return conv

    def get_model(self, images, is_test):
        conv  = self._get_conv_layers(self.channel, images)
        conv_shape = conv.get_shape().as_list()
        in_channel = conv_shape[1] * conv_shape[2] * conv_shape[3]
        reshape = tf.reshape(conv, [conv_shape[0], in_channel])
        hidden, last_dim = self._get_hidden_layers(in_channel, reshape, is_test)
        if not is_test:
            hidden = tf.nn.dropout(hidden, 0.5)
        return self._get_softmax_model(hidden, last_dim, is_test)
