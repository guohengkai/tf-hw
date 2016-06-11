import sys
import math
import tensorflow as tf
from tensorflow.contrib.framework import get_variables_by_name

sys.path.append("..")
from hw2.lr_model import LRModel

class DeepMLPModel(LRModel):
    def __init__(self, hidden_num_list):
        LRModel.__init__(self)
        self.__hidden_num_list = hidden_num_list
        self.__penal_coef = 5e-4

    def _get_hidden_layers(self, last_dim, hidden, is_test):
        for layer_num, hidden_num in enumerate(self.__hidden_num_list):
            layer_name = "hidden_%d" % layer_num
            with tf.variable_scope(layer_name):
                weights = tf.get_variable("weights",
                        initializer=tf.truncated_normal([last_dim, hidden_num],
                            stddev=1.0 / math.sqrt(float(last_dim))))
                biases = tf.get_variable("biases",
                        initializer=tf.zeros([hidden_num]))
                if not is_test:
                    tf.histogram_summary(layer_name + "/weights", weights)
                    tf.histogram_summary(layer_name + "/biases", biases)
                hidden = tf.nn.relu(tf.matmul(hidden, weights) + biases)
            last_dim = hidden_num
        return hidden, last_dim

    def get_model(self, images, is_test):
        hidden, last_dim = self._get_hidden_layers(self.image_pixel, images, is_test)
        if not is_test:
            hidden = tf.nn.dropout(hidden, 0.5)
        return self._get_softmax_model(hidden, last_dim, is_test)

    def _get_l2_loss_by_names(self, loss, names):
        for name in names:
            weights_list = get_variables_by_name(name)
            for weights in weights_list:
                loss += tf.nn.l2_loss(weights) * self.__penal_coef
        return loss

    def get_loss(self, logits, labels):
        loss = super(DeepMLPModel, self).get_loss(logits, labels)
        names = ["weights", "biases"]
        return self._get_l2_loss_by_names(loss, names)

    def get_optimizer(self, loss, learning_rate, batch_size, train_size):
        tf.scalar_summary("train/" + loss.op.name, loss)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        lr = tf.train.exponential_decay(
                learning_rate, global_step * batch_size,
                train_size, 0.95, staircase=True)
        tf.scalar_summary("train/learning_rate", lr)
        optimizer = tf.train.MomentumOptimizer(lr, 0.9)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
