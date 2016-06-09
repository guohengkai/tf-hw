import sys
import tensorflow as tf

sys.path.append("..")
from base_image_model import BaseImageModel
from hw1.not_mnist_dataset import NotMnistDataset

class LRModel(BaseImageModel):
    def __init__(self):
        BaseImageModel.__init__(self, True)

    def _get_softmax_model(self, last_output, input_dim, is_test):
        with tf.variable_scope("softmax_linear"):
            weights = tf.get_variable("weights",
                    initializer=tf.truncated_normal([input_dim, self.num_class]))
            biases = tf.get_variable("biases",
                    initializer=tf.zeros([self.num_class]))
            if not is_test:
                tf.histogram_summary("softmax/weights", weights)
                tf.histogram_summary("softmax/biases", biases)
            logits = tf.matmul(last_output, weights) + biases
        return logits

    def get_model(self, images, is_test=False):
        return self._get_softmax_model(images, self.image_pixel, is_test)

    def get_loss(self, logits, labels):
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name="xentropy")
        loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")
        return loss

    def get_optimizer(self, loss, learning_rate, batch_size=0, train_size=0):
        tf.scalar_summary("train/" + loss.op.name, loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

