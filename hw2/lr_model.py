import sys
import tensorflow as tf

sys.path.append("..")
from base_image_model import BaseImageModel
from hw1.not_mnist_dataset import NotMnistDataset

class LRModel(BaseImageModel):
    def __init__(self):
        BaseImageModel.__init__(self, True)

    def _get_softmax_model(self, last_output, input_dim):
        with tf.name_scope("softmax_linear"):
            weights = tf.Variable(
                    tf.truncated_normal([input_dim, self.num_class]),
                        name="weights")
            biases = tf.Variable(tf.zeros([self.num_class]),
                    name="biases")
            tf.histogram_summary("softmax/weights", weights)
            tf.histogram_summary("softmax/biases", biases)
            logits = tf.matmul(last_output, weights) + biases
        return logits

    def get_model(self, images):
        return self._get_softmax_model(images, self.image_pixel)

    def get_loss(self, logits, labels):
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name="xentropy")
        loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")
        return loss

    def get_optimizer(self, loss, learning_rate):
        tf.scalar_summary("train/" + loss.op.name, loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

