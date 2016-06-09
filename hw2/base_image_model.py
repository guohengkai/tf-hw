from __future__ import print_function
from __future__ import division
from abc import abstractmethod
import tensorflow as tf
import sys
sys.path.append("..")
from hw1.not_mnist_dataset import NotMnistDataset

class BaseImageModel(object):
    image_size = NotMnistDataset.image_size()
    image_pixel = NotMnistDataset.image_pixel()
    num_class = NotMnistDataset.num_class()

    def __init__(self, is_flatten):
        self.__is_flatten = is_flatten

    def get_data_input(self, batch_size):
        if self.__is_flatten:
            images_pl = tf.placeholder(tf.float32, shape=(batch_size,
                    self.image_pixel))
        else:
            images_pl = tf.placeholder(tf.float32, shape=(batch_size,
                    self.image_size, self.image_size))
        labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
        return images_pl, labels_pl

    def fill_feed_dict(self, dataset, images_pl, labels_pl, batch_size):
        images_feed, labels_feed = dataset.next_batch(batch_size, self.__is_flatten)
        feed_dict = {
                images_pl: images_feed,
                labels_pl: labels_feed,
        }
        return feed_dict

    def do_eval(self, sess, evaluation, images_pl, labels_pl, dataset, batch_size):
        precision = 0.0
        steps_per_epoch = dataset.count // batch_size
        for step in xrange(steps_per_epoch):
            feed_dict = self.fill_feed_dict(dataset, images_pl, labels_pl, batch_size)
            precision += sess.run(evaluation, feed_dict=feed_dict)
        precision /= steps_per_epoch
        return precision

    def get_evaluation(self, logits, labels):
        correct = tf.nn.in_top_k(logits, labels, 1)
        precision = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.scalar_summary("precision", precision)
        return precision

    @abstractmethod
    def get_model(self, images):
        pass

    @abstractmethod
    def get_loss(self, logits, labels):
        pass

    @abstractmethod
    def get_optimizer(self, loss, learning_rate):
        pass

