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
            images = tf.placeholder(tf.float32, shape=(batch_size,
                    self.image_pixel))
        else:
            images = tf.placeholder(tf.float32, shape=(batch_size,
                    self.image_size, self.image_size))
        labels = tf.placeholder(tf.int32, shape=(batch_size))
        return images, labels

    @abstractmethod
    def get_model(self, images):
        pass

    @abstractmethod
    def get_loss(self, logits, labels):
        pass

    @abstractmethod
    def get_optimizer(self, loss, learning_rate):
        pass

    @abstractmethod
    def get_evaluation(self, logits, labels):
        pass
