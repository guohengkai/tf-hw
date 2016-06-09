import numpy as np

class Dataset(object):
    def __init__(self, images, labels):
        assert(images.shape[0] == labels.shape[0],
                "images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
        self.__images = images
        self.__labels = labels
        self.__count = images.shape[0]
        self.reset()

    @property
    def images(self):
        return self.__images

    @property
    def labels(self):
        return self.__labels

    @property
    def count(self):
        return self.__count

    def reset(self):
        self.__epochs_completed = 0
        self.__index_in_epoch = 0

    def next_batch(self, batch_size, is_flatten=False):
        start = self.__index_in_epoch
        self.__index_in_epoch += batch_size
        if self.__index_in_epoch > self.count:
            self.__epochs_completed += 1
            perm = np.arange(self.count)
            np.random.shuffle(perm)
            self.__images = self.__images[perm]
            self.__labels = self.__labels[perm]
            start = 0
            self.__index_in_epoch = batch_size
        end = self.__index_in_epoch
        batch_image = self.__images[start:end]
        if is_flatten:
            batch_image = batch_image.reshape((batch_image.shape[0], -1))
        return batch_image, self.__labels[start:end]

import collections
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
