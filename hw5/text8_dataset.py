from __future__ import print_function
import collections
import os
import sys
import tensorflow as tf
import random
import numpy as np
import zipfile
from six.moves.urllib.request import urlretrieve

sys.path.append("..")
from common.common import DATA_DIR

class Text8Dataset(object):
    __url = 'http://mattmahoney.net/dc/'
    __vocabulary_size = 50000
    __data_index = 0

    def __init__(self, data_dir):
        self.__data_dir = data_dir
        filename = self.__maybe_download("text8.zip", 31344016)
        words = self.__read_data(filename)
        print("Data size %d" % len(words))
        self.__data, self.__count, self.__dictionary, self.__reverse_dictionary\
                = self.__build_dataset(words)
        print("Most common words (+UNK)", self.__count[:5])
        print("Sample data", self.__data[:10])
        del words

    @property
    def data(self):
        return self.__data

    @property
    def reverse_dictionary(self):
        return self.__reverse_dictionary

    @property
    def count(self):
        return self.__count

    @property
    def dictionary(self):
        return self.__dictionary

    def reset(self):
        self.__data_index = 0

    def __maybe_download(self, filename, expected_bytes):
        """Download a file if not present, and make sure it's the right size."""
        if not os.path.exists(self.__data_dir + filename):
            _, _ = urlretrieve(self.__url + filename,
                    self.__data_dir + filename)
        statinfo = os.stat(self.__data_dir + filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified %s' % filename)
        else:
            print(statinfo.st_size)
            raise Exception(
                'Failed to verify ' + filename + '. Can you get to it with a browser?')
        return self.__data_dir + filename

    def __read_data(self, filename):
        """Extract the first file enclosed in a zip file as a list of words"""
        with zipfile.ZipFile(filename) as f:
            data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        return data

    def __build_dataset(self, words):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(self.__vocabulary_size - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
        return data, count, dictionary, reverse_dictionary

    def generate_batch(self, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1 # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in xrange(span):
            buffer.append(self.__data[self.__data_index])
            self.__data_index = (self.__data_index + 1) % len(self.__data)
        for i in xrange(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [ skip_window ]
            for j in xrange(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(self.__data[self.__data_index])
            self.__data_index = (self.__data_index + 1) % len(self.__data)
        return batch, labels

if __name__ == "__main__":
    dataset = Text8Dataset(DATA_DIR)
    batch_size = 8
    print("data:", [dataset.reverse_dictionary[di] for di in dataset.data[:batch_size]])
    for num_skips, skip_window in [(2, 1), (4, 2)]:
        dataset.reset()
        batch, labels = dataset.generate_batch(batch_size=batch_size,
                num_skips=num_skips, skip_window=skip_window)
        print("\nwith num_skips = %d and skip_window = %d:" % (num_skips, skip_window))
        print("    batch:", [dataset.reverse_dictionary[bi] for bi in batch])
        print("    labels:", [dataset.reverse_dictionary[li] for li in labels.reshape(batch_size)])
