from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

sys.path.append("..")
from common.common import DATA_DIR

class NotMnistDataset(object):
    _url = 'http://commondatastorage.googleapis.com/books1000/'

    def __init__(self, data_dir):
        self._data_dir = data_dir
        self._last_percent_reported = None

        print("Downloading tars...")
        train_filename = self._maybe_download('notMNIST_large.tar.gz', 247336696)
        test_filename = self._maybe_download('notMNIST_small.tar.gz', 8458043)


    def _download_progress_hook(self, count, blockSize, totalSize):
        """A hook to report the progress of a download. This is mostly intended for users with
        slow internet connections. Reports every 1% change in download progress.
        """
        percent = int(count * blockSize * 100 / totalSize)

        if self._last_percent_reported != percent:
            if percent % 5 == 0:
                sys.stdout.write("%s%%" % percent)
                sys.stdout.flush()
            else:
                sys.stdout.write(".")
                sys.stdout.flush()
          
        self._last_percent_reported = percent
            
    def _maybe_download(self, filename, expected_bytes, force = False):
        """Download a file if not present, and make sure it's the right size."""
        if force or not os.path.exists(self._data_dir + filename):
            print('Attempting to download:', filename) 
            filename, _ = urlretrieve(self._url + filename, self._data_dir + filename,
                    reporthook=self._download_progress_hook)
            print('\nDownload Complete!')
        statinfo = os.stat(self._data_dir + filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified', filename)
        else:
            raise Exception(
              'Failed to verify ' + filename + '. Can you get to it with a browser?')
        return self._data_dir + filename

if __name__ == "__main__":
    dataset = NotMnistDataset(DATA_DIR)
