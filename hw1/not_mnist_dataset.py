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
    __url = 'http://commondatastorage.googleapis.com/books1000/'
    __num_classes = 10
    __image_size = 28  # Pixel width and height.
    __pixel_depth = 255.0  # Number of levels per pixel.
    __train_size = 200000
    __valid_size = 10000
    __test_size = 10000
    __pickle_file = 'notMNIST.pickle'

    def __init__(self, data_dir):
        self.__data_dir = data_dir
        self.__last_percent_reported = None
        np.random.seed(133)
        
        self.__load_datasets()
        print('Training set', self.__train_dataset.shape, self.__train_labels.shape)
        print('Validation set', self.__valid_dataset.shape, self.__valid_labels.shape)
        print('Test set', self.__test_dataset.shape, self.__test_labels.shape)

    def __download_progress_hook(self, count, blockSize, totalSize):
        """A hook to report the progress of a download. This is mostly intended for users with
        slow internet connections. Reports every 1% change in download progress.
        """
        percent = int(count * blockSize * 100 / totalSize)

        if self.__last_percent_reported != percent:
            if percent % 5 == 0:
                sys.stdout.write("%s%%" % percent)
                sys.stdout.flush()
            else:
                sys.stdout.write(".")
                sys.stdout.flush()
          
        self.__last_percent_reported = percent
            
    def __maybe_download(self, filename, expected_bytes, force = False):
        """Download a file if not present, and make sure it's the right size."""
        if force or not os.path.exists(self.__data_dir + filename):
            print('Attempting to download:', filename) 
            filename, _ = urlretrieve(self.__url + filename,
                    self.__data_dir + filename,
                    reporthook=self.__download_progress_hook)
            print('\nDownload Complete!')
        statinfo = os.stat(self.__data_dir + filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified', filename)
        else:
            raise Exception(
              'Failed to verify ' + filename + '. Can you get to it with a browser?')
        return self.__data_dir + filename
    
    def __maybe_extract(self, filename, force=False):
        root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
        if os.path.isdir(root) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping extraction of %s.' % (root, filename))
        else:
            print('Extracting data for %s. This may take a while. Please wait.' % root)
            tar = tarfile.open(filename)
            sys.stdout.flush()
            tar.extractall(self.__data_dir)
            tar.close()
        data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))]
        if len(data_folders) != self.__num_classes:
            raise Exception('Expected %d folders, one per class. Found %d instead.' % (
                self.__num_classes, len(data_folders)))
        print(data_folders)
        return data_folders
        
    def __load_letter(self, folder, min_num_images):
        """Load the data for a single letter label."""
        image_files = os.listdir(folder)
        dataset = np.ndarray(shape=(len(image_files), self.__image_size,
                self.__image_size),
                         dtype=np.float32)
        print(folder)
        num_images = 0
        for image in image_files:
            image_file = os.path.join(folder, image)
            try:
                image_data = (ndimage.imread(image_file).astype(float) - 
                            self.__pixel_depth / 2) / self.__pixel_depth
                if image_data.shape != (self.__image_size, self.__image_size):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                dataset[num_images, :, :] = image_data
                num_images = num_images + 1
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

        dataset = dataset[0:num_images, :, :]
        if num_images < min_num_images:
            raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

        print('Full dataset tensor:', dataset.shape)
        print('Mean:', np.mean(dataset))
        print('Standard deviation:', np.std(dataset))
        return dataset

    def __maybe_pickle(self, data_folders, min_num_images_per_class, force=False):
        dataset_names = []
        for folder in data_folders:
            set_filename = folder + '.pickle'
            dataset_names.append(set_filename)
            if os.path.exists(set_filename) and not force:
                # You may override by setting force=True.
                print('%s already present - Skipping pickling.' % set_filename)
            else:
                print('Pickling %s.' % set_filename)
                dataset = self.__load_letter(folder, min_num_images_per_class)
                try:
                    with open(set_filename, 'wb') as f:
                        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print('Unable to save data to', set_filename, ':', e)

        return dataset_names
        
    def __make_arrays(self, nb_rows, img_size):
        if nb_rows:
            dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
            labels = np.ndarray(nb_rows, dtype=np.int32)
        else:
            dataset, labels = None, None
        return dataset, labels

    def __merge_datasets(self, pickle_files, train_size, valid_size=0):
        valid_dataset, valid_labels = self.__make_arrays(valid_size, self.__image_size)
        train_dataset, train_labels = self.__make_arrays(train_size, self.__image_size)
        vsize_per_class = valid_size // self.__num_classes
        tsize_per_class = train_size // self.__num_classes

        start_v, start_t = 0, 0
        end_v, end_t = vsize_per_class, tsize_per_class
        end_l = vsize_per_class + tsize_per_class
        for label, pickle_file in enumerate(pickle_files):       
            try:
                with open(pickle_file, 'rb') as f:
                    letter_set = pickle.load(f)
                    # let's shuffle the letters to have random validation and training set
                    np.random.shuffle(letter_set)
                    if valid_dataset is not None:
                        valid_letter = letter_set[:vsize_per_class, :, :]
                        valid_dataset[start_v:end_v, :, :] = valid_letter
                        valid_labels[start_v:end_v] = label
                        start_v += vsize_per_class
                        end_v += vsize_per_class
                            
                    train_letter = letter_set[vsize_per_class:end_l, :, :]
                    train_dataset[start_t:end_t, :, :] = train_letter
                    train_labels[start_t:end_t] = label
                    start_t += tsize_per_class
                    end_t += tsize_per_class
            except Exception as e:
                print('Unable to process data from', pickle_file, ':', e)
                raise

        return valid_dataset, valid_labels, train_dataset, train_labels
    
    def __randomize(self, dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation,:,:]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels
        
    def __load_datasets(self):
        if os.path.exists(self.__data_dir + self.__pickle_file):
            with open(self.__data_dir + self.__pickle_file, 'rb') as f:
                save = pickle.load(f)
                self.__train_dataset = save['train_dataset']
                self.__train_labels = save['train_labels']
                self.__valid_dataset = save['valid_dataset']
                self.__valid_labels = save['valid_labels']
                self.__test_dataset = save['test_dataset']
                self.__test_labels = save['test_labels']
                del save  # hint to help gc free up memory
        else:
            train_filename = self.__maybe_download('notMNIST_large.tar.gz', 247336696)
            test_filename = self.__maybe_download('notMNIST_small.tar.gz', 8458043)
            train_folders = self.__maybe_extract(train_filename)
            test_folders = self.__maybe_extract(test_filename)
            train_datasets = self.__maybe_pickle(train_folders, 45000)
            test_datasets = self.__maybe_pickle(test_folders, 1800)
            valid_dataset, valid_labels, train_dataset, train_labels = self.__merge_datasets(
                    train_datasets, self.__train_size, self.__valid_size)
            _, _, test_dataset, test_labels = self.__merge_datasets(test_datasets, self.__test_size)
            
            print("Randomizing the datasets...")
            self.__train_dataset, self.__train_labels = self.__randomize(train_dataset, train_labels)
            self.__test_dataset, self.__test_labels = self.__randomize(test_dataset, test_labels)
            self.__valid_dataset, self.__valid_labels = self.__randomize(valid_dataset, valid_labels)
            
            try:
                print("Saving data to", self.__pickle_file)
                f = open(self.__data_dir + self.__pickle_file, 'wb')
                save = {
                    'train_dataset': self.__train_dataset,
                    'train_labels': self.__train_labels,
                    'valid_dataset': self.__valid_dataset,
                    'valid_labels': self.__valid_labels,
                    'test_dataset': self.__test_dataset,
                    'test_labels': self.__test_labels,
                }
                pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
                f.close()
                del save  # hint to help gc free up memory
            except Exception as e:
                print('Unable to save data to', self.__pickle_file, ':', e)
                raise
            statinfo = os.stat(self.__data_dir + self.__pickle_file)
            print('Compressed pickle size:', statinfo.st_size)

if __name__ == "__main__":
    dataset = NotMnistDataset(DATA_DIR)
