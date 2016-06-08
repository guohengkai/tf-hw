import numpy as np

DATA_DIR = "../data/"

def get_sample_idx(total_num, sample_num):
    import random
    idx = range(total_num)
    random.shuffle(idx)
    return idx[:sample_num]

def get_accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))) / predictions.shape[0]
