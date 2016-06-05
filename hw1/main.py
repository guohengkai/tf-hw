from __future__ import print_function
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from pylab import plot, show, legend
import cv2

sys.path.append("..")
from common.common import DATA_DIR
from hw1.not_mnist_dataset import NotMnistDataset

def _load_dataset(display=False):
    dataset = NotMnistDataset(DATA_DIR)
    train_image, train_label = dataset.get_train_data()
    valid_image, valid_label = dataset.get_valid_data()
    test_image, test_label = dataset.get_test_data()
    if display:  # display 10 images
        for i in xrange(10):
            image = train_image[i]
            label = str(train_label[i])
            cv2.imshow(label, image)
            ch = cv2.waitKey()
            cv2.destroyAllWindows()
            if ch == ord('q'):
                break

    return train_image, train_label, valid_image, valid_label, test_image, test_label

def _l2_error(image1, image2):
    return np.mean((image1 - image2) ** 2) 

def _get_sample_idx(total_num, sample_num):
    import random
    idx = range(total_num)
    random.shuffle(idx)
    return idx[:sample_num]

def _check_duplicate(train_data, image_list, name_list, threshold=1e-3):
    assert(len(image_list) == len(name_list))
    duplicate_count = [0] * len(image_list)
    sample_num = 10000
    for idx, data in enumerate(image_list):
        print("Searching in", name_list[idx] + "...")
        for i, k1 in enumerate(_get_sample_idx(train_data.shape[0],
                sample_num)):
            if i % 1000 == 0:
                print("  image", i, "(" + str(duplicate_count[idx]) + ")")
            for k2 in xrange(data.shape[0]):
                if _l2_error(train_data[k1], data[k2]) <= threshold:
                    duplicate_count[idx] += 1
                    break
    
    print("-------------------------------------")
    print("Duplicate result")
    print("-------------------------------------")
    for idx in xrange(len(name_list)):
        print("train in %s: %0.2f" % (name_list[idx],
            duplicate_count[idx] / float(sample_num)))

def _test_simple_model(train_image, train_label, test_image, test_label, sample_list):
    train_acc = []
    test_acc = []
    train_image = train_image.reshape((train_image.shape[0], -1))
    test_image = test_image.reshape((test_image.shape[0], -1))
    for sample_num in sample_list:
        sample_idx = _get_sample_idx(train_label.shape[0], sample_num)
        sample_train_image = train_image[sample_idx]
        sample_train_label = train_label[sample_idx]
        lr = LogisticRegression()
        lr.fit(sample_train_image, sample_train_label)
        train_acc.append(lr.score(sample_train_image, sample_train_label))
        test_acc.append(lr.score(test_image, test_label))
        print("-------------------------------------")
        print("Sample number:", sample_num)
        print("-------------------------------------")
        print("Train accuracy:", train_acc[-1])
        print("Test accuracy: ", test_acc[-1])

    plot(train_acc, 'b', test_acc, 'r')
    legend(["train accuracy", "test accuracy"])
    show()

def main():
    train_image, train_label, valid_image, valid_label, test_image, test_label = _load_dataset()
    _check_duplicate(train_image, [valid_image, test_image], ["valid", "test"])
    _test_simple_model(train_image, train_label, test_image, test_label, [50, 100, 1000, 5000])

if __name__ == "__main__":
    main()
