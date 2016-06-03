import cv2
from sklearn.linear_model import LogisticRegression
import sys

sys.path.append("..")
from common.common import DATA_DIR
from not_mnist_dataset import NotMnistDataset

def load_dataset(display=False):
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

def main():
    train_image, train_label, valid_image, valid_label, test_image, test_label = load_dataset(True)

if __name__ == "__main__":
    main()
