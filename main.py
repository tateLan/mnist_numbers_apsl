from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import sys
import cv2


def get_mnist_data_to_train():
    print('donwloading mnist data ... ', end='')

    data, labels = fetch_openml('mnist_784', version=1, return_X_y=True)

    print('DONE!')
    return data, labels

@ignore_warnings(category=ConvergenceWarning)
def train_classifier(data, labels):
    print('training ... ', end='')
    train_img, test_img, train_lbl, test_lbl = train_test_split(data, labels, test_size=1 / 7.0, random_state=0)
    logisticRegr = LogisticRegression(solver='lbfgs')
    logisticRegr.fit(train_img, train_lbl)

    print('DONE!')
    return logisticRegr, test_img, test_lbl


def predict_image(logisticRegr, im):
    predictions = logisticRegr.predict(im.reshape(1, -1))
    return predictions[0]


def get_accuracy(classifier, test_input, test_labels):
    score = classifier.score(test_input, test_labels)
    return score


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('too few arguments! pass 1 or 0 to generate test jpg images, and path to image!')
    elif len(sys.argv) == 2:
        try:
            is_images_needed_to_generate = bool(sys.argv[1])
        except Exception as err:
            print('pass 1 or 0 to generate test jpg images, then path to image!')
    elif len(sys.argv) == 3:
        try:
            train_classifier()
        except Exception as err:
            print('pass 1 or 0 to generate test jpg images, and path to image!')


