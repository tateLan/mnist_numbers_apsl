from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import sys
import cv2


def get_mnist_data_to_train():
    pass


@ignore_warnings(category=ConvergenceWarning)
def train_classifier(data, labels):
    pass


def predict_image(logisticRegr, im):
    pass


def get_accuracy(classifier, test_input, test_labels):
    pass


if __name__ == '__main__':
    images, labels = get_mnist_data_to_train()
    classifier, test_data, test_labels = train_classifier(images, labels)

    im = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2GRAY).reshape(-1)
    prediction = predict_image(classifier, im)
    accuracy_of_model = get_accuracy(classifier, test_data, test_labels)

    print(f'with accuracy of {accuracy_of_model}, number at image is {prediction}')

