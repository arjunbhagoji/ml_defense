from __future__ import print_function
import numpy as np
from sklearn import svm

from utils.svm_utils import load_mnist

def main(dataset):


    # Load specified dataset
    if dataset=='mnist':
        X_train, y_train, X_test, y_test=load_mnist('data/')

    # Train desired SVM
    


main('mnist')
