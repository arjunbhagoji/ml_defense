import numpy as np
import argparse
import subprocess
from sklearn import svm
from sklearn.decomposition import PCA

from lib.utils.svm_utils import *
from lib.utils.data_utils import *
from lib.utils.dr_utils import *
from lib.attacks.svm_attacks import *

def main():
    # Parse arguments and store in model_dict
    model_dict = svm_model_dict_create()
    DR = model_dict['dim_red']
    rev_flag = None
    # Reduced dimensions used
    rd_list = [784, 331, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

    # Load dataset and create data_dict to store metadata
    print('Loading data...')
    dataset = model_dict['dataset']
    if (dataset == 'MNIST') or (dataset == 'GTSRB'):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(model_dict)
    elif dataset == 'HAR':
        X_train, y_train, X_test, y_test = load_dataset(model_dict)
    # TODO: 2 classes case
    # if model_dict['classes'] == 2:
    #     X_train = X_train

    data_dict = get_data_shape(X_train, X_test)
    n_features = data_dict['no_of_features']

    # Reshape dataset to have dimensions suitable for SVM
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    # X_val_flat= X_val.reshape(-1, no_of_features)

    # Create a new model or load an existing one
    clf = model_creator(model_dict, X_train_flat, y_train)
    model_tester(model_dict, clf, X_test_flat, y_test)
    for i in range(model_dict['classes']):
        print np.linalg.norm(clf.coef_[i])

    no_of_dims=len(rd_list+1)

    gradient_var_list=[]

    test_len = data_dict['test_len']
    no_of_features = data_dict['no_of_features']
    X_test_dr = X_test.reshape((test_len, no_of_features))

    var_array = np.sqrt(np.var(X_test, axis=0))
    var_list = list(var_array)
    coef_list = clf.coef_[0,:]
    
