import numpy as np
import argparse
import subprocess
from sklearn import svm
from sklearn.decomposition import PCA

from lib.utils.svm_utils import *
from lib.utils.data_utils import load_dataset, get_data_shape
from lib.utils.dr_utils import *
from lib.utils.plot_utils import *
from lib.attacks.svm_attacks import *

def main():
    # Parse arguments and store in model_dict
    model_dict = svm_model_dict_create()
    DR = model_dict['dim_red']
    rev_flag = model_dict['rev']
    strat_flag = 1
    adv = None

    # Load dataset and create data_dict to store metadata
    print('Loading data...')
    dataset = model_dict['dataset']
    if (dataset == 'MNIST') or (dataset == 'GTSRB'):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
            model_dict)
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
    # Center dataset with mean of training set
    mean = np.mean(X_train_flat, axis=0)
    X_train_flat -= mean
    X_test_flat -= mean

    # Create a new model or load an existing one
    clf = model_creator(model_dict, X_train_flat, y_train)
    model_tester(model_dict, clf, X_test_flat, y_test)

    # Checking change in norm of classification hyperplane
    abs_path_o = resolve_path_o(model_dict)
    abs_path_o += 'other/'
    fname = 'norms_' + get_svm_model_name(model_dict)
    ofile = open(abs_path_o + fname + '.txt', 'a')
    ofile.write('No_'+ DR +'\n')
    for i in range(model_dict['classes']):
        ofile.write('{},{} \n'.format(i, np.linalg.norm(clf.coef_[i])))
    ofile.write('\n\n')

    coef_var_list=[]

    test_len = data_dict['test_len']
    no_of_features = data_dict['no_of_features']

    var_array = np.sqrt(np.var(X_test_flat, axis=0))
    var_list = list(var_array)
    coef_norm = np.linalg.norm(clf.coef_[0,:])
    coef_list = list(abs(clf.coef_[0,:])/coef_norm)

    # coef_var_list.append(zip(var_list, coef_list))

    if dataset == 'MNIST':
        rd_list = [100]    # Reduced dimensions to use
        # rd_list = [784, 100]
    elif dataset == 'HAR':
        rd_list = [561, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

    for rd in rd_list:
        print('Reduced dimensions: {}'.format(rd))

        # Dimension reduce dataset and reshape
        X_train_dr, X_test_dr, dr_alg = dr_wrapper(
            X_train_flat, X_test_flat, DR, rd, y_train, rev=rev_flag)

        print X_test_dr.shape

        # With dimension reduced dataset, create new model or load existing one
        clf = model_creator(model_dict, X_train_dr, y_train, adv, rd, rev_flag)
        # Modify classifier to include transformation matrix
        # clf = model_transform(model_dict, clf, dr_alg)
        # Test model on original data
        model_tester(model_dict, clf, X_test_dr, y_test, adv, None, rd, rev_flag)

        print clf.coef_[2,:].shape

        ofile.write(DR+'_{}\n'.format(rd))
        for i in range(model_dict['classes']):
            ofile.write('{},{} \n'.format(i, np.linalg.norm(clf.coef_[i])))
        ofile.write('\n\n')

        no_of_features = data_dict['no_of_features']

        var_array = np.sqrt(np.var(X_test_dr, axis=0))
        var_list = list(var_array)
        coef_norm_dr = np.linalg.norm(clf.coef_[2,:])
        coef_list_dr = list(abs(clf.coef_[2,:]))
        coef_var_list.append(zip(var_list, coef_list_dr))

    mag_var_scatter(model_dict, coef_var_list, len(rd_list), rd, rev_flag)

if __name__ == "__main__":
    main()
