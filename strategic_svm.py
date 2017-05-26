import numpy as np
import argparse
import subprocess
import os

from lib.utils.svm_utils import *
from lib.utils.data_utils import load_dataset, get_data_shape
from lib.utils.dr_utils import *
from lib.attacks.svm_attacks import *

#------------------------------------------------------------------------------#


def main(argv):
    """
    Main function to run strategic_svm.py. Set up SVM classifier, perform
    and evaluate attack, deploy defense and perform strategic attack. Resutls
    and adv. sample images are also saved on each task.
    """

    # Parse arguments and store in model_dict
    model_dict = svm_model_dict_create()
    DR = model_dict['dim_red']
    rev_flag = model_dict['rev']
    strat_flag = 1          # Set to 1 to indicate strategic attack
    clear_flag = 1          # Set to 1 to clear previous output files

    # Load dataset and create data_dict to store metadata
    print('Loading data...')
    dataset = model_dict['dataset']
    if (dataset == 'MNIST'):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
            model_dict)
        # Number of dimensions used
        rd_list = [784, 331, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        # Flag indicating image data
        img_flag = 1
    elif dataset == 'GTSRB':
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
            model_dict)
        if model_dict['channels'] == 3:
            dim = 3072
        else:
            dim = 1024
        rd_list = [dim, 338, 200, 100, 90, 80, 70, 60, 50, 40, 33, 30, 20, 10]
        img_flag = 1
    elif dataset == 'HAR':
        X_train, y_train, X_test, y_test = load_dataset(model_dict)
        rd_list = [561, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        X_val = None
        y_val = None
        img_flag = None
    n_rd = len(rd_list)

    # TODO: 2 classes case
    # if model_dict['classes'] == 2:
    #     X_train = X_train

    data_dict = get_data_shape(X_train, X_test)
    n_features = data_dict['no_of_features']

    # Reshape dataset to have dimensions suitable for SVM
    X_train = X_train.reshape(-1, n_features)
    X_test = X_test.reshape(-1, n_features)
    # Center dataset by subtracting mean of training set
    mean = np.mean(X_train, axis=0)
    X_train -= mean
    X_test -= mean

    # Preprocess data if specified. Transformation matrix M is returned.
    M = None
    if model_dict['preprocess'] is not None:
        X_train, _, _, M = preprocess(model_dict, X_train, X_val, X_test)

    # Create a new model or load an existing one
    clf = model_creator(model_dict, X_train, y_train)
    # Modify classifier to include transformation matrix
    clf = model_transform(model_dict, clf, M=M)
    # Test the model
    model_tester(model_dict, clf, X_test, y_test)

    # Assign parameters
    n_mag = 25                                 # No. of deviations to consider
    dev_list = np.linspace(0.1, 2.5, n_mag)    # List of deviations mag.
    output_list = []                           # List contains attack output
    # Clear prev output files
    if clear_flag:
        abs_path_o = resolve_path_o(model_dict)
        _, fname = file_create(model_dict)
        os.remove(abs_path_o + fname + '.txt')
        _, fname = file_create(
            model_dict, rd=1, strat=strat_flag, rev=rev_flag)
        os.remove(abs_path_o + fname + '.txt')

    # Vanilla attack: test clf against adv. samples
    print('Performing attack...')
    if model_dict['classes'] != 2:
        for i in range(n_mag):
            X_adv, y_ini = mult_cls_atk(clf, X_test_flat, mean, dev_list[i],
                                        img_flag)
            output_list.append(acc_calc_all(clf, X_adv, y_test, y_ini))
            if img_flag:
                save_svm_images(model_dict, data_dict, X_test + mean,
                                X_adv + mean, dev_list[i])
        fname = print_svm_output(model_dict, output_list, dev_list)

    # else:
    #     # TODO: 2 classes
    #     print('TODO')

    if dataset == 'GTSRB':
        dataset += str(model_dict['channels'])
    fname = dataset + '/' + fname
    # Call gnuplot to plot adv. success vs. mag.
    subprocess.call(
        ["gnuplot -e \"mname='{}'\" gnu_in_loop.plg".format(fname)], shell=True)

    # Retrain defense and strategic attack
    print('--------------Retrain Defense & Strategic Attack--------------')
    for rd in rd_list:
        output_list = []
        print('Reduced dimensions: {}'.format(rd))

        # Dimension reduce dataset and reshape
        X_train_dr, _, dr_alg = dr_wrapper(
            X_train, X_test, DR, rd, y_train, rev=rev_flag)

        # With dimension reduced dataset, create new model or load existing one
        clf = model_creator(model_dict, X_train_dr, y_train, rd, rev_flag)
        # Modify classifier to include transformation matrix
        clf = model_transform(model_dict, clf, dr_alg=dr_alg, M=M)
        # Test model trained on dimension reduced data
        model_tester(model_dict, clf, X_test, y_test, rd, rev_flag)

        # Strategic attack: create new adv samples based on retrained clf
        print('Performing strategic attack...')
        for i in range(n_mag):
            X_adv, y_ini = mult_cls_atk(clf, X_test, mean, dev_list[i],
                                        img_flag)
            output_list.append(acc_calc_all(clf, X_adv, y_test, y_ini))
            if img_flag:
                save_svm_images(model_dict, data_dict, X_test + mean,
                                X_adv + mean, dev_list[i], rd, dr_alg, rev_flag)

        fname = print_svm_output(model_dict, output_list, dev_list, rd,
                                 strat_flag, rev_flag)

    fname = dataset + '/' + fname
    subprocess.call(
        ["gnuplot -e \"mname='{}'\" gnu_in_loop.plg".format(fname)], shell=True)
#------------------------------------------------------------------------------#


if __name__ == "__main__":
    main(sys.argv[1:])
#------------------------------------------------------------------------------#
