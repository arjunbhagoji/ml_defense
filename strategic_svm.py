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
    strat_flag = 1

    # Load dataset and create data_dict to store metadata
    print('Loading data...')
    dataset = model_dict['dataset']
    if (dataset == 'MNIST') or (dataset == 'GTSRB'):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
            model_dict)
        img_flag = None
    elif dataset == 'HAR':
        X_train, y_train, X_test, y_test = load_dataset(model_dict)
        img_flag = None
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

    # Assign parameters
    n_mag = 25                                 # No. of deviations to consider
    dev_list = np.linspace(0.1, 2.5, n_mag)    # A list of deviations mag.
    if dataset == 'MNIST':
        rd_list = [784, 331, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]    # Reduced dimensions to use
        # rd_list = [784]
    elif dataset == 'HAR':
        rd_list = [561, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        # rd_list = [561]
    n_rd = len(rd_list)
    output_list = []
    clear_flag = None
    # Clear old output files
    if clear_flag ==1:
        abs_path_o = resolve_path_o(model_dict)
        _, fname = file_create(model_dict)
        os.remove(abs_path_o + fname + '.txt')
        _, fname = file_create(model_dict, rd=1, strat=strat_flag, rev=rev_flag)
        os.remove(abs_path_o + fname + '.txt')

    # Test clf against adv. samples
    print('Performing attack...')
    if model_dict['classes'] != 2:
        for i in range(n_mag):
            X_adv, y_ini = mult_cls_atk(clf, X_test_flat, mean, dev_list[i])
            output_list.append(acc_calc_all(clf, X_adv, y_test, y_ini))
            if img_flag != None:
                save_svm_images(model_dict, data_dict, X_test, X_adv,
                                dev_list[i])
        fname = print_svm_output(model_dict, output_list, dev_list)
        # subprocess.call(["gnuplot -e \"filename='{}.png'; in_name='{}.txt'\" gnu_in_loop.plg".format(fname,fname)], shell=True)
    # else:
    #     # TODO: 2 classes
    #     print('TODO')

    # Retrain defense and strategic attack
    print('--------------Retrain Defense & Strategic Attack--------------')
    for rd in rd_list:
        output_list = []
        print('Reduced dimensions: {}'.format(rd))

        # Dimension reduce dataset and reshape
        X_train_dr, _, dr_alg = dr_wrapper(
            X_train_flat, X_test_flat, DR, rd, y_train, rev=rev_flag)

        # With dimension reduced dataset, create new model or load existing one
        clf = model_creator(model_dict, X_train_dr, y_train, rd, rev_flag)
        # Modify classifier to include transformation matrix
        clf = model_transform(model_dict, clf, dr_alg)

        model_tester(model_dict, clf, X_test_flat, y_test, rd, rev_flag)

        # rev_flag = 1
        # model_dict['rev'] = rev_flag
        # # Dimension reduce dataset and reshape
        # X_train_dr, _, dr_alg = dr_wrapper(
        #     X_train_flat, X_test_flat, DR, rd, y_train, rev=rev_flag)
        #
        # # With dimension reduced dataset, create new model or load existing one
        # clf_1 = model_creator(model_dict, X_train_dr, y_train, rd, rev_flag)
        # # Modify classifier to include transformation matrix
        # clf_1 = model_transform(model_dict, clf_1, dr_alg)
        # # Test model on original data
        # model_tester(model_dict, clf_1, X_test_flat, y_test, rd, rev_flag)
        #
        # print clf_1.coef_[0]-clf.coef_[0]
        # print np.linalg.norm(clf_1.coef_[0]), np.linalg.norm(clf.coef_[0])
        # print np.dot(clf_1.coef_[0],clf.coef_[0])/(np.linalg.norm(clf_1.coef_[0])*np.linalg.norm(clf.coef_[0]))

        # Strategic attack: create new adv samples based on retrained clf
        print('Performing strategic attack...')
        for i in range(n_mag):
            X_adv, y_ini = mult_cls_atk(clf, X_test_flat, mean, dev_list[i])
            output_list.append(acc_calc_all(clf, X_adv, y_test, y_ini))
            if img_flag != None:
                save_svm_images(model_dict, data_dict, X_test_flat, X_adv,
                            dev_list[i], rd, dr_alg, rev_flag)

        fname = print_svm_output(model_dict, output_list, dev_list, rd,
                                 strat_flag, rev_flag)

    # fname = dataset +'_' + fname
    subprocess.call(
        ["gnuplot -e \"mname='{}'\" gnu_in_loop.plg".format(fname)], shell=True)
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
if __name__ == "__main__":
    main(sys.argv[1:])
#------------------------------------------------------------------------------#
