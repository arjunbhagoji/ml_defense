import numpy as np
import argparse
import subprocess
import os
from sklearn.externals import joblib

from lib.utils.svm_utils import *
from lib.utils.dr_utils import *
from lib.attacks.svm_attacks import *

#------------------------------------------------------------------------------#


def main(argv):
    """
    Main function to run strategic_svm.py. Set up SVM classifier, perform
    and evaluate attack, deploy defense and perform strategic attack. Resutls
    and adv. sample images are also saved on each task.
    """

    n_mag = 25                                 # No. of deviations to consider
    dev_list = np.linspace(0.1, 2.5, n_mag)    # A list of deviations mag.

    model_dict, data_dict, X_train, y_train, X_test, y_test, rd_list, mean, img_flag = svm_setup()

    # Create a new model or load an existing one
    clf = model_creator(model_dict, X_train, y_train)
    model_tester(model_dict, clf, X_test, y_test)

    n_features = data_dict['no_of_features']
    test_len = data_dict['test_len']

    X_adv_all = np.zeros((test_len, n_features, n_mag))
    output_list = []
    adv_flag = 1
    # Test clf against adv. samples
    print('Performing attack...')
    if model_dict['classes'] != 2:
        for i in range(n_mag):
            X_adv, y_ini = mult_cls_atk(clf, X_test, mean, dev_list[i])
            output_list.append(acc_calc_all(clf, X_adv, y_test, y_ini))
            X_adv_all[:,:,i] = X_adv
            if img_flag != None:
                save_svm_images(model_dict, data_dict, X_test, X_adv,
                                dev_list[i])
        fname = print_svm_output(model_dict, output_list, dev_list, adv_flag)

    C = model_dict['penconst']
    abs_path_m = resolve_path_m(model_dict)
    # Adversarial training
    dev_list_train = [0.1,0.5,1.0,1.5,2.0]
    # dev_list_train = [0.5]
    for dev_adv in dev_list_train:
        print ('Adversarial training with dev. {}'.format(dev_adv))
        clf_adv = linear_model.SGDClassifier(alpha=C, l1_ratio=0)
        clf_adv.partial_fit(X_train,y_train,np.unique(y_train))
        for epoch in range(5):
            output_list = []
            X_adv_train, _ = mult_cls_atk(clf_adv, X_train, mean, dev_adv)
            X_train_new = np.vstack((X_train,X_adv_train))
            y_train_new = np.hstack((y_train,y_train))
            clf_adv.partial_fit(X_train_new,y_train_new)
            # model_creator(model_dict, X_train_new, y_train_new, adv_flag)
        # Save model
        joblib.dump(clf_adv, abs_path_m + get_svm_model_name(model_dict, adv_flag, dev_adv) + '.pkl')
        # clf_adv = joblib.load(abs_path_m + get_svm_model_name(model_dict, adv_flag, dev_adv) + '.pkl')
        model_tester(model_dict, clf_adv, X_test, y_test, adv_flag, dev_adv)
        for i in range(n_mag):
            X_adv, y_ini = mult_cls_atk(clf_adv, X_test, mean, dev_list[i])
            output_list.append(acc_calc_all(clf_adv, X_adv, y_test, y_ini))
        fname = print_svm_output(model_dict, output_list, dev_list, adv_flag, dev_adv)

        # subprocess.call(["gnuplot -e \"filename='{}.png'; in_name='{}.txt'\" gnu_in_loop.plg".format(fname,fname)], shell=True)
    # else:
    #     # TODO: 2 classes
    #     print('TODO')

    DR = model_dict['dim_red']
    rev_flag = model_dict['rev']
    
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
        X_train_dr, _, _, dr_alg = dr_wrapper(
            X_train, X_test, None, DR, rd, y_train, rev=rev_flag)

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

    fname = dataset + '/' + fname
    subprocess.call(
        ["gnuplot -e \"mname='{}'\" gnu_in_loop.plg".format(fname)], shell=True)
#------------------------------------------------------------------------------#


if __name__ == "__main__":
    main(sys.argv[1:])
#------------------------------------------------------------------------------#
