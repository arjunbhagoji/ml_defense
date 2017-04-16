import numpy as np
import argparse
from sklearn import svm
from sklearn.decomposition import PCA

from lib.utils.svm_utils import *
from lib.utils.data_utils import *
from lib.utils.dr_utils import *

#------------------------------------------------------------------------------#
def mult_cls_atk(clf, X_test, dev_mag, rd=None, rev=None):

    """
    Returns
    (1) Adversarial samples generated from <X_test> for linear SVM <clf>
        with perturbation magnitude <dev_mag>
    (2) Predicted labels of <X_test> by <clf>
    """

    test_len = len(X_test)
    X_adv = np.zeros((test_len, X_test.shape[1]))
    y_ini = np.zeros(test_len)
    classes = clf.intercept_.shape[0]

    for i in range(test_len):
        x_ini = X_test[i,:].reshape(1, -1)
        ini_class = clf.predict(x_ini)
        w = clf.coef_[ini_class[0],:]
        d_list = []
        i_list = []
        distances = clf.decision_function(x_ini)
        for j in range(classes):
            if j == ini_class[0]: continue
            w_curr = clf.coef_[j,:]
            d_list.append(abs(distances[0,j] - distances[0,ini_class[0]])
                          /np.linalg.norm(w_curr - w))
            i_list.append(j)
            i_d_list = zip(i_list, d_list)
        i_d_list = sorted(i_d_list, key = lambda x:x[1])
        min_index = i_d_list[0][0]
        min_dist = i_d_list[0][1]
        w_min = clf.coef_[min_index,:]
        x_adv = (x_ini - dev_mag*((w - w_min)/(np.linalg.norm(w - w_min))))
        if rd == None or rev != None: np.clip(x_adv, 0, 1)
        X_adv[i,:] = x_adv
        y_ini[i] = ini_class

    return X_adv, y_ini
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def main(argv):

    # Parse arguments and store in model_dict
    model_dict = svm_model_dict_create()
    DR = model_dict['dim_red']
    rev_flag = None

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
    no_of_features = data_dict['no_of_features']

    # Reshape dataset to have dimensions suitable for SVM
    X_train_flat = X_train.reshape(-1, no_of_features)
    X_test_flat = X_test.reshape(-1, no_of_features)
    # X_val_flat= X_val.reshape(-1, no_of_features)

    # Create a new model or load an existing one
    clf = model_creator(model_dict, X_train_flat, y_train)
    model_tester(model_dict, clf, X_test_flat, y_test)

    # Assign parameters
    n_mag = 10                                  # No. of deviations to consider
    dev_list = np.linspace(0.1, 1.0, n_mag)    # A list of deviations mag.
    # rd_list = [784, 331, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]    # Reduced dimensions to use
    rd_list = [784, 100]
    n_rd = len(rd_list)
    output_list = []

    # Test clf against adv. samples
    print('Performing attack...')
    if model_dict['classes'] != 2:
        for i in range(n_mag):
            X_adv, y_ini = mult_cls_atk(clf, X_test_flat, dev_list[i])
            output_list.append(acc_calc_all(clf, X_adv, y_test, y_ini))
            save_svm_images(model_dict, n_features, X_test, X_adv, dev_list[i])
        print_svm_output(model_dict, output_list, dev_list)
    else:
        # TODO: 2 classes
        print('TODO')

    # Retrain defense and strategic attack
    print('--------------Retrain Defense & Strategic Attack--------------')
    for rd in rd_list:
        output_list = []
        print('Reduced dimensions: {}'.format(rd))

        # Dimension reduce dataset and reshape
        X_train_dr, X_test_dr, dr_alg = dr_wrapper(X_train_flat, X_test_flat,
                                                    DR, rd)

        X_train_dr = X_train_dr.reshape(-1, rd)
        X_test_dr = X_test_dr.reshape(-1, rd)

        # With dimension reduced dataset, create new model or load existing one
        clf = model_creator(model_dict, X_train_dr, y_train, rd, rev_flag)
        model_tester(model_dict, clf, X_test_dr, y_test)

        # Strategic attack: create new adv samples based on retrained clf
        print('Performing strategic attack...')
        for i in range(n_mag):
            X_adv, y_ini = mult_cls_atk(clf, X_test_dr, dev_list[i], rd,
                                                                    rev_flag)
            output_list.append(acc_calc_all(clf, X_adv, y_test, y_ini))
            if model_dict['dim_red']=='pca' or model_dict['dim_red']==None:
                dr_alg = pca
                save_svm_images(model_dict, n_features, X_test_dr, X_adv,
                                    dev_list[i], rd, dr_alg)
        print_svm_output(model_dict, output_list, dev_list, rd, strat_flag=1)

if __name__ == "__main__":
   main(sys.argv[1:])
