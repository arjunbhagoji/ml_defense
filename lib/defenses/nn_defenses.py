import os
import numpy as np
import theano
import theano.tensor as T
from os.path import dirname

from sklearn.decomposition import PCA

from ..utils.theano_utils import *
from ..utils.lasagne_utils import *
from ..utils.data_utils import *
from ..utils.attack_utils import *
from ..utils.model_utils import *
from ..utils.dr_utils import *

#------------------------------------------------------------------------------#
# Function to implement the reconstruction defense
def recons_defense(model_dict, data_dict, input_var, target_var, test_prediction,
                   dev_list, adv_x_ini, rd, X_train, y_train, X_test, y_test):
    """
    Evaluates effect of reconstruction defense on adversarial success. Prints
    output to a .txt file in '/outputs'. All 3 adversarial success counts
    are reported.
    : param model_dict: name of the model
    : param input_var: symbolic input variable
    : param target_var: symbolic output variable
    : param test_prediction: model output on test data_utils
    : param adv_examples_test: Array to store adversarial samples
    : param X_test: Test data
    : param y_test: Test data labels
    """

    rev_flag = 1
    dim_red = model_dict['dim_red']
    X_val = None

    # Doing dimensionality reduction on dataset
    print("Doing {} with rd={} over the training data".format(dim_red, rd))
    X_train, X_test, dr_alg = dr_wrapper(X_train, X_test, dim_red, rd,
                                                X_val, rev_flag)

    # Evaluating on re-constructed inputs
    test_model_eval(model_dict, input_var, target_var, test_prediction,
                    X_test, y_test, rd)
    validator, indexer, predictor, confidence = local_fns(input_var, target_var,
                                                          test_prediction)
    indices_c = indexer(X_test, y_test)
    i_c = np.where(indices_c == 1)[0]

    adv_len = len(adv_x_ini)
    dev_len = len(dev_list)
    no_of_features = data_dict['no_of_features']
    adv_x = np.zeros((adv_len, no_of_features, dev_len))

    output_list = []
    for mag_count in range(dev_len):
        X_adv = dr_alg.transform(adv_x_ini[:,:,mag_count])
        X_adv_rev = invert_dr(X_adv, dr_alg, dim_red)
        adv_x[:,:,mag_count] = X_adv_rev
        X_adv_rev = reshape_data(X_adv_rev, data_dict, rd, rev=rev_flag)
        output_list.append(acc_calc_all(X_adv_rev, y_test, X_test, i_c,
                                     validator, indexer, predictor, confidence))
    # Printing result to file
    print_output(model_dict, output_list, dev_list, is_defense=True, rd=rd)

    # Saving images
    save_images(model_dict, data_dict, X_test, adv_x, dev_list, rd, dr_alg,
                rev_flag)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Function to implement the re-training defense
def retrain_defense(model_dict, dev_list, adv_x_ini, rd, X_train, y_train,
                    X_test, y_test, X_val, y_val):

    """
    Evaluates effect of reconstruction defense on adversarial success. Prints
    output to a .txt file in '/outputs'. All 3 adversarial success counts
    are reported.
    : param model_dict: name of the model
    : param input_var: symbolic input variable
    : param target_var: symbolic output variable
    : param test_prediction: model output on test data_utils
    : param adv_examples_test: Array to store adversarial samples
    : param X_test: Test data
    : param y_test: Test data labels
    """

    # Parameters
    rev_flag = None

    data_dict, test_prediction, dr_alg, X_test, input_var, target_var = \
        model_setup(model_dict, X_train, y_train, X_test, y_test, X_val, y_val,
                    rd)

    validator, indexer, predictor, confidence = local_fns(input_var, target_var,
                                                          test_prediction)

    indices_c = indexer(X_test, y_test)
    i_c = np.where(indices_c == 1)[0]

    adv_len = len(adv_x_ini)
    dev_len = len(dev_list)
    adv_x = np.zeros((adv_len, rd, dev_len))

    output_list = []
    for mag_count in range(dev_len):
        X_adv = dr_alg.transform(adv_x_ini[:,:,mag_count])
        adv_x[:,:,mag_count] = X_adv
        X_adv = reshape_data(X_adv, data_dict, rd)
        output_list.append(acc_calc_all(X_adv, y_test, X_test, i_c,
                                     validator, indexer, predictor, confidence))
    # Printing result to file
    is_defense = True
    print_output(model_dict, output_list, dev_list, is_defense, rd)

    # Saving images
    save_images(model_dict, data_dict, X_test, adv_x, dev_list, rd, dr_alg)
#------------------------------------------------------------------------------#
