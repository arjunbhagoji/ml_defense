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

#------------------------------------------------------------------------------#
def pca_dr(X_train, X_test, rd, recons_flag=None):
    train_len = len(X_train)
    test_len = len(X_test)
    height = X_train.shape[2]
    width = X_train.shape[3]
    n_features = height*width
    # Reshaping for PCA function
    PCA_in_train = X_train.reshape(train_len, n_features)
    PCA_in_test = X_test.reshape(test_len, n_features)
    # Fitting the PCA model on training data
    pca = PCA(n_components=rd)
    pca_train = pca.fit(PCA_in_train)
    # Reconstructing training and test data
    X_train_dr = pca.transform(PCA_in_train)
    X_test_dr = pca.transform(PCA_in_test)

    if recons_flag != None:
        X_train_rev = pca.inverse_transform(X_train_dr)
        X_train_rev = X_train_rev.reshape((train_len, 1, height, width))
        X_test_rev = pca.inverse_transform(X_test_dr)
        X_test_rev = X_test_rev.reshape((test_len, 1, height, width))
        return X_test_rev, pca
    elif recons_flag == None:
        X_train_dr = X_train_dr.reshape((train_len, 1, rd))
        X_test_dr = X_test_dr.reshape((test_len, 1, rd))
        return X_train_dr, X_test_dr, pca
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Function to implement the reconstruction defense
def recons_defense(model_dict, input_var, target_var, test_prediction, dev_list,
                   adv_x_all, rd, X_train, y_train, X_test, y_test):
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
    recons_flag = 1
    test_len = len(X_test)
    height = X_train.shape[2]
    width = X_train.shape[3]
    n_features = height*width

    print("Doing PCA with rd={} over the training data".format(rd))

    X_test_rev, pca = pca_dr(X_train, X_test, rd, recons_flag)

    # Evaluating on re-constructed inputs
    test_model_eval(model_dict, input_var, target_var, test_prediction,
                    X_test_rev, y_test, rd)
    validator, indexer, predictor, confidence = local_fns(input_var, target_var,
                                                          test_prediction)
    indices_c = indexer(X_test_rev, y_test)
    i_c = np.where(indices_c == 1)[0]

    output_list = []
    for mag_count in range(len(dev_list)):
        X_adv_dr = pca.transform(adv_x_all[:,:,mag_count])
        recons_adv = pca.inverse_transform(X_adv_dr).reshape((test_len, 1,
                                                              height, width))
        output_list.append(acc_calc_all(recons_adv, y_test, X_test_rev, i_c,
                                     validator, indexer, predictor, confidence))
    # Printing result to file
    print_output(model_dict, output_list, dev_list, is_defense=True, rd=rd)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Function to implement the re-training defense
def retrain_defense(model_dict, input_var, target_var, test_prediction,
                    dev_list, adv_x_all, rd, X_train, y_train, X_test, y_test,
                    X_val, y_val):
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

    recons_flag = 0
    test_len = len(X_test)
    height = X_train.shape[2]
    width = X_train.shape[3]
    n_features = height*width

    input_var = T.tensor3('inputs')
    target_var = T.ivector('targets')

    network, model_exist_flag, _ = model_creator(input_var, target_var, rd=rd,
                                                 model_dict=model_dict)

    # Defining symbolic variable for network output
    prediction = lasagne.layers.get_output(network)
    # Defining symbolic variable for network parameters
    params = lasagne.layers.get_all_params(network, trainable=True)
    # Defining symbolic variable for network output with dropout disabled
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    print("Doing PCA with rd={} over the training data".format(rd))

    X_train_dr, X_test_dr, pca = pca_dr(X_train, X_test, rd)
    val_len = len(X_val)
    X_val = X_val.reshape(val_len, n_features)
    X_val_dr = pca.transform(X_val).reshape((val_len, 1, rd))

    # Fixing batchsize
    batchsize = 500

    # Building or loading model depending on existence
    if model_exist_flag == 1:
        # Load the correct model:
        param_values = model_loader(model_dict, rd)
        lasagne.layers.set_all_param_values(network, param_values)
    elif model_exist_flag == 0:
        # Launch the training loop.
        print("Starting training...")
        model_trainer(input_var, target_var, prediction, test_prediction,
                      params, model_dict, batchsize, X_train_dr, y_train,
                      X_val_dr, y_val)
        model_saver(network, model_dict, rd)

    # Evaluating on retrained inputs
    test_model_eval(model_dict, input_var, target_var, test_prediction,
                    X_test_dr, y_test, rd)

    validator, indexer, predictor, confidence = local_fns(input_var, target_var,
                                                          test_prediction)
    indices_c = indexer(X_test_dr, y_test)
    i_c = np.where(indices_c == 1)[0]

    output_list = []
    for mag_count in range(len(dev_list)):
        X_adv_dr = pca.transform(adv_x_all[:,:,mag_count]).reshape((test_len,
                                                                    1, rd))
        output_list.append(acc_calc_all(X_adv_dr, y_test, X_test_dr, i_c,
                                     validator, indexer, predictor, confidence))
    # Printing result to file
    print_output(model_dict, output_list, dev_list, is_defense=True, rd=rd)
#------------------------------------------------------------------------------#
