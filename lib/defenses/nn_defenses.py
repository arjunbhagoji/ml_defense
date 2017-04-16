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
    batchsize = 500               # Fixing batchsize
    rev_flag = None
    dim_red = model_dict['dim_red']

    # Doing dimensionality reduction on dataset
    print("Doing {} with rd={} over the training data".format(dim_red, rd))
    X_train, X_test, X_val, dr_alg = dr_wrapper(X_train, X_test, dim_red, rd,
                                                X_val, rev_flag)

    # Getting data parameters after dimensionality reduction
    data_dict = get_data_shape(X_train, X_test, X_val)
    no_of_dim = data_dict['no_of_dim']

    # Prepare Theano variables for inputs and targets
    if no_of_dim == 2: input_var = T.tensor('inputs')
    elif no_of_dim == 3: input_var = T.tensor3('inputs')
    elif no_of_dim == 4: input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Check if model already exists
    network, model_exist_flag = model_creator(model_dict, data_dict, input_var,
                                              target_var, rd, rev=rev_flag)

    #Defining symbolic variable for network output
    prediction = lasagne.layers.get_output(network)
    #Defining symbolic variable for network parameters
    params = lasagne.layers.get_all_params(network, trainable=True)
    #Defining symbolic variable for network output with dropout disabled
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    # Building or loading model depending on existence
    if model_exist_flag == 1:
        # Load the correct model:
        param_values = model_loader(model_dict, rd, dim_red, rev=rev_flag)
        lasagne.layers.set_all_param_values(network, param_values)
    elif model_exist_flag == 0:
        # Launch the training loop.
        print("Starting training...")
        model_trainer(input_var, target_var, prediction, test_prediction,
                      params, model_dict, batchsize, X_train, y_train,
                      X_val, y_val, network)
        model_saver(network, model_dict, rd, rev=rev_flag)

    # Evaluating on retrained inputs
    test_model_eval(model_dict, input_var, target_var, test_prediction,
                    X_test, y_test, rd, rev=rev_flag)

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
    print_output(model_dict, output_list, dev_list, is_defense=True, rd=rd)

    # Saving images
    save_images(model_dict, data_dict, X_test, adv_x, dev_list, rd, dr_alg)
#------------------------------------------------------------------------------#
