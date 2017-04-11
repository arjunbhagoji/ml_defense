#from __future__ import print_function

import sys, argparse
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from lib.utils.theano_utils import *
from lib.utils.lasagne_utils import *
from lib.utils.data_utils import *
from lib.utils.attack_utils import *
from lib.utils.dr_utils import *
from lib.utils.model_utils import *
from lib.attacks.nn_attacks import *
from lib.defenses.nn_defenses import *

#from lasagne.regularization import l2

#-----------------------------------------------------------------------------#
def main(argv):

    # Parameters
    batchsize = 500                             # Fixing batchsize
    no_of_mags = 10                             # No. of deviations to consider
    dev_list = np.linspace(0.1, 1.0, no_of_mags)
    rd_list = [331, 100, 50, 40, 30, 20, 10]    # Reduced dimensions used

    # Create model_dict from arguments
    model_dict = model_dict_create()

    # Load and parse specified dataset into numpy arrays
    print('Loading data...')
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(model_dict)

    # Create data_dict containing metadata of dataset
    data_dict = get_data_shape(X_train, X_test, X_val)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create model and check if model already exists
    network, model_exist_flag = model_creator(model_dict, data_dict, input_var,
                                              target_var)

    # Defining symbolic variable for network output
    prediction = lasagne.layers.get_output(network)
    # Defining symbolic variable for network parameters
    params = lasagne.layers.get_all_params(network, trainable=True)
    # Defining symbolic variable for network output with dropout disabled
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    # Building or loading model depending on existence
    if model_exist_flag == 1:
        # Load the correct model:
        param_values = model_loader(model_dict)
        lasagne.layers.set_all_param_values(network, param_values)
    elif model_exist_flag == 0:
        # Launch the training loop
        print('Starting training...')
        model_trainer(input_var, target_var, prediction, test_prediction,
                      params, model_dict, batchsize, X_train, y_train, X_val,
                      y_val)
        model_saver(network, model_dict)

    # Checking performance on test set
    test_model_eval(model_dict, input_var, target_var, test_prediction, X_test,
                    y_test)

    # Running attack and saving samples
    print('Creating adversarial samples...')
    adv_x_all, output_list = attack_wrapper(model_dict, input_var, target_var,
                               test_prediction, dev_list, X_test, y_test)
    print_output(model_dict, output_list, dev_list)
    save_images(model_dict, data_dict, X_test, adv_x_all, dev_list)

    # Run defense
    defense = model_dict['defense']
    for rd in rd_list:
        if defense == 'recons':
            recons_defense(model_dict, input_var, target_var, test_prediction,
                           dev_list, adv_x_all, rd, X_train, y_train, X_test,
                           y_test)
        elif defense == 'retrain':
            retrain_defense(model_dict, data_dict, input_var, target_var,
                            test_prediction, dev_list, adv_x_all, rd, X_train,
                            y_train, X_test, y_test, X_val, y_val)
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
if __name__ == '__main__':
   main(sys.argv[1:])
#-----------------------------------------------------------------------------#
