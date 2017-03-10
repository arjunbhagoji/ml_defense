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
from lib.attacks.nn_attacks import *

#from lasagne.regularization import l2

def main(argv):

    # Parse argument to train a new model or load an existing model
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Force to train a new model')
    parser.add_argument('-m', '--model', default='mlp', type=str,
                        help='Specify neural network model')
    parser.add_argument('-n_epoch', type=int,
                        help='Specify number of epochs for training')
    args = parser.parse_args()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network, model_exist_flag, model_dict = model_creator(args, input_var, target_var)

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    train_len = len(X_train)
    test_len = len(X_test)

    # Fixing batchsize
    batchsize = 500
    p_flag = 1

    # Defining symbolic variable for network output
    prediction = lasagne.layers.get_output(network)
    # Defining symbolic variable for network parameters
    params = lasagne.layers.get_all_params(network, trainable=True)
    # Defining symbolic variable for network output with dropout disabled
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    # Building or loading model depending on existence
    if model_exist_flag == 1 and not args.train:
        # Load the correct model:
        param_values = model_loader(model_dict)
        lasagne.layers.set_all_param_values(network, param_values)
    else:
        # Launch the training loop.
        print("Starting training...")
        model_trainer(input_var, target_var, prediction, test_prediction,
                      params, model_dict, batchsize, X_train, y_train, X_val,
                      y_val)
        model_saver(network, model_dict)

    # Checking performance on test set
    test_model_eval(model_dict, input_var, target_var, test_prediction, X_test,
                    y_test)

    # No. of deviations to consider
    no_of_mags = 10

    # Reduced dimensions used
    rd_list = [331, 100, 50, 40, 30, 20, 10]
    # rd_list=[100]

    # Creating adv. examples
    adv_x_all, output_list, dev_list = attack_wrapper(input_var, target_var,
                                                test_prediction, no_of_mags,
                                                X_test, y_test)

    plotfile = file_create(model_dict, fsg_flag=1)

    for i in range(len(dev_list)):
        o_list = output_list[i]
        eps = dev_list[i]
        file_out(o_list, eps, plotfile)


    # for rd in rd_list:
    #     recons_defense(model_dict,input_var,target_var,test_prediction,
    #                     adv_x_all,rd,X_train,y_train,X_test,y_test)

    #
    # pool=multiprocessing.Pool(processes=8)
    # pool.map(pca_attack,rd_list)
    # pool.close()
    # pool.join()
if __name__ == "__main__":
   main(sys.argv[1:])
