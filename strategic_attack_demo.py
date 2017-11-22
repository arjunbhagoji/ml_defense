import os
import numpy as np
import theano
import theano.tensor as T
from os.path import dirname
import multiprocessing

from matplotlib import pyplot as plt
from functools import partial

from lib.utils.data_utils import *
from lib.utils.model_utils import *
from lib.attacks.nn_attacks import *

#-----------------------------------------------------------------------------#


def strategic_attack(rd, model_dict, dev_list, X_train, y_train, X_test, y_test,
                     mean, X_val=None, y_val=None):
    """
    Helper function called by main() to setup NN model, attack it, print results
    and save adv. sample images.
    """

    # Parameters
    rev_flag = model_dict['rev']
    layer_flag = None
    dim_red = model_dict['dim_red']

    data_dict, test_prediction, dr_alg, X_test, input_var, target_var = \
        model_setup(model_dict, X_train, y_train, X_test, y_test, X_val, y_val,
                    rd, layer=layer_flag)

    print ("Starting strategic attack...")
    adv_x_all, output_list = attack_wrapper(model_dict, data_dict, input_var,
                                            target_var, test_prediction,
                                            dev_list, X_test, y_test, mean,
                                            dr_alg, rd)
    # Printing result to file
    # print_output(model_dict, output_list, dev_list, is_defense=False, rd=rd,
                 # strat_flag=1)

    # Save adv. samples to images
    # if (dim_red == 'pca') or (dim_red == 'dca') or (dim_red == None):
    #     save_images(model_dict, data_dict, X_test, adv_x_all, dev_list,
    #                 rd, dr_alg, rev=rev_flag)
#-----------------------------------------------------------------------------#


def main():
    """
    Main function to run strategic_attack_demo.py. It parses arguments, loads
    dataset and then calls strategic_attack() helper function.
    """

    # Create model_dict from arguments
    model_dict = model_dict_create()

    # No. of deviations to consider
    no_of_mags = 1
    dev_list = np.linspace(0.1, 0.1, no_of_mags)

    # Load dataset specified in model_dict
    print('Loading data...')
    dataset = model_dict['dataset']
    if (dataset == 'MNIST'):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(model_dict)
        # rd_list = [784, 331, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        rd_list = [100]
    elif dataset == 'GTSRB':
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(model_dict)
        rd_list = [1024, 338, 200, 100, 90, 80, 70, 60, 50, 40, 33, 30, 20, 10]
    elif dataset == 'HAR':
        X_train, y_train, X_test, y_test = load_dataset(model_dict)
        # rd_list = [561, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        rd_list = [561]
        X_val = None
        y_val = None

    mean = np.mean(X_train, axis=0)
    X_train -= mean
    X_test -= mean
    if (dataset == 'MNIST') or (dataset == 'GTSRB'):
        X_val -= mean

    # Set up model
    data_dict, test_prediction, dr_alg, X_test, input_var, target_var = \
        model_setup(model_dict, X_train, y_train, X_test, y_test, X_val, y_val)

    # Running attack and saving samples
    print('Creating adversarial samples...')
    adv_x_ini = attack_wrapper(model_dict, data_dict, input_var,
                                            target_var, test_prediction, dev_list, X_test, y_test, mean=mean)
    # print_output(model_dict, output_list, dev_list)

    # save_images(model_dict, data_dict, X_test, adv_x_ini, dev_list, mean)

    # partial_strategic_attack=partial(strategic_attack,X_train=X_train,
    # y_train=y_train,X_test=X_test,y_test=y_test,X_val=X_val,y_val=y_val)

    # for rd in rd_list:
    #     strategic_attack(rd, model_dict, dev_list, X_train, y_train, X_test,
    #                      y_test, mean, X_val, y_val)

    # partial_strategic_attack(784)
    # pool=multiprocessing.Pool(processes=8)
    # pool.map(partial_strategic_attack,rd_list,1)
    # pool.close()
    # pool.join()
#-----------------------------------------------------------------------------#


if __name__ == "__main__":
    main()
#-----------------------------------------------------------------------------#
