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
                     X_val=None, y_val=None):

    """
    Helper function called by main() to setup NN model, attack it, print results
    and save adv. sample images.
    """

    # Parameters
    rev_flag = None
    dim_red = model_dict['dim_red']

    data_dict, test_prediction, dr_alg, X_test, input_var, target_var = \
        model_setup(model_dict, X_train, y_train, X_test, y_test, X_val, y_val,
                    rd, rev=rev_flag)

    print ("Starting attack...")
    adv_x_all, output_list = attack_wrapper(model_dict, data_dict, input_var,
                                            target_var, test_prediction,
                                            dev_list, X_test, y_test, rd,
                                            rev=rev_flag)

    # Printing result to file
    print_output(model_dict, output_list, dev_list, is_defense=False, rd=rd,
                 rev=rev_flag, strat_flag=1)

    # Save adv. samples to images
    if (dim_red == 'pca') or (dim_red == None):
        save_images(model_dict, data_dict, X_test, adv_x_all, dev_list,
                    rd, dr_alg, rev=rev_flag)
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
def main():

    """
    Main function to run strategic_attack_demo.py. It parses arguments, loads
    dataset and then calls strategic_attack() helper function.
    """

    # Create model_dict from arguments
    model_dict = model_dict_create()

    # Reduced dimensions used
    rd_list = [784, 331, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    # No. of deviations to consider
    no_of_mags = 50
    dev_list = np.linspace(0.1, 5.0, no_of_mags)

    # Load dataset specified in model_dict
    print('Loading data...')
    dataset = model_dict['dataset']
    if (dataset == 'MNIST') or (dataset == 'GTSRB'):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(model_dict)
    elif dataset == 'HAR':
        X_train, y_train, X_test, y_test = load_dataset(model_dict)

    # partial_strategic_attack=partial(strategic_attack,X_train=X_train,
    # y_train=y_train,X_test=X_test,y_test=y_test,X_val=X_val,y_val=y_val)

    for rd in rd_list:
        # partial_strategic_attack(rd)
        strategic_attack(rd, model_dict, dev_list, X_train, y_train, X_test,
                            y_test, X_val, y_val)

    # partial_strategic_attack(784)
    # pool=multiprocessing.Pool(processes=8)
    # pool.map(partial_strategic_attack,rd_list,1)
    # pool.close()
    # pool.join()
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
#-----------------------------------------------------------------------------#
