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
def main():

    """
    Main function to run strategic_attack_demo.py. It parses arguments, loads
    dataset and then calls strategic_attack() helper function.
    """

    # Create model_dict from arguments
    model_dict = model_dict_create()

    # No. of deviations to consider
    no_of_mags = 50
    dev_list = np.linspace(0.1, 5.0, no_of_mags)

    # Load dataset specified in model_dict
    print('Loading data...')
    dataset = model_dict['dataset']
    if (dataset == 'MNIST'):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(model_dict)
        # rd_list = [None, 784, 331, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        rd_list = [None, 331, 100, 80, 60, 40, 20]
        # rd_list = [None,784,100]
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
    if (dataset == 'MNIST') or (dataset == 'GTSRB'): X_val -= mean

    # fig, ax = plt.subplots(nrows=1, ncols=1)

    # for rd in rd_list:
    #     model_setup_carlini(model_dict, X_train, y_train, X_test, y_test, X_val, y_val, mean, rd, ax)

    partial_carlini = partial(model_setup_carlini, model_dict=model_dict, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val,
                                mean=mean)
    pool=multiprocessing.Pool(processes=8)
    pool.map(partial_carlini,rd_list,1)
    pool.close()
    pool.join()

    # dim_red = model_dict['dim_red']
    # plt.legend()
    # plt.savefig('carlini_l2_hist_'+dim_red+'.png')

#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
#-----------------------------------------------------------------------------#
