import os
import numpy as np
import theano
import theano.tensor as T
from os.path import dirname
import multiprocessing

from matplotlib import pyplot as plt
from functools import partial

from lib.utils.theano_utils import *
from lib.utils.lasagne_utils import *
from lib.utils.data_utils import *
from lib.utils.attack_utils import *
from lib.utils.dr_utils import *
from lib.utils.model_utils import *
from lib.attacks.nn_attacks import *

def strategic_attack(rd, model_dict, dev_list, X_train, y_train, X_test, y_test,
                        X_val=None, y_val=None):

    # Parameters
    batchsize = 500                             # Fixing batchsize
    rev_flag = 1
    DR = model_dict['dim_red']

    print("Doing {} with rd={} over the training data".format(DR, rd))
    # train_len = data_dict['train_len']
    # test_len = data_dict['test_len']
    # val_len = data_dict['val_len']

    # if rev_flag == None:
    X_train, X_test, X_val, dr_alg = dr_wrapper(X_train, X_test, DR, rd, X_val,
                                                                    rev_flag)

    # elif rev_flag != None:
    #     X_train, X_test, pca = pca_dr(X_train, X_test, rd, rev_flag)
    #     X_val_dr = pca.transform(X_val).reshape((test_len,channels,rd))
    #     X_val = pca.inverse_transform(X_val_dr).reshape((test_len,channels
    #                                                             ,height,width))
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
        param_values = model_loader(model_dict, rd, DR, rev=rev_flag)
        lasagne.layers.set_all_param_values(network, param_values)
    elif model_exist_flag == 0:
        # Launch the training loop.
        print("Starting training...")
        model_trainer(input_var, target_var, prediction, test_prediction,
                      params, model_dict, batchsize, X_train, y_train,
                      X_val, y_val)
        model_saver(network, model_dict, rd, rev=rev_flag)

    # Evaluating on retrained inputs
    test_model_eval(model_dict, input_var, target_var, test_prediction,
                    X_test, y_test, rd, rev=rev_flag)
    print ("Starting attack...")
    adv_x_all, output_list = attack_wrapper(model_dict, input_var, target_var,
                test_prediction, dev_list, X_test, y_test, rd, rev=rev_flag)

    # Printing result to file
    print_output(model_dict, output_list, dev_list, is_defense=False, rd=rd,
                    rev=rev_flag, strat_flag=1)

    # Save adv. samples to images
    if model_dict['dim_red']=='pca' or model_dict['dim_red']==None:
        save_images(model_dict, data_dict, X_test, adv_x_all, dev_list,
                    rd, dr_alg, rev=rev_flag)


def main():

    model_dict = model_dict_create()

    # rd_list = [784, 331, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]    # Reduced dimensions used
    rd_list =[1024, 100]
    no_of_mags = 50                             # No. of deviations to consider
    dev_list = np.linspace(0.1, 5.0, no_of_mags)
    print('Loading data...')
    if model_dict['dataset'] == 'MNIST' or model_dict['dataset'] == 'GTSRB':
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(model_dict)
    elif  model_dict['dataset'] == 'HAR':
        X_train, y_train, X_test, y_test = load_dataset(model_dict)

    # partial_strategic_attack=partial(strategic_attack,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,X_val=X_val,y_val=y_val)

    for rd in rd_list:
        # partial_strategic_attack(rd)
        strategic_attack(rd, model_dict, dev_list, X_train, y_train, X_test,
                            y_test, X_val, y_val)

    # partial_strategic_attack(784)
    # pool=multiprocessing.Pool(processes=8)
    # pool.map(partial_strategic_attack,rd_list,1)
    # pool.close()
    # pool.join()

if __name__ == "__main__":
    main()
