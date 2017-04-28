"""
This file contains helper functions that assist in managing NN models including
creating Lasagne model, loading parameters to model, saving parameters, and
setting up model.
"""

import sys, os, argparse
import numpy as np

from lib.utils.data_utils import *
from lib.utils.lasagne_utils import *
from lib.utils.theano_utils import *
from lib.utils.dr_utils import *

#------------------------------------------------------------------------------#
def model_creator(model_dict, data_dict, input_var, target_var, rd=None,
                  rev=None):

    """
    Create a Lasagne model/network as specified in <model_dict> and check
    whether the model already exists in model folder.
    """

    n_epoch = model_dict['num_epochs']
    dataset = model_dict['dataset']
    model_name = model_dict['model_name']
    DR = model_dict['dim_red']
    n_out = model_dict['n_out']
    no_of_dim = data_dict['no_of_dim']

    # Determine input size
    if no_of_dim == 2:
        no_of_features = data_dict['no_of_features']
        in_shape = (None, no_of_features)
    elif no_of_dim == 3:
        channels = data_dict['channels']
        features_per_c = data_dict['features_per_c']
        in_shape = (None, channels, features_per_c)
    elif no_of_dim == 4:
        channels = data_dict['channels']
        height = data_dict['height']
        width = data_dict['width']
        in_shape = (None, channels, height, width)

    #------------------------------- CNN model --------------------------------#
    if model_name == 'cnn':
        # # No dimension reduction on CNN
        # if rd != None:
        #     raise ValueError('Cannot reduce dimension on CNN')
        if n_epoch is not None: num_epochs = n_epoch
        else: num_epochs = 50
        depth = 9
        width = 'papernot'
        rate = 0.01
        activation = 'relu'
        model_dict.update({'num_epochs':num_epochs, 'rate':rate, 'depth':depth,
                           'width':width, 'activation':activation})
        network = build_cnn(in_shape, n_out, input_var)

    #------------------------------- MLP model --------------------------------#
    elif model_name == 'mlp':
        if n_epoch is not None: num_epochs = n_epoch
        else: num_epochs = 500
        depth = 2
        width = 100
        rate = 0.01
        activation = 'sigmoid'
        model_dict.update({'num_epochs':num_epochs, 'rate':rate, 'depth':depth,
                           'width':width, 'activation':activation})
        network, _, _ = build_hidden_fc(in_shape, n_out, input_var, activation,
                                        width)

    #------------------------------ Custom model ------------------------------#
    elif model_name == 'custom':
        if n_epoch is not None: num_epochs = n_epoch
        else: num_epochs = 500
        depth = 2
        width = 100
        drop_in = 0.2
        drop_hidden = 0.5
        rate = 0.01
        activation = 'sigmoid'
        model_dict.update({'num_epochs':num_epochs, 'rate':rate, 'depth':depth,
                           'width':width, 'activation':activation,
                           'drop_in':drop_in, 'drop_hidden':drop_hidden})
        network = build_custom_mlp(in_shape, n_out, input_var, activation,
                                   int(depth), int(width), float(drop_in),
                                   float(drop_hidden))

    abs_path_m = resolve_path_m(model_dict)
    model_path = abs_path_m + get_model_name(model_dict, rd, rev)
    model_exist_flag = 0
    if os.path.exists(model_path + '.npz'): model_exist_flag = 1

    return network, model_exist_flag
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_loader(model_dict, rd=None, DR=None, rev=None):

    """
    Load parameters of the saved Lasagne model
    """

    mname = get_model_name(model_dict, rd, rev)
    abs_path_m = resolve_path_m(model_dict)

    model_path = abs_path_m + mname

    # if model_name == 'cnn':
    #     model_path = abs_path_m + 'cnn_{}_{}_papernot'.format(depth, width)
    # elif model_name == 'mlp':
    #     depth = model_dict['depth']
    #     width = model_dict['width']
    #     model_path = abs_path_m + 'model_FC_{}_{}'.format(depth, width)
    # elif model_name == 'custom':
    #     depth = model_dict['depth']
    #     width = model_dict['width']
    #     model_path = abs_path_m + 'model_FC_{}_{}'.format(depth, width)
    #
    # reg = model_dict['reg']
    #
    # if rd != None: model_path += '_{}_{}'.format(rd, DR)
    # if rev != None: model_path += '_rev'
    # if reg != None: model_path += '_reg_{}'.format(reg)
    # if model_name == 'custom': model_path += '_drop'
    with np.load(model_path + '.npz') as f:
        param_values = [np.float32(f['arr_%d' % i]) for i in range(len(f.files))]
    return param_values
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_saver(network, model_dict, rd=None, rev=None):

    """
    Save model parameters in model foler as .npz file compatible with Lasagne
    """

    mname = get_model_name(model_dict, rd, rev)
    abs_path_m = resolve_path_m(model_dict)

    model_path = abs_path_m + mname

    # model_name = model_dict['model_name']
    # abs_path_m = resolve_path_m(model_dict)
    # DR = model_dict['dim_red']
    #
    # if model_name == 'cnn':
    #     model_path = abs_path_m + 'model_cnn_{}_{}_papernot'.format(depth, width)
    # elif model_name == 'mlp':
    #     depth = model_dict['depth']
    #     width = model_dict['width']
    #     model_path = abs_path_m + 'model_FC_{}_{}'.format(depth, width)
    # elif model_name == 'custom':
    #     depth = model_dict['depth']
    #     width = model_dict['width']
    #     model_path = abs_path_m + 'model_FC_{}_{}'.format(depth, width)
    #
    # reg = model_dict['reg']
    #
    # if rd != None: model_path += '_{}_{}'.format(rd, DR)
    # if rev != None: model_path += '_rev'
    # if reg != None: model_path += '_reg_{}'.format(reg)
    # if model_name == 'custom': model_path += '_drop'
    np.savez(model_path + '.npz', *lasagne.layers.get_all_param_values(network))
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_setup(model_dict, X_train, y_train, X_test, y_test, X_val=None,
                y_val=None, rd=None, rev=None):

    """
    Main function to set up network (create, load, test, save)
    """

    dim_red = model_dict['dim_red']
    if rd != None:
        # Doing dimensionality reduction on dataset
        print("Doing {} with rd={} over the training data".format(dim_red, rd))
        X_train, X_test, X_val, dr_alg = dr_wrapper(X_train, X_test, dim_red,
                                                    rd, rev, X_val)
    else: dr_alg = None

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
                                              target_var, rd, rev)

    #Defining symbolic variable for network output
    prediction = lasagne.layers.get_output(network)
    #Defining symbolic variable for network parameters
    params = lasagne.layers.get_all_params(network, trainable=True)
    #Defining symbolic variable for network output with dropout disabled
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    # Building or loading model depending on existence
    if model_exist_flag == 1:
        # Load the correct model:
        param_values = model_loader(model_dict, rd, dim_red, rev)
        lasagne.layers.set_all_param_values(network, param_values)
    elif model_exist_flag == 0:
        # Launch the training loop.
        print("Starting training...")
        model_trainer(input_var, target_var, prediction, test_prediction,
                      params, model_dict, X_train, y_train,
                      X_val, y_val, network)
        model_saver(network, model_dict, rd, rev)

    # Evaluating on retrained inputs
    test_model_eval(model_dict, input_var, target_var, test_prediction,
                    X_test, y_test, rd, rev)

    return data_dict, test_prediction, dr_alg, X_test, input_var, target_var
#------------------------------------------------------------------------------#
