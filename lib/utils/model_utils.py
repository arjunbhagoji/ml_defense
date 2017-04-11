import sys, os, argparse
import numpy as np

from lib.utils.data_utils import *
from lib.utils.lasagne_utils import *

#------------------------------------------------------------------------------#
def model_creator(model_dict, data_dict, input_var, target_var, rd=None,
                                                                    rev=None):

    n_epoch = model_dict['num_epochs']
    dataset = model_dict['dataset']
    model_name = model_dict['model_name']
    DR = model_dict['dim_red']
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

    # Determine output size
    if dataset == 'HAR':
        n_out = 6
    elif dataset == 'MNIST':
        n_out = 10
    elif dataset == 'GTSRB':
        n_out = 43
    model_dict.update({'n_out':n_out})

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
def model_loader(model_dict, rd=None, DR=None,  rev=None):

    model_name = model_dict['model_name']
    abs_path_m = resolve_path_m(model_dict)

    if model_name == 'cnn':
        model_path = abs_path_m + 'model_cnn_9_layers_papernot'
    elif model_name == 'mlp':
        depth = model_dict['depth']
        width = model_dict['width']
        model_path = abs_path_m + 'model_FC10_{}_{}'.format(depth, width)
    elif model_name == 'custom':
        depth = model_dict['depth']
        width = model_dict['width']
        model_path = abs_path_m + 'model_FC10_{}_{}'.format(depth, width)

    if rd != None: model_path += '_{}_{}'.format(rd, DR)
    if rev != None: model_path += '_rev'
    if model_name == 'custom': model_path += '_drop'
    with np.load(model_path + '.npz') as f:
        param_values = [np.float32(f['arr_%d' % i]) for i in range(len(f.files))]
    return param_values
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_saver(network, model_dict, rd=None, rev=None):

    model_name = model_dict['model_name']
    abs_path_m = resolve_path_m(model_dict)
    DR = model_dict['dim_red']

    if model_name == 'cnn':
        model_path = abs_path_m + 'model_cnn_9_layers_papernot'
    elif model_name == 'mlp':
        depth = model_dict['depth']
        width = model_dict['width']
        model_path = abs_path_m + 'model_FC10_{}_{}'.format(depth, width)
    elif model_name == 'custom':
        depth = model_dict['depth']
        width = model_dict['width']
        model_path = abs_path_m + 'model_FC10_{}_{}'.format(depth, width)

    if rd != None: model_path += '_{}_'.format(rd)+DR
    if rev != None: model_path += '_rev'
    if model_name == 'custom': model_path += '_drop'
    np.savez(model_path + '.npz', *lasagne.layers.get_all_param_values(network))
#------------------------------------------------------------------------------#
