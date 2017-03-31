import sys, os, argparse
import numpy as np

from lib.utils.data_utils import *
from lib.utils.lasagne_utils import *

#------------------------------------------------------------------------------#
def model_dict_create():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='Specify dataset')
    parser.add_argument('-c', '--channels', default=1, type=int,
                        help='Specify number of input channels')
    parser.add_argument('-m', '--model', default='mlp', type=str,
                        help='Specify neural network model')
    parser.add_argument('--n_epoch', default=None, type=int,
                        help='Specify number of epochs for training')
    parser.add_argument('-a', '--attack', default='fg', type=str,
                        help='Specify method to create adversarial samples')
    parser.add_argument('-d', '--defense', default=None, type=str,
                        help='Specify defense mechanism')
    parser.add_argument('-dr', '--dim_red', default='PCA', type=str,
                        help='Specify defense mechanism')
    args = parser.parse_args()

    model_dict = {}

    n_epoch = args.n_epoch
    model_dict.update({'dataset':args.dataset})
    model_dict.update({'channels':args.channels})
    model_dict.update({'model_name':args.model})
    model_dict.update({'attack':args.attack})
    model_dict.update({'defense':args.defense})
    model_dict.update({'num_epochs':args.n_epoch})
    model_dict.update({'dim_red':args.dim_red})

    return model_dict
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_creator(model_dict,input_var, target_var, rd=None, rev=None):

    n_epoch = model_dict['num_epochs']
    dataset = model_dict['dataset']
    model_name = model_dict['model_name']
    DR = model_dict['dim_red']
    abs_path_m = resolve_path_m(model_dict)

    # Determine input size
    if rd == None:
        if dataset == 'MNIST':
            in_shape = (None, 1, 28, 28)
            n_out = 10
        elif dataset == 'GTSRB':
            in_shape = (None, model_dict['channels'], 32, 32)
            n_out = 43
    else:
        if dataset == 'MNIST':
            in_shape = (None, 1, rd)
            n_out = 10
        elif dataset == 'GTSRB':
            in_shape = (None, model_dict['channels'], rd)
            n_out = 43

    #------------------------------- CNN model --------------------------------#
    if model_name == 'cnn':
        # No dimension reduction on CNN
        if rd != None:
            raise ValueError('Cannot reduce dimension on CNN')
        if n_epoch is not None: num_epochs = n_epoch
        else: num_epochs = 50
        rate = 0.01
        activation = 'relu'
        model_dict.update({'num_epochs':num_epochs, 'rate':rate,
                           'activation':activation})
        model_path = abs_path_m + 'model_cnn_9_layers_papernot'
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
        model_path = abs_path_m + 'model_FC10_{}_{}'.format(depth, width)
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
        model_path = abs_path_m + 'model_FC10_{}_{}'.format(depth, width)
        network = build_custom_mlp(in_shape, n_out, input_var, activation,
                                   int(depth), int(width), float(drop_in),
                                   float(drop_hidden))

    if rd != None:
        model_path += '_{}'.format(rd)
        model_path += '_'+DR
    if rev != None: model_path += '_rev'
    if model_name == 'custom': model_path += '_drop'

    model_exist_flag = 0
    if os.path.exists(model_path + '.npz'): model_exist_flag = 1

    return network, model_exist_flag
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_loader(model_dict, rd=None, rev=None):

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

    if rd != None: model_path += '_{}_PCA'.format(rd)
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

    if rd != None: model_path += '_{}_'+DR.format(rd)
    if rev != None: model_path += '_rev'
    if model_name == 'custom': model_path += '_drop'
    np.savez(model_path + '.npz', *lasagne.layers.get_all_param_values(network))
#------------------------------------------------------------------------------#
