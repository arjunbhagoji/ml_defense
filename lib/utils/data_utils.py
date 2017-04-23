"""
This utility file contains functions that deal with input, output, visual data
of the program. It contains functions that parse arguments, load and parse
datasets, save outputs, save images, etc.
"""

import sys
import os
import argparse
import numpy as np
import pickle
# from scipy.misc import imsave
from matplotlib import pyplot as plt
from matplotlib import image as img

from os.path import dirname

#------------------------------------------------------------------------------#


def model_dict_create():
    """
    Parse arguments and save them in model_dict
    """

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
    parser.add_argument('-dr', '--dim_red', default='pca', type=str,
                        help='Specify dimension reduction scheme')
    parser.add_argument('-r', '--reg', default=None, type=str,
                        help='Specify type of regularization to use')
    parser.add_argument('-b', '--batchsize', default=500, type=int,
                        help='Specify batchsize to use')
    args = parser.parse_args()

    model_dict = {}

    n_epoch = args.n_epoch
    model_dict.update({'dataset': args.dataset})
    model_dict.update({'channels': args.channels})
    model_dict.update({'model_name': args.model})
    model_dict.update({'attack': args.attack})
    model_dict.update({'defense': args.defense})
    model_dict.update({'num_epochs': args.n_epoch})
    model_dict.update({'dim_red': args.dim_red})
    model_dict.update({'reg': args.reg})
    model_dict.update({'batchsize': args.batchsize})

    # Determine output size
    dataset = model_dict['dataset']
    if dataset == 'HAR':
        n_out = 6
    elif dataset == 'MNIST':
        n_out = 10
    elif dataset == 'GTSRB':
        n_out = 43
    model_dict.update({'n_out': n_out})

    return model_dict
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#


def get_model_name(model_dict, rd=None, rev=None):
    """
    Resolve a model's name to save/load based on model_dict
    """

    model_name = model_dict['model_name']
    depth = model_dict['depth']
    width = model_dict['width']
    DR = model_dict['dim_red']

    if model_name == 'mlp' or model_name == 'custom':
        m_name = 'nn_FC_{}_{}'.format(depth, width)
    elif model_name == 'cnn':
        m_name = 'cnn_{}_{}'.format(depth, width)

    reg = model_dict['reg']
    if rd != None:
        m_name += '_{}_{}'.format(rd, DR)
    if rev != None:
        m_name += '_rev'
    if reg != None:
        m_name += '_reg_{}'.format(reg)
    if model_name == 'custom':
        m_name += '_drop'

    return m_name
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#


def load_dataset_MNIST(model_dict):
    """
    Load MNIST data as a (datasize) x 1 x (height) x (width) numpy matrix.
    Each pixel is rescaled to lie in [0,1].
    """

    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, abs_path_i,
                 source='http://yann.lecun.com/exdb/mnist/'):
        print('Downloading %s' % filename)
        urlretrieve(source + filename, abs_path_i + filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(abs_path_i, filename):
        if not os.path.exists(abs_path_i + filename):
            download(filename, abs_path_i)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(abs_path_i + filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(abs_path_i, filename):
        if not os.path.exists(abs_path_i + filename):
            download(filename, abs_path_i)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(abs_path_i + filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    abs_path_i = resolve_path_i(model_dict)
    X_train = load_mnist_images(abs_path_i, 'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(abs_path_i, 'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(abs_path_i, 't10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(abs_path_i, 't10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#


def load_dataset_GTSRB(model_dict):
    """
    Load GTSRB data as a (datasize) x (channels) x (height) x (width) numpy
    matrix. Each pixel is rescaled to lie in [0,1].
    """

    def load_pickled_data(file, columns):
        """
        Loads pickled training and test data.

        Parameters
        ----------
        file    : string
                  Name of the pickle file.
        columns : list of strings
                  List of columns in pickled data we're interested in.

        Returns
        -------
        A tuple of datasets for given columns.
        """

        with open(file, mode='rb') as f:
            dataset = pickle.load(f)
        return tuple(map(lambda c: dataset[c], columns))

    def preprocess(X, channels):
        """
        Preprocess dataset: turn images into grayscale if specified, normalize
        input space to [0,1], reshape array to appropriate shape for NN model
        """

        if channels == 3:
            # Scale features to be in [0, 1]
            X = (X / 255.).astype(np.float32)
            # Rearrange axes to match the desired dimensions
            X = X.swapaxes(1, 3).swapaxes(2, 3)
        else:
            # Convert to grayscale, e.g. single Y channel
            X = 0.299 * X[:, :, :, 0] + 0.587 * \
                X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
            # Scale features to be in [0, 1]
            X = (X / 255.).astype(np.float32)
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        return X

    # Load pickle dataset
    abs_path_i = resolve_path_i(model_dict)
    X_train, y_train = load_pickled_data(abs_path_i + 'train.p',
                                         ['features', 'labels'])
    X_val, y_val = load_pickled_data(abs_path_i + 'valid.p',
                                     ['features', 'labels'])
    X_test, y_test = load_pickled_data(abs_path_i + 'test.p',
                                       ['features', 'labels'])
    # Preprocess loaded data
    channels = model_dict['channels']
    X_train = preprocess(X_train, channels)
    X_val = preprocess(X_val, channels)
    X_test = preprocess(X_test, channels)
    return X_train, y_train, X_val, y_val, X_test, y_test
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#


def load_dataset(model_dict):
    """
    Load and return dataset specified in model_dict
    """

    dataset = model_dict['dataset']
    if dataset == 'MNIST':
        return load_dataset_MNIST(model_dict)
    elif dataset == 'GTSRB':
        return load_dataset_GTSRB(model_dict)
    elif dataset == 'HAR':
        return load_dataset_HAR(model_dict)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#


def get_data_shape(X_train, X_test, X_val=None):
    """
    Creates, updates and returns data_dict containing metadata of the dataset
    """

    # Creates data_dict
    data_dict = {}

    # Updates data_dict with lenght of training, test, validation sets
    train_len = len(X_train)
    test_len = len(X_test)
    data_dict.update({'train_len': train_len, 'test_len': test_len})
    if X_val is not None:
        val_len = len(X_val)
        data_dict.update({'val_len': val_len})
    # else : val_len = None

    # Updates number of dimensions of data
    no_of_dim = X_train.ndim
    data_dict.update({'no_of_dim': no_of_dim})

    # Updates number of features(, number of channels, width, height)
    if no_of_dim == 2:
        no_of_features = X_train.shape[1]
        data_dict.update({'no_of_features': no_of_features})
    elif no_of_dim == 3:
        channels = X_train.shape[1]
        features_per_c = X_train.shape[2]
        no_of_features = channels * features_per_c
        data_dict.update({'no_of_features': no_of_features, 'channels': channels,
                          'features_per_c': features_per_c})
    elif no_of_dim == 4:
        channels = X_train.shape[1]
        height = X_train.shape[2]
        width = X_train.shape[3]
        no_of_features = channels * height * width
        data_dict.update({'height': height, 'width': width, 'channels': channels,
                          'no_of_features': no_of_features})

    return data_dict
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
def reshape_data(X, data_dict, rd=None, rev=None):
    """
    Reshape data into an appropriate shape based on <data_dict> and <rd>, <rev>
    flags.
    """

    no_of_dim = data_dict['no_of_dim']
    X_len = len(X)

    if no_of_dim == 2:
        if rd == None or (rd != None and rev != None):
            no_of_features = data_dict['no_of_features']
            X = X.reshape((X_len, no_of_features))
        elif rd != None:
            X = X.reshape((X_len, rd))
    elif no_of_dim == 3:
        channels = data_dict['channels']
        if rd == None:
            features_per_c = data_dict['features_per_c']
            X = X.reshape((X_len, channels, features_per_c))
        elif rd != None:
            X = X.reshape((X_len, channels, rd))
    elif no_of_dim == 4:
        channels = data_dict['channels']
        if rd == None or (rd != None and rev != None):
            height = data_dict['height']
            width = data_dict['width']
            X = X.reshape((X_len, channels, height, width))

    return X
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#


def save_images(model_dict, data_dict, X_test, adv_x, dev_list, rd=None,
                dr_alg=None, rev=None):
    """
    Save <no_of_img> samples as image files in visual_data folder.
    """

    from lib.utils.dr_utils import invert_dr

    no_of_img = 1
    indices = range(no_of_img)
    X_curr = X_test[indices]
    channels = data_dict['channels']
    atk = model_dict['attack']
    dataset = model_dict['dataset']
    DR = model_dict['dim_red']
    abs_path_v = resolve_path_v(model_dict)

    if (rd != None) and (rev == None):
        X_curr = invert_dr(X_curr, dr_alg, DR)
        features_per_c = X_curr.shape[-1]
        height = int(np.sqrt(features_per_c))
        width = height
        X_curr_rev = X_curr.reshape((no_of_img, channels, height, width))
    elif (rd == None) or (rd != None and rev != None):
        height = data_dict['height']
        width = data_dict['width']

    if channels == 1:
        dev_count = 0
        for dev_mag in dev_list:
            adv_curr = adv_x[indices, :, dev_count]
            if (rd != None) and (rev == None):
                adv_x_rev = invert_dr(adv_curr, dr_alg, DR)
                adv_x_rev = adv_x_rev.reshape(
                    (no_of_img, channels, height, width))
                np.clip(adv_x, 0, 1)
                for i in indices:
                    adv = adv_x_rev[i].reshape((height, width))
                    orig = X_curr_rev[i].reshape((height, width))
                    img.imsave(abs_path_v + '{}_{}_{}_{}_mag{}.png'.format(atk,
                                                                           i, DR, rd, dev_mag), adv * 255, vmin=0, vmax=255,
                               cmap='gray')
                    img.imsave(abs_path_v + '{}_{}_{}_orig.png'.format(i, DR,
                                                                       rd), orig * 255, vmin=0, vmax=255, cmap='gray')

            elif (rd == None) or (rev != None):
                adv_x_curr = adv_x[indices, :, dev_count]
                for i in indices:
                    adv = adv_x_curr[i].reshape((height, width))
                    orig = X_curr[i].reshape((height, width))
                    if rd != None:
                        fname = abs_path_v + ' {}_{}_{}_rev_{}'.format(atk, i,
                                                                       DR, rd)
                    elif rd == None:
                        fname = abs_path_v + '{}_{}'.format(atk, i)
                    img.imsave(fname + '_mag{}.png'.format(dev_mag), adv * 255,
                               vmin=0, vmax=255, cmap='gray')
                    img.imsave(fname + '_orig.png', orig * 255, vmin=0, vmax=255,
                               cmap='gray')
            dev_count += 1
    # else:
        # TODO
        # adv = adv_x[i].swapaxes(0, 2).swapaxes(0, 1)
        # orig = X_test[i].swapaxes(0, 2).swapaxes(0, 1)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#


def utility_write(model_dict, test_acc, test_conf, rd, rev):
    """
    Write utility (accuracy and confidence on test set) of the model on a file.
    The output file is saved in output_data folder.
    """
    # model_name = model_dict['model_name']
    # if model_name in ('mlp', 'custom'):
    #     depth = model_dict['depth']
    #     width = model_dict['width']
    #     fname = 'Utility_nn_{}_{}.txt'.format(depth, width)
    # elif model_name == 'cnn':
    #     fname = 'Utility_cnn_papernot.txt'

    fname = get_model_name(model_dict)
    fname = 'Utility_' + fname + '.txt'
    abs_path_o = resolve_path_o(model_dict)
    ofile = open(abs_path_o + fname, 'a')
    DR = model_dict['dim_red']
    if rd == None:
        ofile.write('No_' + DR + ':\t')
    else:
        if rev == None:
            ofile.write(DR + '_{}:\t'.format(rd))
        else:
            ofile.write(DR + '_rev_{}:\t'.format(rd))
    ofile.write('{:.3f}, {:.3f}\n'.format(test_acc, test_conf))
    ofile.close()
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#


def file_create(model_dict, is_defense, rd, rev=None, strat_flag=None):
    """
    Creates and returns a file descriptor, named corresponding to model,
    attack type, strat, and rev
    """

    # Resolve absolute path to output directory
    abs_path_o = resolve_path_o(model_dict)

    fname = model_dict['attack']
    fname += '_' + get_model_name(model_dict)

    # model_name = model_dict['model_name']
    DR = model_dict['dim_red']
    reg = model_dict['reg']

    # # MLP model
    # if model_name in ('mlp', 'custom'):
    #     depth = model_dict['depth']
    #     width = model_dict['width']
    #     fname += '_nn_{}_{}'.format(depth, width)
    # # CNN model
    # elif model_name == 'cnn':
    #     fname += '_cnn_papernot'

    if strat_flag != None:
        fname += '_strat'
    if rev != None:
        fname += '_rev'
    if rd != None:
        fname += '_' + DR
    if reg != None:
        fname += '_reg_{}'.format(reg)
    if is_defense:
        fname += ('_' + model_dict['defense'])
    plotfile = open(abs_path_o + fname + '.txt', 'a')
    return plotfile
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#


def print_output(model_dict, output_list, dev_list, is_defense=False, rd=None,
                 rev=None, strat_flag=None):
    """
    Creates an output file reporting accuracy and confidence of attack
    """

    plotfile = file_create(model_dict, is_defense, rd, rev, strat_flag)
    plotfile.write('\\\small{' + str(rd) + '}\n')
    # plotfile.write('Mag.   Wrong            Adversarial    Pure      \n')
    for i in range(len(dev_list)):
        plotfile.write('{0:<7.3f}'.format(dev_list[i]))
        for item in output_list[i]:
            plotfile.write('{0:<8.3f}'.format(item))
        plotfile.write('\n')
    plotfile.write('\n\n')
    plotfile.close()
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#


def resolve_path_i(model_dict):
    """
    Resolve absolute paths of input data for different datasets

    Parameters
    ----------
    dataset : string
              Name of desired dataset

    Returns
    -------
    absolute path to input data directory
    """

    script_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
    rel_path_i = 'input_data/' + model_dict['dataset'] + '/'
    abs_path_i = os.path.join(script_dir, rel_path_i)
    if not os.path.exists(abs_path_i):
        os.makedirs(abs_path_i)
    return abs_path_i
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#


def resolve_path_m(model_dict):
    """
    Resolve absolute paths of models for different datasets

    Parameters
    ----------
    model_dict : dictionary
                 contains model's parameters

    Returns
    -------
    absolute path to models directory
    """

    dataset = model_dict['dataset']
    channels = model_dict['channels']
    script_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
    rel_path_m = 'nn_models/' + dataset
    if dataset == 'GTSRB':
        rel_path_m += str(channels)
    abs_path_m = os.path.join(script_dir, rel_path_m + '/')
    if not os.path.exists(abs_path_m):
        os.makedirs(abs_path_m)
    return abs_path_m
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#


def resolve_path_o(model_dict):
    """
    Resolve absolute paths of output data for different datasets

    Parameters
    ----------
    model_dict : dictionary
                 contains model's parameters

    Returns
    -------
    absolute path to output directory
    """

    dataset = model_dict['dataset']
    channels = model_dict['channels']
    script_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
    rel_path_o = 'output_data/' + dataset
    if dataset == 'GTSRB':
        rel_path_o += str(channels)
    abs_path_o = os.path.join(script_dir, rel_path_o + '/')
    if not os.path.exists(abs_path_o):
        os.makedirs(abs_path_o)
    return abs_path_o
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#


def resolve_path_v(model_dict):
    """
    Resolve absolute paths of visual data for different datasets

    Parameters
    ----------
    model_dict : dictionary
                 contains model's parameters

    Returns
    -------
    absolute path to visual data directory
    """

    model_name = get_model_name(model_dict)
    dataset = model_dict['dataset']
    channels = model_dict['channels']
    defense = model_dict['defense']
    script_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
    rel_path_v = 'visual_data/' + dataset + '/' + model_name
    if dataset == 'GTSRB':
        rel_path_v += str(channels)
    if defense != None:
        rel_path_v += '/' + defense
    abs_path_v = os.path.join(script_dir, rel_path_v + '/')
    if not os.path.exists(abs_path_v):
        os.makedirs(abs_path_v)
    return abs_path_v
#------------------------------------------------------------------------------#
