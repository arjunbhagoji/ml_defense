import sys, os, argparse
import numpy as np
import pickle

from lib.utils.lasagne_utils import *
from os.path import dirname

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
    rel_path_i = 'input_data/' + model_dict['dataset'] +'/'
    abs_path_i = os.path.join(script_dir, rel_path_i)
    if not os.path.exists(abs_path_i): os.makedirs(abs_path_i)
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
    rel_path_m = 'models/' + dataset
    if dataset == 'GTSRB': rel_path_m += str(channels)
    abs_path_m = os.path.join(script_dir, rel_path_m + '/')
    if not os.path.exists(abs_path_m): os.makedirs(abs_path_m)
    return abs_path_m
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
        return data/np.float32(256)

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
        if channels == 3:
            # Scale features to be in [0, 1]
            X = (X/255.).astype(np.float32)
            # Rearrange axes to match the desired dimensions
            X.swapaxes(1, 3).swapaxes(2, 3)
        else:
            # Convert to grayscale, e.g. single Y channel
            X = 0.299*X[:,:,:,0] + 0.587*X[:,:,:,1] + 0.114*X[:,:,:,2]
            # Scale features to be in [0, 1]
            X = (X/255.).astype(np.float32)
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
    dataset = model_dict['dataset']
    if dataset == 'MNIST':
        return load_dataset_MNIST(model_dict)
    elif dataset == 'GTSRB':
        return load_dataset_GTSRB(model_dict)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_creator(input_var, target_var, rd=None, rev=None, model_dict=None):

    model_exist_flag = 0
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='MNIST', type=str,
                        help='Specify dataset')
    parser.add_argument('--channels', default=1, type=int,
                        help='Specify number of input channels')
    parser.add_argument('-m', '--model', default='mlp', type=str,
                        help='Specify neural network model')
    parser.add_argument('-n_epoch', type=int,
                        help='Specify number of epochs for training')
    args = parser.parse_args()

    if model_dict == None:
        model_dict = {}
        n_epoch = args.n_epoch
        model_dict.update({'dataset':args.dataset})
        model_dict.update({'channels':args.channels})
        model_dict.update({'model_name':args.model})
    dataset = model_dict['dataset']
    model_name = model_dict['model_name']
    abs_path_m = resolve_path_m(model_dict)

    if dataset == 'MNIST':
        in_shape = (None, 1, 28, 28)
        n_out = 10
    elif dataset == 'GTSRB':
        in_shape = (None, model_dict['channels'], 32, 32)
        n_out = 43

    # TODO: in_shape for rd, rd of 3 channels?

    #------------------------------- CNN model --------------------------------#
    if model_name == 'cnn':
        if n_epoch is not None: num_epochs = n_epoch
        else: num_epochs = 50
        rate = 0.01
        activation = 'relu'
        model_dict.update({'num_epochs':num_epochs, 'rate':rate,
                           'activation':activation})
        model_path = abs_path_m + 'model_cnn_9_layers_papernot'
        if (rd == None) or (rev != None): network = build_cnn(in_shape, n_out, input_var)
        else: network = build_cnn_rd(in_shape, n_out, input_var, rd)

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
        if (rd == None) or (rev != None):
            network, _, _ = build_hidden_fc(in_shape, n_out, input_var, activation, width)
        else:
            network, _, _ = build_hidden_fc_rd(in_shape, n_out, rd, input_var, activation, width)

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
        if (rd == None) or (rev != None):
            network = build_custom_mlp(in_shape, n_out, input_var, activation, int(depth),
                                       int(width), float(drop_in),
                                       float(drop_hidden))
        else:
            network = build_custom_mlp_rd(in_shape, n_out, input_var, activation, int(depth),
                                          int(width), float(drop_in),
                                          float(drop_hidden))

    if rd != None: model_path += '_{}_PCA'.format(rd)
    if rev != None: model_path += '_rev'
    if model_name == 'custom': model_path += '_drop'
    if os.path.exists(model_path + '.npz'): model_exist_flag = 1
    return network, model_exist_flag, model_dict
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
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    return param_values
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_saver(network, model_dict, rd=None, rev=None):

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
    np.savez(model_path + '.npz', *lasagne.layers.get_all_param_values(network))
#------------------------------------------------------------------------------#
