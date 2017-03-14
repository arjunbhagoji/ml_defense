import sys, os, argparse
import numpy as np

from lib.utils.lasagne_utils import *
from os.path import dirname

script_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
rel_path_i = 'input_data/'
abs_path_i = os.path.join(script_dir,rel_path_i)
rel_path_m = 'models/'
abs_path_m = os.path.join(script_dir,rel_path_m)
if not os.path.exists(abs_path_m): os.makedirs(abs_path_m)
if not os.path.exists(abs_path_i): os.makedirs(abs_path_i)

#------------------------------------------------------------------------------#
#Function to load MNIST data
def load_dataset():
    """
    Load MNIST data as a (datasize) x (no_of_features) numpy matrix for use with
    scikit's SVM module. Each pixel is rescaled to lie in [0,1].
    : dir_name: Specify the directory where the data is/should be located
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
    # script_dir = os.path.dirname(os.path.dirname(os.path.dirname(
    #                                              os.path.abspath(__file__))))

    if not os.path.exists(abs_path_i): os.makedirs(abs_path_i)
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
def model_creator(input_var, target_var, rd=None, rev=None, model_dict=None):

    model_exist_flag = 0
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='mlp', type=str,
                        help='Specify neural network model')
    parser.add_argument('-n_epoch', type=int,
                        help='Specify number of epochs for training')
    args = parser.parse_args()

    if model_dict == None:
        model_dict = {}
        n_epoch = args.n_epoch
        model_dict.update({'model_name':args.model})
    model_name = model_dict['model_name']

    #------------------------------- CNN model --------------------------------#
    if model_name == 'cnn':
        if n_epoch is not None: num_epochs = n_epoch
        else: num_epochs = 50
        rate = 0.01
        activation = 'relu'
        model_dict.update({'num_epochs':num_epochs, 'rate':rate,
                           'activation':activation})
        model_path = abs_path_m + 'model_cnn_9_layers_papernot'
        if rd == None:
            network = build_cnn(input_var)
        elif rd != None:
            if rev == None:
                network = build_cnn_rd(input_var, rd)
                model_path += '_%d_PCA' % rd
            elif rev != None:
                network = build_cnn(input_var)
                model_path += '_%d_PCA_rev' % rd
        if os.path.exists(model_path + '.npz'): model_exist_flag = 1

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
        model_path = abs_path_m + 'model_FC10_%d_%d' % (depth, width)
        if rd == None:
            network, _, _ = build_hidden_fc(input_var, activation, width)
        elif rd != None:
            network, _, _ = build_hidden_fc_rd(rd, input_var, activation, width)
            model_path += '_%d_PCA' % rd
        if os.path.exists(model_path + '.npz'): model_exist_flag = 1

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
        model_path = abs_path_m + 'model_FC10_%d_%d' % (depth, width)
        if rd == None:
            network = build_custom_mlp(input_var, activation, int(depth),
                                       int(width), float(drop_in),
                                       float(drop_hidden))
        elif rd != None:
            network = build_custom_mlp_rd(input_var, activation, int(depth),
                                          int(width), float(drop_in),
                                          float(drop_hidden))
            model_path += '_%d_PCA' % rd
        if os.path.exists(model_path + '_drop.npz'): model_exist_flag = 1

    return network, model_exist_flag, model_dict
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_loader(model_dict, rd=None, rev=None):

    model_name = model_dict['model_name']

    #------------------------------- CNN model --------------------------------#
    if model_name == 'cnn':
        model_path = abs_path_m + 'model_cnn_9_layers_papernot'
        if rd != None:
            if rev == None: model_path += '_%d_PCA' % rd
            elif rev != None: model_path += '_%d_PCA_rev' % rd

    #------------------------------- MLP model --------------------------------#
    elif model_name == 'mlp':
        depth = model_dict['depth']
        width = model_dict['width']
        model_path = abs_path_m + 'model_FC10_%d_%d' % (depth, width)
        if rd != None: model_path += '_%d_PCA' % rd

    #------------------------------ Custom model ------------------------------#
    elif model_name == 'custom':
        depth=model_dict['depth']
        width=model_dict['width']
        model_path = abs_path_m + 'model_FC10_%d_%d' % (depth, width)
        if rd != None: model_path += '_%d_PCA' % rd
        model_path += '_drop'

    with np.load(model_path + '.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    return param_values
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_saver(network, model_dict, rd=None, rev=None):

    model_name = model_dict['model_name']
    if not os.path.exists(abs_path_m): os.makedirs(abs_path_m)

    #------------------------------- CNN model --------------------------------#
    if model_name == 'cnn':
        model_path = abs_path_m + 'model_cnn_9_layers_papernot'
        if rd != None:
            if rev == None: model_path += '_%d_PCA' % rd
            elif rev != None: model_path += '_%d_PCA_rev' % rd

    #------------------------------- MLP model --------------------------------#
    elif model_name == 'mlp':
        depth = model_dict['depth']
        width = model_dict['width']
        model_path = abs_path_m + 'model_FC10_%d_%d' % (depth, width)
        if rd != None: model_path += '_%d_PCA' % rd

    #------------------------------ Custom model ------------------------------#
    elif model_name == 'custom':
        depth=model_dict['depth']
        width=model_dict['width']
        model_path = abs_path_m + 'model_FC10_%d_%d' % (depth, width)
        if rd != None: model_path += '_%d_PCA' % rd
        model_path += '_drop'

    np.savez(model_path + '.npz', *lasagne.layers.get_all_param_values(network))
#------------------------------------------------------------------------------#
