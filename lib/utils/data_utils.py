import sys, os, argparse
import numpy as np
import pickle
from scipy.misc import imsave

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
    rel_path_m = 'nn_models/' + dataset
    if dataset == 'GTSRB': rel_path_m += str(channels)
    abs_path_m = os.path.join(script_dir, rel_path_m + '/')
    if not os.path.exists(abs_path_m): os.makedirs(abs_path_m)
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
    if dataset == 'GTSRB': rel_path_o += str(channels)
    abs_path_o = os.path.join(script_dir, rel_path_o + '/')
    if not os.path.exists(abs_path_o): os.makedirs(abs_path_o)
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
    absolute path to output directory
    """
    dataset = model_dict['dataset']
    channels = model_dict['channels']
    script_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
    rel_path_v = 'visual_data/' + dataset
    if dataset == 'GTSRB': rel_path_v += str(channels)
    abs_path_v = os.path.join(script_dir, rel_path_v + '/')
    if not os.path.exists(abs_path_v): os.makedirs(abs_path_v)
    return abs_path_v
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
            X = X.swapaxes(1, 3).swapaxes(2, 3)
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
# Saves first 10 images from the test set and their adv. samples
def save_images(model_dict, n_features, X_test, adv_x, dev_list, rd=None, dr_alg=None):
    no_of_img = 10
    indices = range(no_of_img)
    X_curr = X_test[indices]
    channels = X_curr.shape[1]
    atk = model_dict['attack']
    dataset = model_dict['dataset']
    DR =model_dict['dim_red']
    abs_path_v=resolve_path_v(model_dict)
    if rd!=None:
        height = int(np.sqrt(n_features))
        width = height
        X_curr_rev = dr_alg.inverse_transform(X_curr).reshape((no_of_img, channels, height, width))
    elif rd==None:
        height = X_test.shape[2]
        width = X_test.shape[3]

    if channels == 1:
        dev_count=0
        for dev_mag in dev_list:
            if rd!=None:
                adv_x_curr=dr_alg.inverse_transform(adv_x[indices,:,dev_count]).reshape((no_of_img, channels, height, width))
                for i in indices:
                    adv = adv_x_curr[i].reshape((height, width))
                    orig = X_curr_rev[i].reshape((height, width))
                    imsave(abs_path_v+'{}_{}_{}_{}_{}_mag{}.jpg'.format(atk, dataset, i, DR, rd, dev_mag), adv)
                    imsave(abs_path_v+'{}_{}_{}_orig.jpg'.format(atk, dataset, i, DR, rd), orig)
            elif rd == None:
                adv_x_curr=adv_x[indices,:,dev_count]
                for i in indices:
                    adv = adv_x_curr[i].reshape((height,width))
                    orig = X_curr[i].reshape((height,width))
                    imsave(abs_path_v+'{}_{}_{}_{}_{}_mag{}.jpg'.format(atk, dataset, i, dev_mag), adv)
                    imsave(abs_path_v+'{}_{}_{}_orig.jpg'.format(atk, dataset, i), orig)
    else:
        adv = adv_x[i].swapaxes(0, 2).swapaxes(0, 1)
        orig = X_test[i].swapaxes(0, 2).swapaxes(0, 1)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def utility_write(model_dict,test_acc,test_conf,rd,rev):
    model_name = model_dict['model_name']
    if model_name in ('mlp', 'custom'):
        depth = model_dict['depth']
        width = model_dict['width']
        fname = 'Utility_nn_{}_{}.txt'.format(depth, width)
    elif model_name == 'cnn':
        fname = 'Utility_cnn_papernot.txt'

    abs_path_o = resolve_path_o(model_dict)
    ofile = open(abs_path_o + fname, 'a')
    DR=model_dict['dim_red']
    if rd == None:
        ofile.write('No_'+DR+':\t')
    else:
        if rev == None: ofile.write(DR+'_{}: '.format(rd))
        else: ofile.write(DR+'_rev {}:\t'.format(rd))
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

    model_name = model_dict['model_name']
    DR = model_dict['dim_red']
    fname = model_dict['attack']
    # MLP model
    if model_name in ('mlp', 'custom'):
        depth = model_dict['depth']
        width = model_dict['width']
        fname += '_nn_{}_{}'.format(depth, width)
    # CNN model
    elif model_name == 'cnn':
        fname += '_cnn_papernot'

    if strat_flag != None: fname += '_strat'
    if rev != None: fname += '_rev'
    if rd != None: fname += '_'+DR
    if is_defense: fname += ('_' + model_dict['defense'])
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
    plotfile.write('\\\small{{}}\n'.format(rd))
    # plotfile.write('Mag.   Wrong            Adversarial    Pure      \n')
    for i in range(len(dev_list)):
        plotfile.write('{0:<7.3f}'.format(dev_list[i]))
        for item in output_list[i]:
            plotfile.write('{0:<8.3f}'.format(item))
        plotfile.write("\n")
    plotfile.close()
#------------------------------------------------------------------------------#
