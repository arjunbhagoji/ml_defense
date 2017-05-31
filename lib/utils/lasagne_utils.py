import numpy as np
import theano
import theano.tensor as T

import lasagne

#------------------------------------------------------------------------------#


def build_custom_mlp(in_shape, n_out, input_var=None, activation='sigmoid',
                     DEPTH=2, WIDTH=800, DROP_INPUT=.2, DROP_HIDDEN=.5):
    """
    Function to create a customizable MLP, with dropout for hidden and input
    layers using Lasagne.
    """

    # All layers are called network, since only the last one is returned.
    # The function can be modified to return the other layers if regularization
    # is to be performed.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=in_shape, input_var=input_var)
    if DROP_INPUT:
        network = lasagne.layers.dropout(network, p=DROP_INPUT)
    # Hidden layers and dropout. Can change 'nonlin' to desired activation
    # function:
    if activation == 'sigmoid':
        nonlin = lasagne.nonlinearities.sigmoid
    elif activation == 'relu':
        nonlin = lasagne.nonlinearities.rectify
    for _ in range(DEPTH):
        network = lasagne.layers.DenseLayer(
            network, WIDTH, nonlinearity=nonlin)
        if DROP_HIDDEN:
            network = lasagne.layers.dropout(network, p=DROP_HIDDEN)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, n_out, nonlinearity=softmax)
    return network
#------------------------------------------------------------------------------#


def build_hidden_fc(in_shape, n_out, input_var=None, activation='sigmoid',
                    WIDTH=100):
    """
    Function to create a neural network with 2 hidden layers of configurable
    width Returns each hidden layer as well, so regularization can be added.
    """

    layers = []
    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    in_layer = lasagne.layers.InputLayer(shape=in_shape, input_var=input_var)
    # Hidden layers and dropout:
    if activation == 'sigmoid':
        nonlin = lasagne.nonlinearities.sigmoid
    elif activation == 'relu':
        nonlin = lasagne.nonlinearities.rectify
    layer_1 = lasagne.layers.DenseLayer(in_layer, WIDTH, nonlinearity=nonlin)
    layers.append(layer_1)
    layer_2 = lasagne.layers.DenseLayer(layer_1, WIDTH, nonlinearity=nonlin)
    layers.append(layer_2)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    layer_3 = lasagne.layers.DenseLayer(layer_2, n_out, nonlinearity=None)
    layers.append(layer_3)
    network = lasagne.layers.NonlinearityLayer(layer_3, nonlinearity=softmax)

    return network, layers
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#

def build_hidden_fc_rd(in_shape, n_out, input_var=None, activation='sigmoid',
                    WIDTH=100, rd = None):

    """
    Function to create a neural network with 2 hidden layers of configurable
    width Returns each hidden layer as well, so regularization can be added.
    """
    layers = []
    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    in_layer = lasagne.layers.InputLayer(shape=in_shape, input_var=input_var)
    # Hidden layers and dropout:
    if activation == 'sigmoid':
        nonlin = lasagne.nonlinearities.sigmoid
    elif activation == 'relu':
        nonlin = lasagne.nonlinearities.rectify
    layer_1 = lasagne.layers.DenseLayer(in_layer, rd, nonlinearity=None)
    layers.append(layer_1)
    layer_2 = lasagne.layers.DenseLayer(layer_1, WIDTH, nonlinearity=nonlin)
    layers.append(layer_2)
    layer_3 = lasagne.layers.DenseLayer(layer_2, WIDTH, nonlinearity=nonlin)
    layers.append(layer_3)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(layer_3, n_out, nonlinearity=softmax)

    return network, layers
#------------------------------------------------------------------------------#


def build_hidden_fc_rd(in_shape, n_out, input_var=None, activation='sigmoid',
                       WIDTH=100, rd=None):
    """
    Function to create a neural network with 2 hidden layers of configurable
    width Returns each hidden layer as well, so regularization can be added.
    """
    layers = []
    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    in_layer = lasagne.layers.InputLayer(shape=in_shape, input_var=input_var)
    # Hidden layers and dropout:
    if activation == 'sigmoid':
        nonlin = lasagne.nonlinearities.sigmoid
    elif activation == 'relu':
        nonlin = lasagne.nonlinearities.rectify
    layer_1 = lasagne.layers.DenseLayer(in_layer, rd, nonlinearity=None)
    layers.append(layer_1)
    layer_2 = lasagne.layers.DenseLayer(layer_1, WIDTH, nonlinearity=nonlin)
    layers.append(layer_2)
    layer_3 = lasagne.layers.DenseLayer(layer_2, WIDTH, nonlinearity=nonlin)
    layers.append(layer_3)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(layer_3, n_out, nonlinearity=softmax)

    return network, layers
#------------------------------------------------------------------------------#


def build_cnn(in_shape, n_out, input_var=None):
    """
    Function to create a CNN according to the specification in Papernot et. al.
    (2016). This CNN has two convolution + pooling stages and two
    fully-connected hidden layers in front of the output layer.
    """

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=in_shape, input_var=input_var)

    # 2 Convolutional layers with 32 kernels of size 5x5.
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 64 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Two fully-connected layers of 200 units:
    network = lasagne.layers.DenseLayer(network, num_units=200,
                                    nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(network, num_units=200,
                                    nonlinearity=lasagne.nonlinearities.rectify)

    # The 10-unit output layer:
    network = lasagne.layers.DenseLayer(network, num_units=n_out,
                                    nonlinearity=lasagne.nonlinearities.softmax)

    return network
#------------------------------------------------------------------------------#


def build_cnn_rd(input_var, rd):
    """
    Function to create a CNN with a similar architecture to the specification in
    Papernot et. al. (2016), but modified to be used with dimension reduced
    data.
    """

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, rd),
                                        input_var=input_var)

    # 2 Convolutional layers with 32 kernels of size 5x5.
    network = lasagne.layers.Conv1DLayer(
        network, num_filters=32, filter_size=5,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv1DLayer(
        network, num_filters=32, filter_size=5,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=2)

    # Another convolution with 64 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv1DLayer(
        network, num_filters=64, filter_size=5,
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv1DLayer(
        network, num_filters=64, filter_size=5,
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=2)

    # Two fully-connected layers of 200 units:
    network = lasagne.layers.DenseLayer(network, num_units=200,
                                    nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(network, num_units=200,
                                    nonlinearity=lasagne.nonlinearities.rectify)

    # The 10-unit output layer:
    network = lasagne.layers.DenseLayer(network, num_units=10,
                                    nonlinearity=lasagne.nonlinearities.softmax)

    return network
#------------------------------------------------------------------------------#
