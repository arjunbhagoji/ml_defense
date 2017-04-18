import os,sys
import numpy as np
import theano
import theano.tensor as T
import lasagne
import time

from os.path import dirname
from lasagne.regularization import l2, l1
from lib.utils.data_utils import *

#------------------------------------------------------------------------------#
def predict_fn(input_var, test_prediction):

    """Function to predict network output"""

    return theano.function([input_var], T.argmax(test_prediction, axis=1),
                           allow_input_downcast=True)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def grad_fn(input_var, target_var, test_prediction):

    """
    Function to calculate gradient at given data point. Note that the gradient
    in the FSG method attack is taken with respect to the true label.
    The recently discovered 'label leaking' phenomenon can be corrected by
    taking the loss with respect to the predicted label.
    """

    test_loss = loss_fn(test_prediction, target_var)
    req_gradient = T.grad(test_loss, input_var)
    return theano.function([input_var, target_var], req_gradient,
                           allow_input_downcast=True)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):

    """Function to go over minibatches required for training"""

    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def loss_fn(model_predict, target_var, reg=None, network=None):

    """
    Create a loss expression for training, i.e., a scalar objective we want
    to minimize (for our multi-class problem, it is the cross-entropy loss)
    """

    loss_temp = lasagne.objectives.categorical_crossentropy(model_predict,
                                                            target_var)
    loss_temp = loss_temp.mean()
    # Optional regularization
    #layers={layer_1:1e-7,layer_2:1e-7,network:1e-7}
    #l2_penalty=lasagne.regularization.regularize_layer_params_weighted(layers,
                                                                            #l2)
    if reg == 'l2':
        l2_penalty = lasagne.regularization.regularize_network_params(network, l2)
        loss_temp = loss_temp + 1e-7 * l2_penalty
    elif reg =='l1':
        l1_penalty = lasagne.regularization.regularize_network_params(network, l1)
        loss_temp = loss_temp + 1e-7 * l1_penalty
    return loss_temp
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def acc_fn(model_predict, target_var):

    """Theano function to calculate accuracy of input data"""

    return T.mean(T.eq(T.argmax(model_predict, axis=1), target_var),
                       dtype=theano.config.floatX)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def index_fn(model_predict, input_var, target_var):

    """
    Theano function returns an array containing boolean values that indicate
    whether predicted label matches target_var
    """

    index_temp = T.eq(T.argmax(model_predict, axis=1), target_var)
    return theano.function([input_var,target_var], index_temp,
                           allow_input_downcast=True)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def val_fn(input_var, target_var, test_loss, test_acc):

    """Theano function returns total loss and accuracy"""

    return theano.function([input_var, target_var], [test_loss, test_acc],
                           allow_input_downcast=True)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def conf_fn(input_var, model_predict):

    """Theano function to calculate average confidence on input data"""

    conf_temp = T.mean(T.max(model_predict, axis=1), dtype=theano.config.floatX)
    return theano.function([input_var], conf_temp, allow_input_downcast=True)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_trainer(input_var, target_var, prediction, test_prediction, params,
                  model_dict, X_train, y_train, X_val, y_val, network):

    """
    Train NN model with (X_train, y_train) and output training loss, validation
    loss and validation accuracy for (X_val, y_val) at every epoch.
    """

    rate = model_dict['rate']
    num_epochs = model_dict['num_epochs']
    batchsize = model_dict['batchsize']

    if model_dict['reg'] == None:
        loss = loss_fn(prediction, target_var)
    elif model_dict['reg'] != None:
        reg = model_dict['reg']
        loss = loss_fn(prediction, target_var, reg, network)
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum.
    updates = lasagne.updates.nesterov_momentum(loss, params,
                                                learning_rate=rate,
                                                momentum=0.9)

    test_loss = loss_fn(test_prediction, target_var)
    test_acc = acc_fn(test_prediction, target_var)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates,
                               allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    validator = val_fn(input_var, target_var, test_loss, test_acc)
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batchsize,
                                         shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
        if X_val != None:
        # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, batchsize,
                                             shuffle=False):
                inputs, targets = batch
                err, acc = validator(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
              epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        if X_val != None:
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                  val_acc / val_batches * 100))
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def test_model_eval(model_dict, input_var, target_var, test_prediction, X_test,
                    y_test, rd=None, rev=None):

    """
    Evaluate accuracy and confidence of NN model on (X_test, y_test). Result is
    printed on a corresponding utility output file saved in nn_output_data
    folder.
    """

    test_loss = loss_fn(test_prediction, target_var)
    test_acc = acc_fn(test_prediction, target_var)
    validator = val_fn(input_var, target_var, test_loss, test_acc)
    confidence = conf_fn(input_var, test_prediction)
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    test_conf = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = validator(inputs, targets)
        conf = confidence(inputs)
        test_err += err
        test_acc += acc
        test_conf += conf
        test_batches += 1

    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err/test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc/test_batches*100))
    print("  test confidence:\t\t{:.3f} ".format(test_conf/test_batches))

    test_acc = test_acc/test_batches*100
    test_conf = test_conf/test_batches

    utility_write(model_dict,test_acc,test_conf,rd,rev)
#------------------------------------------------------------------------------#
