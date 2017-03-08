import os,sys
import numpy as np
import theano
import theano.tensor as T
import time
from os.path import dirname

import lasagne

script_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
rel_path_o="output_data/"
abs_path_o=os.path.join(script_dir,rel_path_o)

#------------------------------------------------------------------------------#
# Function to predict network output
def predict_fn(input_var,test_prediction):
    return theano.function([input_var],T.argmax(test_prediction,axis=1),
                        allow_input_downcast=True)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Function to calculate gradient at given data point. Note that the gradient
# in the FSG method attack is taken with respect to the true label.
# The recently discovered 'label leaking' phenomenon can be corrected by taking
# the loss with respect to the predicted label.
def grad_fn(input_var,target_var,test_prediction):
    test_loss = loss_fn(test_prediction, target_var)
    req_gradient=T.grad(test_loss,input_var)
    return theano.function([input_var,target_var],req_gradient,
                                    allow_input_downcast=True)
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Function to go over minibatches required for training
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
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
# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
def loss_fn(model_predict,target_var):
    loss_temp = lasagne.objectives.categorical_crossentropy(model_predict,
                                                                target_var)
    # Optional regularization
    #layers={layer_1:1e-7,layer_2:1e-7,network:1e-7}
    #l2_penalty=lasagne.regularization.regularize_layer_params_weighted(layers,
                                                                            #l2)
    #loss=loss+l2_penalty
    loss_temp=loss_temp.mean()
    return loss_temp
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def acc_fn(model_predict,target_var):
    return T.mean(T.eq(T.argmax(model_predict, axis=1), target_var),
                      dtype=theano.config.floatX)
#------------------------------------------------------------------------------#

def index_fn(model_predict,input_var,target_var):
    index_temp=T.eq(T.argmax(model_predict, axis=1), target_var)
    return theano.function([input_var,target_var],index_temp,
                            allow_input_downcast=True)

#------------------------------------------------------------------------------#
def val_fn(input_var,target_var,test_loss,test_acc):
    return theano.function([input_var, target_var], [test_loss, test_acc],
                                allow_input_downcast=True)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def conf_fn(input_var,model_predict):
    conf_temp=T.mean(T.max(model_predict, axis=1),dtype=theano.config.floatX)
    # conf_temp=conf_temp.mean()
    return theano.function([input_var],conf_temp,
                                allow_input_downcast=True)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_trainer(input_var,target_var,prediction,test_prediction,params,
                    model_dict,batchsize,X_train,y_train,X_val,y_val):

    rate=model_dict['rate']
    num_epochs=model_dict['num_epochs']

    loss=loss_fn(prediction, target_var)
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum.
    updates = lasagne.updates.nesterov_momentum(loss, params,
                                            learning_rate=rate, momentum=0.9)

    test_loss = loss_fn(test_prediction, target_var)
    test_acc = acc_fn(test_prediction, target_var)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates,
                                                    allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    validator=val_fn(input_var, target_var, test_loss, test_acc)
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
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batchsize, shuffle=False):
            inputs, targets = batch
            err, acc = validator(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def test_model_eval(model_dict,input_var,target_var,test_prediction,X_test,
                                                    y_test,rd=None,rev=None):
    test_loss = loss_fn(test_prediction, target_var)
    test_acc = acc_fn(test_prediction, target_var)
    validator=val_fn(input_var,target_var,test_loss,test_acc)
    confidence= conf_fn(input_var,test_prediction)
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    test_conf = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = validator(inputs, targets)
        conf= confidence(inputs)
        test_err += err
        test_acc += acc
        test_conf += conf
        test_batches += 1

    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
    print("  test confidence:\t\t{:.3f} ".format(test_conf / test_batches))

    test_acc=test_acc/test_batches*100
    test_conf=test_conf/test_batches

    model_name=model_dict['model_name']

    if not os.path.exists(abs_path_o):
        os.makedirs(abs_path_o)

    if model_name in ('mlp','custom'):
        depth=model_dict['depth']
        width=model_dict['width']
        plotfile=open(abs_path_o+'Utility_MNIST_nn_'+str(depth)+'_'
                    +str(width)+'.txt','a')
        if rd==None:
            plotfile.write('no_dr,'+str.format("{0:.3f}",test_acc)+','+
                                        str.format("{0:.3f}",test_conf)+'\n')
        elif rd!=None:
            plotfile.write(str(rd)+','+str.format("{0:.3f}",test_acc)+','+
                                        str.format("{0:.3f}",test_conf)+'\n')
    elif model_name=='cnn':
        plotfile=open(abs_path_o+'MNIST_cnn_papernot.txt','a')
        if rd==None:
            plotfile.write('no_dr,'+str.format("{0:.3f}",test_acc)+','+
                                        str.format("{0:.3f}",test_conf)+'\n')
        elif rd!=None:
            if rev==None:
                plotfile.write(str(rd)+','+str.format("{0:.3f}",test_acc)+','+
                                            str.format("{0:.3f}",test_conf)+'\n')
            if rev!=None:
                plotfile.write(str(rd)+'_rev,'+str.format("{0:.3f}",test_acc)+','+
                                            str.format("{0:.3f}",test_conf)+'\n')
#------------------------------------------------------------------------------#
