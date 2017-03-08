import os
import numpy as np
import theano
import theano.tensor as T
from os.path import dirname

from sklearn.decomposition import PCA

from ..utils.theano_utils import *
from ..utils.lasagne_utils import *
from ..utils.data_utils import *

script_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
rel_path_o="output_data/"
abs_path_o=os.path.join(script_dir,rel_path_o)


def local_fns(input_var,target_var,test_prediction):
    predictor=predict_fn(input_var, test_prediction)
    confidence= conf_fn(input_var,test_prediction)
    gradient=grad_fn(input_var, target_var, test_prediction)
    test_loss=loss_fn(test_prediction, target_var)
    test_acc=acc_fn(test_prediction, target_var)
    validator=val_fn(input_var, target_var, test_loss, test_acc)
    indexer=index_fn(test_prediction, input_var, target_var)
    return validator,indexer,predictor,confidence

def acc_calc(X_adv,y,validator,indexer,confidence):
    loss_i,acc_i=validator(X_adv,y)
    c_i=100-acc_i*100
    indices_i=indexer(X_adv,y)
    i_i=np.where(indices_i==0)[0]
    conf_i=np.float64(confidence(X_adv[i_i]))
    return [c_i,conf_i]


def acc_calc_all(X_adv,y_test,X_test_mod,i_c,validator,indexer,predictor,
                confidence):
    o_list=[]
    # Accuracy vs. true labels. Confidence on mismatched predictions
    c_w,conf_w=acc_calc(X_adv, y_test, validator, indexer, confidence)
    o_list.extend([c_w,conf_w])
    #Accuracy vs. predicted labels
    c_a,conf_a=acc_calc(X_adv, predictor(X_test_mod), validator, indexer,
                        confidence)
    o_list.extend([c_a,conf_a])
    # Accuracy for adv. examples generated from correctly classified
    # examples
    c_p,conf_p=acc_calc(X_adv[i_c], y_test[i_c], validator, indexer, confidence)
    o_list.extend([c_p,conf_p])
    return o_list

def file_out(o_list,dev_mag,plotfile):
    plotfile.write(str(dev_mag)+",")
    for item in o_list[0:-1]:
        plotfile.write(str.format("{0:.3f}",item)+",")
    plotfile.write(str.format("{0:.3f}",o_list[-1]))
    plotfile.write("\n")

def pca_dr(X_train,X_test,rd,recons_flag=None):
    train_len=len(X_train)
    test_len=len(X_test)
    #Reshaping for PCA function
    PCA_in_train=X_train.reshape(train_len,784)
    PCA_in_test=X_test.reshape(test_len,784)
    #Fitting the PCA model on training data
    pca=PCA(n_components=rd)
    pca_train=pca.fit(PCA_in_train)
    # Reconstructing training and test data
    X_train_dr=pca.transform(PCA_in_train)
    X_test_dr=pca.transform(PCA_in_test)

    if recons_flag!=None:
        X_train_rev=pca.inverse_transform(X_train_dr)
        X_train_rev=X_train_rev.reshape((train_len,1,28,28))
        X_test_rev=pca.inverse_transform(X_test_dr)
        X_test_rev=X_test_rev.reshape((test_len,1,28,28))
        return X_test_rev,pca
    elif recons_flag==None:
        X_train_dr=X_train_dr.reshape((train_len,1,rd))
        X_test_dr=X_test_dr.reshape((test_len,1,rd))
        return X_train_dr,X_test_dr, pca

# Function to implement the reconstruction defense
def recons_defense(model_dict,input_var,target_var,test_prediction,
        adv_x_all,rd,X_train,y_train,X_test,y_test):

    """
    Evaluates effect of reconstruction defense on adversarial success. Prints
    output to a .txt file in '/outputs'. All 3 adversarial success counts
    are reported.
    : param model_dict: name of the model
    : param input_var: symbolic input variable
    : param target_var: symbolic output variable
    : param test_prediction: model output on test data_utils
    : param adv_examples_test: Array to store adversarial samples
    : param X_test: Test data
    : param y_test: Test data labels
    """
    recons_flag=1
    test_len=len(X_test)

    plotfile=file_create(model_dict,rd,recons_flag)

    print("Doing PCA with rd={} over the training data".format(rd))

    X_test_rev,pca=pca_dr(X_train,X_test,rd,recons_flag)

    # Evaluating on re-constructed inputs
    test_model_eval(model_dict,input_var,target_var,test_prediction,
                                                        X_test_rev,y_test,rd)
    validator,indexer,predictor,confidence=local_fns(input_var,target_var,
                                                                test_prediction)
    indices_c=indexer(X_test_rev,y_test)
    i_c=np.where(indices_c==1)[0]

    eps=np.linspace(0.01,0.1,10)

    mag_count=0
    for dev_mag in eps:
        X_adv_dr=pca.transform(adv_x_all[:,:,mag_count])
        recons_adv=(pca.inverse_transform(X_adv_dr)).reshape((test_len,1,28,28))
        o_list=[]
        o_list=acc_calc_all(recons_adv,y_test,X_test_rev,i_c,validator,indexer,
                                                        predictor,confidence)
        #Printing out to file
        file_out(o_list,dev_mag,plotfile)
        mag_count=mag_count+1
    plotfile.close()

#------------------------------------------------------------------------------#
# Function to implement the re-training defense
def retrain_defense(model_dict,input_var,target_var,test_prediction,
        adv_x_all,rd,X_train,y_train,X_test,y_test,X_val,y_val):

    """
    Evaluates effect of reconstruction defense on adversarial success. Prints
    output to a .txt file in '/outputs'. All 3 adversarial success counts
    are reported.
    : param model_dict: name of the model
    : param input_var: symbolic input variable
    : param target_var: symbolic output variable
    : param test_prediction: model output on test data_utils
    : param adv_examples_test: Array to store adversarial samples
    : param X_test: Test data
    : param y_test: Test data labels
    """
    plotfile=file_create(model_dict,rd)

    input_var = T.tensor3('inputs')
    target_var = T.ivector('targets')

    network,model_exist_flag=model_creator(input_var,target_var,model_dict,rd)

    #Defining symbolic variable for network output
    prediction = lasagne.layers.get_output(network)
    #Defining symbolic variable for network parameters
    params = lasagne.layers.get_all_params(network, trainable=True)
    #Defining symbolic variable for network output with dropout disabled
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    print("Doing PCA with rd={} over the training data".format(rd))

    X_train_dr,X_test_dr,pca=pca_dr(X_train,X_test,rd)
    test_len=len(X_test)
    X_val=X_val.reshape(test_len,784)
    X_val_dr=pca.transform(X_val).reshape((test_len,1,rd))

    # Fixing batchsize
    batchsize=500
    p_flag=1

    # Building or loading model depending on existence
    if model_exist_flag==1:
        # Load the correct model:
        param_values=model_loader(model_dict,rd)
        lasagne.layers.set_all_param_values(network, param_values)
    elif model_exist_flag==0:
        # Launch the training loop.
        print("Starting training...")
        model_trainer(input_var,target_var,prediction,test_prediction,params,
                        model_dict,batchsize,X_train_dr,y_train,X_val_dr,y_val)
        model_saver(network,model_dict,rd)

    # Evaluating on retrained inputs
    test_model_eval(model_dict,input_var,target_var,test_prediction,X_test_dr,
                                                                    y_test,rd)

    mag_count=0

    validator,indexer,predictor,confidence=local_fns(input_var, target_var,
                                                            test_prediction)
    indices_c=indexer(X_test_dr,y_test)
    i_c=np.where(indices_c==1)[0]
    for dev_mag in np.linspace(0.01,0.1,10):
        X_adv_dr=pca.transform(adv_x_all[:,:,mag_count]).reshape((test_len,1,rd))
        o_list=acc_calc_all(X_adv_dr, y_test, X_test_dr, i_c, validator, indexer,
                            predictor, confidence)
        file_out(o_list, dev_mag, plotfile)
        mag_count=mag_count+1
    plotfile.close()
#------------------------------------------------------------------------------#
