import os
import numpy as np
import theano
import theano.tensor as T
from os.path import dirname

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from lib.utils.theano_utils import *
from lib.utils.lasagne_utils import *
from lib.utils.data_utils import *
from lib.utils.attack_utils import *
from lib.attacks.nn_attacks import *

script_dir = dirname(os.path.abspath(__file__))
rel_path_v="visual_data/"
abs_path_v=os.path.join(script_dir,rel_path_v)
if not os.path.exists(abs_path_v):
    os.makedirs(abs_path_v)

def pca_dr(X_train,X_test,rd):
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

    X_train_dr=X_train_dr.reshape((train_len,1,rd))
    X_test_dr=X_test_dr.reshape((test_len,1,rd))
    return X_train_dr,X_test_dr,pca

def strategic_attack(X_train,y_train,X_test,y_test,X_val,y_val,rd):

    input_var = T.tensor3('inputs')
    target_var = T.ivector('targets')

    network,model_exist_flag,model_dict=model_creator(input_var,target_var,rd)

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
    no_of_mags=50

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

    adv_x_all=fsg_attack(model_dict,input_var,target_var,test_prediction,
                        no_of_mags,X_test_dr,y_test,p_flag,rd)

    return adv_x_all,pca


def main():

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # print length_scales(X_test.reshape((10000,784)), y_test)
    # adv_x_all=fsg_attack(model_dict,input_var,target_var,
    #                 test_prediction,no_of_mags,X_test,y_test,p_flag)

    # rd_list=[784,331,100,50,40,30,20,10]
    rd_list=[50]
    # rd_list=[784]

    hist_data=naive_untargeted_attack(X_test.reshape(10000,784),y_test)
    plt.hist(hist_data,bins=100,normed=1, histtype='step',
                           cumulative=True)
    plt.show()
    # for rd in rd_list:
    #     adv_x_all,pca=strategic_attack(X_train,y_train,X_test,y_test,X_val,y_val,rd)
    #
    #     for i in range(50):
    #         x=pca.inverse_transform(adv_x_all[0,:,i])
    #         x=x.reshape((28,28))
    #         # print x.shape
    #         plt.imsave(abs_path_v+'mnist_'+str(rd)+'_'+str(i)+'.png',x*255, cmap='gray',vmin=0, vmax=255)


if __name__ == "__main__":
    main()
