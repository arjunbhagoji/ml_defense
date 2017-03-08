import os
import numpy as np
import theano
import theano.tensor as T
from os.path import dirname
import multiprocessing

from matplotlib import pyplot as plt
from functools import partial

from lib.utils.theano_utils import *
from lib.utils.lasagne_utils import *
from lib.utils.data_utils import *
from lib.utils.attack_utils import *
from lib.utils.dr_utils import *
from lib.attacks.nn_attacks import *

script_dir = dirname(os.path.abspath(__file__))
rel_path_v="visual_data/"
abs_path_v=os.path.join(script_dir,rel_path_v)
if not os.path.exists(abs_path_v):
    os.makedirs(abs_path_v)


def strategic_attack(rd,X_train,y_train,X_test,y_test,X_val,y_val):

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network,model_exist_flag,model_dict=model_creator(input_var,target_var,rd,rev=1)

    #Defining symbolic variable for network output
    prediction = lasagne.layers.get_output(network)
    #Defining symbolic variable for network parameters
    params = lasagne.layers.get_all_params(network, trainable=True)
    #Defining symbolic variable for network output with dropout disabled
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    print("Doing PCA with rd={} over the training data".format(rd))

    train_len=len(X_train)
    test_len=len(X_test)

    X_train_dr,X_test_dr,pca=pca_dr(X_train,X_test,rd)
    X_train_rev=pca.inverse_transform(X_train_dr).reshape((train_len,1,28,28))
    X_test_rev=pca.inverse_transform(X_test_dr).reshape((test_len,1,28,28))
    test_len=len(X_test)
    X_val=X_val.reshape(test_len,784)
    X_val_dr=pca.transform(X_val).reshape((test_len,1,rd))
    X_val_rev=pca.inverse_transform(X_val_dr).reshape((test_len,1,28,28))

    # Fixing batchsize
    batchsize=500
    p_flag=1
    no_of_mags=50

    # Building or loading model depending on existence
    if model_exist_flag==1:
        # Load the correct model:
        param_values=model_loader(model_dict,rd,rev=1)
        lasagne.layers.set_all_param_values(network, param_values)
    elif model_exist_flag==0:
        # Launch the training loop.
        print("Starting training...")
        model_trainer(input_var,target_var,prediction,test_prediction,params,
                        model_dict,batchsize,X_train_rev,y_train,X_val_rev,y_val)
        model_saver(network,model_dict,rd,rev=1)

    # Evaluating on retrained inputs
    test_model_eval(model_dict,input_var,target_var,test_prediction,X_test_rev,
                                                                    y_test,rd,rev=1)
    print ("Starting attack...")
    adv_x_all,output_list,dev_list=attack_wrapper(input_var,target_var,test_prediction,
                        no_of_mags,X_test_rev,y_test,rd,rev=1)
    # dev_list=[2.0]
    # for max_dev in dev_list:
    #     adv_x_all,output_list,deviation_list=l_bfgs_attack(input_var, target_var, test_prediction, X_test_dr,
    #                                 y_test, rd,max_dev)

    plotfile=file_create(model_dict, rd, fsg_flag=1,strat_flag=1,rev=1)

    for i in range(len(dev_list)):
        o_list=output_list[i]
        eps=dev_list[i]
        file_out(o_list,eps,plotfile)

    # for i in range(50):
    #     x=pca.inverse_transform(adv_x_all[0,:,i])
    #     x=x.reshape((28,28))
    #     # print x.shape
    #     plt.imsave(abs_path_v+'fsg_mod_mnist_'+str(rd)+'_'+str(i)+'_'+str(dev_list[i])+'.png',x*255, cmap='gray',vmin=0, vmax=255)


    # return adv_x_all,pca
    # return pca


def main():

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    rd_list=[200,100,90,80,70,60,50,40,30,20,10]
    # rd_list=[30]
    # rd_list=[784,100]
    # rd_list=[100]

    # hist_data=naive_untargeted_attack(X_test.reshape(10000,784),y_test)
    # plt.hist(hist_data,bins=100,normed=1, histtype='step',
    #                        cumulative=True)
    # plt.show()
    # partial_strategic_attack=partial(strategic_attack,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,X_val=X_val,y_val=y_val)

    for rd in rd_list:
        # partial_strategic_attack(rd)
        strategic_attack(rd,X_train,y_train,X_test,y_test,X_val,y_val)

    # partial_strategic_attack(784)
    # pool=multiprocessing.Pool(processes=8)
    # pool.map(partial_strategic_attack,rd_list,1)
    # pool.close()
    # pool.join()

if __name__ == "__main__":
    main()
