#from __future__ import print_function

import sys, getopt
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from lib.utils.theano_utils import *
from lib.utils.data_utils import *
from lib.utils.lasagne_utils import *
from lib.attacks.nn_attacks import *
from lib.defenses.nn_defenses import *

#from lasagne.regularization import l2

def main(argv):
    # Getting main directory name
    script_dir=os.path.dirname(__file__)
    print("Loading data...")
    # global X_train, y_train, X_val, y_val, X_test, y_test
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    # global train_len, test_len
    train_len=len(X_train)
    test_len=len(X_test)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    try:
        opts, args = getopt.getopt(argv,"hm:f:")
        print("Building model and compiling functions...")
    except getopt.GetoptError:
        print("MNIST_fast_sign_gradient_w_train_reconst.py -m <model> -f"
                    +"<modelexistflag>")
        sys.exit()
    for opt, arg in opts:
        if opt=='-h':
            print("MNIST_fast_sign_gradient_w_train_reconst.py -m <model> -f"
                                +"<modelexistflag>")
            sys.exit()
        # Create neural network model (depending on first command line
        # parameter)
        elif opt=='-m':
            model_name=arg
            if arg=='cnn':
                NUM_EPOCHS=50
                rate=0.1
                network=build_cnn(input_var)
            elif arg=='mlp':
                NUM_EPOCHS=500
                DEPTH=2
                WIDTH=100
                rate=0.01
                global layer_1, layer_2
                network, layer_1, layer_2=build_hidden_fc(input_var,
                                                            WIDTH=WIDTH)
            elif arg=='custom':
                NUM_EPOCHS=500
                DEPTH=2
                WIDTH=100
                DROP_IN=0.2
                DROP_HIDDEN=0.5
                rate=0.01
                network = build_custom_mlp(input_var, int(DEPTH), int(WIDTH),
                                            float(DROP_IN), float(DROP_HIDDEN))
        elif opt=='-f':
            model_exist_flag=int(arg)
    # Fixing batchsize
    batchsize=500

    #Defining symbolic variable for network output
    prediction = lasagne.layers.get_output(network)
    #Defining symbolic variable for network parameters
    params = lasagne.layers.get_all_params(network, trainable=True)
    #Defining symbolic variable for network output with dropout disabled
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # Function to get network prediction on given input
    predict_fn=theano.function([input_var],T.argmax(test_prediction,
                                        axis=1),allow_input_downcast=True)

    # Building or loading model depending on existence
    rel_path_m="models/"
    abs_path_m=os.path.join(script_dir,rel_path_m)
    if not os.path.exists(abs_path_m):
        os.makedirs(abs_path_m)
    if model_exist_flag==1:
        # Load the correct model:
        if model_name in ('mlp','custom'):
            with np.load(abs_path_m+'model_FC10_'+str(DEPTH)+'_'+
                                                    str(WIDTH)+'_.npz') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(network, param_values)
        elif model_name=='cnn':
            with np.load(abs_path_m+'model_cnn_9_layers_papernot.npz') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(network, param_values)
    elif model_exist_flag==0:
        # Launch the training loop.
        print("Starting training...")
        model_trainer(input_var,target_var,prediction,test_prediction,params,
                        NUM_EPOCHS,rate,batchsize,X_train,y_train,X_val,y_val)
        if not os.path.exists(abs_path_m):
            os.makedirs(abs_path_m)
        if model_name in ('mlp','custom'):
            np.savez(abs_path_m+'model_FC10_'+str(DEPTH)+'_'+str(WIDTH)+'_.npz',
                *lasagne.layers.get_all_param_values(network))
        elif model_name=='cnn':
            np.savez(abs_path_m+'model_cnn_9_layers_papernot.npz',
                        *lasagne.layers.get_all_param_values(network))

    # Checking performance on test set
    test_model_eval(input_var,target_var,test_prediction,X_test,y_test)

    # No. of deviations to consider
    no_of_mags=10

    # Arrays to store adv. examples
    adv_examples_train=np.zeros((train_len,784,no_of_mags))
    adv_examples_test=np.zeros((test_len,784,no_of_mags))

    # Reduced dimensions used
    rd_list=[331,100,50,40,30,20,10]

    rel_path_o="output_data/"
    abs_path_o=os.path.join(script_dir,rel_path_o)
    if not os.path.exists(abs_path_o):
        os.makedirs(abs_path_o)
    # Creating adv. examples
    if model_name in ('mlp', 'custom'):
        fsg_attack(model_name,abs_path_o,input_var,target_var,
                        test_prediction,adv_examples_test,X_test,y_test)
        for rd in rd_list:
            recons_defense(model_name,abs_path_o,input_var,target_var,
                            test_prediction,adv_examples_test,rd,X_train,
                            y_train,X_test,y_test,DEPTH,WIDTH)
    elif model_name=='cnn':
        fsg_attack(model_name,abs_path_o)

    #
    # pool=multiprocessing.Pool(processes=8)
    # pool.map(pca_attack,rd_list)
    # pool.close()
    # pool.join()
if __name__ == "__main__":
   main(sys.argv[1:])
