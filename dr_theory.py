import os
import numpy as np
import theano
import theano.tensor as T

from lib.utils.theano_utils import *
from lib.utils.lasagne_utils import *
from lib.utils.data_utils import *
from lib.utils.dr_utils import *
from lib.utils.attack_utils import *
from lib.utils.plot_utils import *

def gradient_calc(rd,X_train,y_train,X_test,y_test,X_val,y_val):

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
        test_model_eval(model_dict,input_var,target_var,test_prediction,
                            X_test_dr,y_test,rd)

        var_array=np.sqrt(np.var(X_train_dr,axis=0))
        var_list=list(var_array)
        gradient_comp=avg_grad_calc(input_var,target_var,test_prediction,
                                    X_test_dr,y_test)
        gradient_list=list(gradient_comp)

        return zip(var_list, gradient_list)



def main():

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    rd_list=[784,331,200,100,90,80,70,60,50,40,30,20,10]
    no_of_dims=len(rd_list)

    gradient_var_list=[]

    for rd in rd_list:
        gradient_var_list.append(gradient_calc(rd,X_train,y_train,X_test,y_test,
                                    X_val,y_val))
    mag_var_scatter(gradient_var_list,no_of_dims)




if __name__ == "__main__":
    main()
