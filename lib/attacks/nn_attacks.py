import os

from os.path import dirname

import numpy as np
import theano
import theano.tensor as T

from ..utils.theano_utils import *
from ..utils.attack_utils import *

script_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
rel_path_o="output_data/"
abs_path_o=os.path.join(script_dir,rel_path_o)

# Function to create adv. examples using the FSG method
def fsg_attack(model_dict,input_var,target_var,test_prediction,
                    no_of_mags,X_test,y_test,print_flag=1,rd=None):
    """
    Creates adversarial examples using the Fast Sign Gradient method. Prints
    output to a .txt file in '/outputs'. All 3 adversarial success counts
    are reported.
    : param model_name: name of the model
    : param abs_path_o: path to output
    : param input_var: symbolic input variable
    : param target_var: symbolic output variable
    : param test_prediction: model output on test data_utils
    : param no_of_mags: No. of epsilons to consider
    : param X_test: Test data
    : param y_test: Test data labels
    """
    model_name=model_dict['model_name']

    if print_flag==1:
        if model_name in ('mlp','custom'):
            depth=model_dict['depth']
            width=model_dict['width']
            if rd==None:
                plotfile=open(abs_path_o+'FSG_mod_MNIST_nn_'+str(depth)+'_'
                            +str(width)+'.txt','a')
            elif rd!=None:
                plotfile=open(abs_path_o+'FSG_mod_MNIST_nn_'+str(depth)+'_'
                            +str(width)+'_strategic.txt','a')
        elif model_name=='cnn':
            plotfile=open(abs_path_o+'FSG_MNIST_cnn_papernot.txt','a')
        plotfile.write('eps,c_w,conf_w,c_a,conf_a,c_p,conf_p \n')
        if rd==None:
            plotfile.write('no_defense'+'\n')
        elif rd!=None:
            plotfile.write(str(rd)+'\n')

    test_len=len(X_test)
    if rd==None:
        adv_x_all=np.zeros((test_len,784,no_of_mags))
    elif rd!=None:
        adv_x_all=np.zeros((test_len,rd,no_of_mags))

    mag_count=0
    predictor=predict_fn(input_var, test_prediction)
    confidence= conf_fn(input_var,test_prediction)
    gradient=grad_fn(input_var, target_var, test_prediction)
    test_loss=loss_fn(test_prediction, target_var)
    test_acc=acc_fn(test_prediction, target_var)
    validator=val_fn(input_var, target_var, test_loss, test_acc)
    indexer=index_fn(test_prediction, input_var, target_var)
    indices_c=indexer(X_test,y_test)
    i_c=np.where(indices_c==1)[0]
    if rd==None:
        scales=length_scales(X_test.reshape(test_len,784), y_test)
        adv_x=np.zeros((10000,1,28,28))
    elif rd!=None:
        scales=length_scales(X_test.reshape(test_len,rd), y_test)
        adv_x=np.zeros((10000,1,rd))
    for dev_mag in np.linspace(0.01,1.0,50):
        start_time=time.time()
        o_list=[]
        # Gradient w.r.t to input and current class
        delta_x=gradient(X_test,y_test)
        # Norm of gradient
        delta_x_norm=np.linalg.norm(delta_x.reshape(test_len,rd),axis=1)
        # Sign of gradient
        delta_x_sign=np.sign(delta_x)
        #delta_x_sign=delta_x_sign/np.linalg.norm((delta_x_sign))
        #Perturbed images
        for i in range(test_len):
            adv_x[i]=X_test[i]+dev_mag*scales[y_test[i]]*(delta_x[i]/delta_x_norm[i])
        # Accuracy vs. true labels. Confidence on mismatched predictions
        loss_w,acc_w=validator(adv_x,y_test)
        c_w=100-acc_w*100
        indices_w=indexer(adv_x,y_test)
        i_w=np.where(indices_w==0)[0]
        conf_w=np.float64(confidence(adv_x[i_w]))
        o_list.extend([c_w,conf_w])
        #Accuracy vs. predicted labels
        loss_a,acc_a=validator(adv_x,predictor(X_test))
        c_a=100-acc_a*100
        indices_a=indexer(adv_x,predictor(X_test))
        i_a=np.where(indices_a==0)[0]
        conf_a=np.float64(confidence(adv_x[i_a]))
        o_list.extend([c_a,conf_a])
        # Accuracy for adv. examples generated from correctly classified
        # examples
        loss_p,acc_p=validator(adv_x[i_c],y_test[i_c])
        c_p=100-acc_p*100
        indices_p=indexer(adv_x[i_c],y_test[i_c])
        # print indices_p
        i_p=np.where(indices_p==0)[0]
        conf_p=np.float64(confidence(adv_x[i_c][i_p]))
        o_list.extend([c_p,conf_p])

        # Saving adversarial examples
        if rd==None:
            adv_x_all[:,:,mag_count]=adv_x.reshape((test_len,784))
        elif rd!=None:
            adv_x_all[:,:,mag_count]=adv_x.reshape((test_len,rd))

        mag_count=mag_count+1

        # print("Deviation {} took {:.3f}s".format(
        #     DEV_MAG, time.time() - start_time))
        if print_flag==1:
            plotfile.write(str(dev_mag)+',')
            for item in o_list[0:-1]:
                plotfile.write(str.format("{0:.3f}",item)+",")
            plotfile.write(str.format("{0:.3f}",o_list[-1]))
            plotfile.write("\n")

    if print_flag==1:
        plotfile.close()

    return adv_x_all
