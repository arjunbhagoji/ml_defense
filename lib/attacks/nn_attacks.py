import numpy as np
import theano
import theano.tensor as T

from ..utils.theano_utils import *

# Function to create adv. examples using the FSG method
def fsg_attack(model_name,abs_path_o,input_var,target_var,test_prediction
                    ,adv_examples_test,X_test,y_test,DEPTH=2,WIDTH=100):
    if model_name in ('mlp','custom'):
        plotfile=open(abs_path_o+'FSG_MNIST_nn_'+str(DEPTH)+'_'
                    +str(WIDTH)+'.txt','a')
    elif model_name=='cnn':
        plotfile=open(abs_path_o+'FSG_MNIST_cnn_papernot.txt','a')
    plotfile.write('rd,Dev_mag,c_w,conf_w,c_a,conf_a,c_p,conf_p \n')


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
    for DEV_MAG in np.linspace(0.01,0.1,10):
        start_time=time.time()
        o_list=[]
        # Gradient w.r.t to input and current class
        delta_x=gradient(X_test,y_test)
        # Sign of gradient
        delta_x_sign=np.sign(delta_x)
        #delta_x_sign=delta_x_sign/np.linalg.norm((delta_x_sign))
        #Perturbed images
        adv_x=X_test+DEV_MAG*delta_x_sign
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
        adv_x=adv_x.reshape((10000,784))
        adv_examples_test[:,:,mag_count]=adv_x

        print("Deviation {} took {:.3f}s".format(
            DEV_MAG, time.time() - start_time))

        plotfile.write('N.A.,')
        for item in o_list[0:-1]:
            plotfile.write(str.format("{0:.3f}",item)+",")
        plotfile.write(str.format("{0:.3f}",o_list[-1]))
        plotfile.write("\n")
        mag_count=mag_count+1
    plotfile.close()
