import numpy as np
import theano
import theano.tensor as T

from sklearn.decomposition import PCA

from ..utils.theano_utils import *

# Function to create adv. examples using the FSG method
def recons_defense(model_name,abs_path_o,input_var,target_var,test_prediction,
        adv_examples_test,rd,X_train,y_train,X_test,y_test,DEPTH=2,WIDTH=100):
    if model_name in ('mlp','custom'):
        plotfile=open(abs_path_o+'FSG_MNIST_nn_'+str(DEPTH)+'_'
                    +str(WIDTH)+'.txt','a')
    elif model_name=='cnn':
        plotfile=open(abs_path_o+'FSG_MNIST_cnn_papernot.txt','a')
    plotfile.write('\n')

    train_len=len(X_train)
    test_len=len(X_test)

    #Reshaping for PCA function
    PCA_in_train=X_train.reshape(train_len,784)
    PCA_in_test=X_test.reshape(test_len,784)

    print("Doing PCA over the training data")
    #Fitting the PCA model on training data
    pca=PCA(n_components=rd)
    pca_train=pca.fit(PCA_in_train)

    # Reconstructing training and test data
    X_train_dr=pca.transform(PCA_in_train)
    X_test_dr=pca.transform(PCA_in_test)
    X_train_rev=pca.inverse_transform(X_train_dr)
    X_train_rev=X_train_rev.reshape((train_len,1,28,28))
    X_test_rev=pca.inverse_transform(X_test_dr)
    X_test_rev=X_test_rev.reshape((test_len,1,28,28))

    # Evaluating on re-constructed inputs
    test_model_eval(input_var,target_var,test_prediction,X_test_rev,y_test)

    mag_count=0
    predictor=predict_fn(input_var, test_prediction)
    confidence= conf_fn(input_var,test_prediction)
    gradient=grad_fn(input_var, target_var, test_prediction)
    test_loss=loss_fn(test_prediction, target_var)
    test_acc=acc_fn(test_prediction, target_var)
    validator=val_fn(input_var, target_var, test_loss, test_acc)
    indexer=index_fn(test_prediction, input_var, target_var)
    indices_c=indexer(X_test_rev,y_test)
    i_c=np.where(indices_c==1)[0]
    for DEV_MAG in np.linspace(0.01,0.1,10):
        start_time=time.time()
        o_list=[]
        X_adv_dr=pca.transform(adv_examples_test[:,:,mag_count])
        recons_adv=(pca.inverse_transform(X_adv_dr)).reshape((test_len,1,28,28))
        # Accuracy vs. true labels. Confidence on mismatched predictions
        loss_w,acc_w=validator(recons_adv,y_test)
        c_w=100-acc_w*100
        indices_w=indexer(recons_adv,y_test)
        i_w=np.where(indices_w==0)[0]
        conf_w=np.float64(confidence(recons_adv[i_w]))
        o_list.extend([c_w,conf_w])
        #Accuracy vs. predicted labels
        loss_a,acc_a=validator(recons_adv,predictor(X_test_rev))
        c_a=100-acc_a*100
        indices_a=indexer(recons_adv,predictor(X_test_rev))
        i_a=np.where(indices_a==0)[0]
        conf_a=np.float64(confidence(recons_adv[i_a]))
        o_list.extend([c_a,conf_a])
        # Accuracy for adv. examples generated from correctly classified
        # examples
        loss_p,acc_p=validator(recons_adv[i_c],y_test[i_c])
        c_p=100-acc_p*100
        indices_p=indexer(recons_adv[i_c],y_test[i_c])
        i_p=np.where(indices_p==0)[0]
        conf_p=np.float64(confidence(recons_adv[i_c][i_p]))
        o_list.extend([c_p,conf_p])

        recons_adv=recons_adv.reshape((10000,784))
        adv_examples_test[:,:,mag_count]=recons_adv

        plotfile.write(str(rd)+",")
        for item in o_list[0:-1]:
            plotfile.write(str.format("{0:.3f}",item)+",")
        plotfile.write(str.format("{0:.3f}",o_list[-1]))
        plotfile.write("\n")
        mag_count=mag_count+1
    plotfile.close()
