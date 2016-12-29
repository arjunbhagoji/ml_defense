import numpy as np
import theano
import theano.tensor as T

from ..utils.theano_utils import *

# Function to create adv. examples using the FSG method
def no_pca_attack(*args):
    model_name=args[0]
    # global network
    # network=args[1]
    abs_path_o=args[1]
    if model_name in ('mlp','custom'):
        DEPTH=args[2]
        WIDTH=args[3]
        plotfile=open(abs_path_o+'FSG_MNIST_data_hidden_'+str(DEPTH)+'_'
                    +str(WIDTH)+'_.txt','a')
    elif model_name=='cnn':
        plotfile=open(abs_path_o+'FSG_MNIST_data_cnn_papernot.txt','a')
    # plotfile.write('rd,Dev.,Wrong,C.,n_C.,Adv.,C.,n_C.,Pure,C.,n_C.,Train \n')
    plotfile.close()
    input_var=args[4]
    target_var=args[5]
    test_prediction=args[6]
    adv_examples_test=args[7]
    X_test=args[8]
    y_test=args[9]
    start_time=time.time()

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
        # current_predict=predictor(X_test)
        # current_conf=confidence(X_test)
        # Gradient w.r.t to input and current class
        delta_x=gradient(X_test,y_test)
        # Sign of gradient
        delta_x_sign=np.sign(delta_x)
        #delta_x_sign=delta_x_sign/np.linalg.norm((delta_x_sign))
        #Perturbed images
        adv_x=X_test+DEV_MAG*delta_x_sign
        # Accuracy vs. true labels. Confidence on mismatched predictions
        loss_w,acc_w=validator(adv_x,y_test)
        indices_w=indexer(adv_x,y_test)
        i_w=np.where(indices_w==0)[0]
        conf_w=confidence(adv_x[i_w])
        #Accuracy vs. predicted labels
        loss_a,acc_a=validator(adv_x,predictor(X_test))
        indices_a=indexer(adv_x,predictor(X_test))
        i_a=np.where(indices_a==0)[0]
        conf_a=confidence(adv_x[i_a])
        # Accuracy for adv. examples generated from correctly classified
        # examples
        loss_p,acc_p=validator(adv_x[i_c],y_test[i_c])
        indices_p=indexer(adv_x[i_c],y_test[i_c])
        i_p=np.where(indices_p==0)[0]
        conf_p=confidence(adv_x[i_p])
        # print indices.type
        # count_wrong=count_wrong+ca
        adv_predict=predictor(adv_x)
        # conf_wrong=conf_wrong+adv_conf
        adv_x=adv_x.reshape((10000,784))
        adv_examples_test[:,:,mag_count]=adv_x
        # adv_examples_test[b_count*1000:(b_count+1)*1000,:,mag_count]=adv_x
        # b_count=b_count+1
        print 100-acc_w*100
        print conf_w
        print 100-acc_a*100
        print conf_a
        print 100-acc_p*100
        print conf_p
        # plotfile=open(abs_path_o+'FSG_MNIST_data_hidden_'+str(DEPTH)+'_'
        #                 +str(WIDTH)+'_'+'.txt','a')
        # adv_acc=100-count_wrong/b_count*100
        # adv_acc_2=100-count_adv/b_count*100
        # adv_count=adv_acc*test_len/100
        # c_count=test_len-adv_count
        # ini_count=test_acc/test_batches*test_len
        # # plotfile=open(abs_path_o+'FSG_MNIST_data_cnn_papernot.txt','a')
        # print("Deviation {} took {:.3f}s".format(
        #     DEV_MAG, time.time() - start_time))
        # plotfile.write('no_dr'+","+str(DEV_MAG)+","+
        #                 str.format("{0:.3f}",adv_acc)+","+
        #                 str.format("{0:.3f}",conf_wrong/adv_count)+","+
        #                 str.format("{0:.3f}",conf_correct/c_count)+","+
        #                 str(count_adv/test_len*100)+","+
        #                 str.format("{0:.3f}",conf_adv/count_adv)+","+
        #                 str.format("{0:.3f}",conf_n_adv/(test_len-count_adv))+","+
        #                 str(count_abs_wrong/test_len*100)+","+
        #                 str.format("{0:.3f}",conf_abs/count_abs_wrong)+","+
        #                 str.format("{0:.3f}",conf_abs_c/(ini_count-count_abs_wrong))+","+
        #                 str(1)+"\n")
        # plotfile.close()
        mag_count=mag_count+1
