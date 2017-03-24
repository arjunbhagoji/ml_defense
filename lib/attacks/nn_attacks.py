import os

from os.path import dirname

import numpy as np
import theano
import theano.tensor as T
import scipy

from ..utils.theano_utils import *
from ..utils.attack_utils import *

script_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
rel_path_o = "output_data/"
abs_path_o = os.path.join(script_dir, rel_path_o)

#------------------------------------------------------------------------------#
def fsg(x_curr, y_curr, adv_x, dev_mag, batch_len, b_c, gradient, rd, rev):
    # Gradient w.r.t to input and current class
    delta_x = gradient(x_curr, y_curr)
    # Sign of gradient
    delta_x_sign = np.sign(delta_x)
    adv_x[b_c*batch_len:(b_c + 1)*batch_len] = x_curr + dev_mag*delta_x_sign
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def fg(x_curr, y_curr, adv_x, dev_mag, batch_len, b_c, gradient, rd, rev):
    # Gradient w.r.t to input and current class
    delta_x = gradient(x_curr, y_curr)
    # Calulating norm of gradient
    if rd == None or rev != None:
        delta_x_norm = np.linalg.norm(delta_x.reshape(batch_len, 784), axis=1)
    elif rd != None and rev == None:
        delta_x_norm = np.linalg.norm(delta_x.reshape(batch_len, rd), axis=1)

    # Perturbed images
    for i in range(batch_len):
        if delta_x_norm[i] == 0.0:
            adv_x[b_c*batch_len + i] = x_curr[i]
        else:
            adv_x[b_c*batch_len + i] = (x_curr[i]
                                        + dev_mag*(delta_x[i]/delta_x_norm[i]))
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Function to create adv. examples using the FSG method
def attack_wrapper(input_var, target_var, test_prediction, no_of_mags, X_test,
                   y_test, rd=None, rev=None):
    """
    Creates adversarial examples using the Fast Sign Gradient method. Prints
    output to a .txt file in '/outputs'. All 3 adversarial success counts
    are reported.
    : param input_var: symbolic input variable
    : param target_var: symbolic output variable
    : param test_prediction: model output on test data_utils
    : param no_of_mags: No. of epsilons to consider
    : param X_test: Test data
    : param y_test: Test data labels

    : return adv_x_all: list of adv. samples
    : return o_list: list of [acc., conf.] tested on adv. samples
    : return dev_list: list of used epsilons
    """

    test_len = len(X_test)
    height = X_test.shape[2]
    width = X_test.shape[3]
    n_features = height*width

    if rd == None or rev != None:
        adv_x_all = np.zeros((test_len, n_features, no_of_mags))
    elif rd != None or rev == None:
        adv_x_all = np.zeros((test_len, rd, no_of_mags))

    if rd == None or rev != None:
        scales = length_scales(X_test.reshape(test_len, n_features), y_test)
        adv_x = np.zeros((test_len, 1, height, width))
    elif rd != None and rev == None:
        scales = length_scales(X_test.reshape(test_len, rd), y_test)
        adv_x = np.zeros((test_len, 1, rd))

    validator, indexer, predictor, confidence = local_fns(input_var, target_var,
                                                          test_prediction)
    indices_c = indexer(X_test, y_test)
    i_c = np.where(indices_c == 1)[0]

    dev_list = np.linspace(0.01, 0.1, no_of_mags)

    gradient = grad_fn(input_var, target_var, test_prediction)

    o_list = []

    mag_count = 0
    for dev_mag in dev_list:
        # start_time=time.time()
        batch_len = 1000
        b_c = 0
        for batch in iterate_minibatches(X_test, y_test, batch_len,
                                         shuffle=False):
            x_curr, y_curr = batch
            fsg(x_curr, y_curr, adv_x, dev_mag, batch_len, b_c, gradient, rd,
                rev)
            # fg(x_curr, y_curr, adv_x, dev_mag, batch_len, b_c, gradient, rd, rev)
            b_c += 1
        # Accuracy vs. true labels. Confidence on mismatched predictions
        o_list.append(acc_calc_all(adv_x, y_test, X_test, i_c, validator,
                                   indexer, predictor, confidence))
        # Saving adversarial examples
        if rd == None or rev != None:
            adv_x_all[:, :, mag_count] = adv_x.reshape((test_len, n_features))
        elif rd != None and rev == None:
            adv_x_all[:, :, mag_count] = adv_x.reshape((test_len, rd))
        mag_count += 1

    return adv_x_all, o_list, dev_list
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def l_bfgs_attack(input_var, target_var, test_prediction, X_test, y_test,
                  rd=None, max_dev=None):
    # C_list=[0.7]
    C=0.7
    bfgs_iter=None
    trial_size=1000
    X_test=X_test[0:trial_size]
    y_test=y_test[0:trial_size]
    validator,indexer,predictor,confidence=local_fns(input_var,target_var,
                                                                test_prediction)
    deviation_list=[]
    # for C in C_list:
    count_wrong=0.0
    count_tot=0
    deviation=0.0
    magnitude=0.0
    count_correct=0.0
    adv_x=[]
    r_mat=[]
    x_used=[]
    y_used=[]
    o_list=[]
    for i in range(trial_size):
        print i
        def f(x):
            loss_curr,acc_curr=validator(X_curr+
                                            x.reshape((1,1,rd)),y_curr)
            return C*np.linalg.norm(x)+loss_curr

        y_old=y_test[i].reshape((1,))
        y_curr=y_test[np.random.randint(0,trial_size,1)]
        if y_old==y_curr:
            continue
        X_curr=X_test[i].reshape((1,1,rd))
        X_curr_flat=X_curr.reshape((rd))
        x_used.append(X_curr)
        y_used.append(y_old)
        ini_class=predictor(X_curr)
        #print ("Actual class is {}".format(y_old))
        # upper_limit=np.ones(rd)-X_curr_flat
        # lower_limit=np.zeros(rd)-X_curr_flat
        # bound=zip(lower_limit,upper_limit)
        x_0=np.zeros(rd)
        r,fval,info=scipy.optimize.fmin_l_bfgs_b(f,x_0,approx_grad=1)
                                                # bounds=bound)
        adv_x.append(X_curr+r.reshape((1,1,rd)))
        r_mat.append(r.reshape((1,1,rd)))
        # adv_x=X_adv_dr[count_tot,:].reshape((1,1,rd))
        prediction_curr=predictor(X_curr+r.reshape((1,1,rd)))
        # r=adv_x.reshape((rd))-X_curr_flat
        #Counting successful adversarial examples
        if ini_class[0]==y_test[i]:
            count_correct=count_correct+1
            # magnitude=magnitude+np.sqrt(np.sum(X_curr_flat**2)/rd)
            magnitude=magnitude+np.linalg.norm(X_curr_flat)
        if prediction_curr[0]!=ini_class[0] and ini_class[0]==y_test[i]:
            if max_dev!=None:
                if np.linalg.norm(r)<max_dev:
                    count_wrong=count_wrong+1
            elif max_dev==None:
                count_wrong=count_wrong+1
            # deviation=deviation+np.sqrt(np.sum(r**2)/rd)
            deviation=deviation+np.linalg.norm(r)
        deviation_list.append(np.linalg.norm(r))
        count_tot+=1
    adv_x=np.array(adv_x).reshape((count_tot,1,rd))
    y_used=np.array(y_used).reshape(count_tot)
    x_used=np.array(x_used).reshape((count_tot,1,rd))
    # indices_c=indexer(x_used,y_used)
    # i_c=np.where(indices_c==1)[0]
    #
    # o_list.append(acc_calc_all(adv_x,y_used,x_used,i_c,validator,indexer,
    #                                                 predictor,confidence))
    o_list.append([deviation/count_wrong,magnitude/count_correct,count_wrong/count_correct*100])
    # print o_list
    # print deviation_list
    print deviation/count_wrong
    print magnitude/count_correct
    print count_wrong/count_correct
    print count_correct
    return adv_x,o_list,deviation_list
