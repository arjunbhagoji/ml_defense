import os

from os.path import dirname

import numpy as np
import theano
import theano.tensor as T
import scipy

from ..utils.theano_utils import *
from ..utils.attack_utils import *
from ..utils.data_utils import *

#------------------------------------------------------------------------------#
def fgs(x_curr, y_curr, adv_x, dev_mag, b_c, gradient, rd, rev):

    """
    Performs Fast Sign Gradient attack and put perturbed examples in <adv_x>.

    Parameters
    ----------
    x_curr   : a batch of input samples
    y_curr   : a batch of input labels
    adv_x    : an array to save attack samples
    dev_mag  : perturbation magnitude
    b_c      : bactch count
    gradient : gradient function
    rd       : dimension reduction flag
    rev      : inverse transform flag
    """

    batch_len = x_curr.shape[0]
    # Gradient w.r.t to input and current class
    delta_x = gradient(x_curr, y_curr)
    # Sign of gradient
    delta_x_sign = np.sign(delta_x)
    if rd == None or rev != None: # Clipping if in pixel space
        adv_x[b_c*batch_len:(b_c + 1)*batch_len] = np.clip(x_curr +
                                                   dev_mag*delta_x_sign, 0 , 1)
    elif rd !=None and rev ==None:
        adv_x[b_c*batch_len:(b_c + 1)*batch_len] = x_curr + dev_mag*delta_x_sign
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def fg(x_curr, y_curr, adv_x, dev_mag, b_c, gradient, rd, rev):

    """
    Performs Fast Gradient attack and put perturbed examples in <adv_x>.

    Parameters
    ----------
    x_curr   : a batch of input samples
    y_curr   : a batch of input labels
    adv_x    : an array to save attack samples
    dev_mag  : perturbation magnitude
    b_c      : bactch count
    gradient : gradient function
    rd       : dimension reduction flag
    rev      : inverse transform flag
    """

    batch_len = x_curr.shape[0]
    # Gradient w.r.t to input and current class
    delta_x = gradient(x_curr, y_curr)
    # Calulating norm of gradient
    channels = x_curr.shape[1]
    if rd == None or rev != None:
        n_features = x_curr.shape[2]*x_curr.shape[3]
        delta_x_norm = np.linalg.norm(delta_x.reshape(batch_len, channels,
                                      n_features), axis=2)
    elif rd != None and rev == None:
        delta_x_norm = np.linalg.norm(delta_x.reshape(batch_len, channels, rd),
                                      axis=2)

    # Perturbed images
    for i in range(batch_len):
        for j in range(channels):
            if delta_x_norm[i, j] == 0.0:
                adv_x[b_c*batch_len + i, j] = x_curr[i, j]
            else:
                if rd == None or rev != None: # Clipping in pixel space
                    adv_x[b_c*batch_len + i, j] = np.clip(x_curr[i, j] + dev_mag
                                    *(delta_x[i, j]/delta_x_norm[i, j]), 0, 1)
                elif rd != None and rev == None:
                    adv_x[b_c*batch_len + i, j] = x_curr[i, j] + dev_mag*(delta_x[i, j]/delta_x_norm[i, j])
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Function to create adv. examples using the FSG method
def attack_wrapper(model_dict, data_dict, input_var, target_var, test_prediction,
                   dev_list, X_test, y_test, rd=None, rev=None):

    """
    Creates adversarial examples using the Fast Sign Gradient method. Prints
    output to a .txt file in '/outputs'. All 3 adversarial success counts
    are reported.
    : param input_var: symbolic input variable
    : param target_var: symbolic output variable
    : param test_prediction: model output on test data_utils
    : param dev_list: list of deviations (mags)
    : param X_test: Test data
    : param y_test: Test data labels

    : return adv_x_all: list of adv. samples
    : return o_list: list of [acc., conf.] tested on adv. samples
    : return dev_list: list of used epsilons
    """

    adv_len = data_dict['test_len']
    no_of_dim = data_dict['no_of_dim']
    channels = data_dict['channels']
    no_of_features = data_dict['no_of_features']

    n_mags = len(dev_list)
    # Creating array to store adversarial samples
    adv_x_all = np.zeros((adv_len, no_of_features, n_mags))

    validator, indexer, predictor, confidence = local_fns(input_var, target_var,
                                                          test_prediction)
    indices_c = indexer(X_test, y_test)
    i_c = np.where(indices_c == 1)[0]

    gradient = grad_fn(input_var, target_var, test_prediction)

    # Creating array of zeros to store adversarial samples
    if no_of_dim == 2: adv_x = np.zeros((adv_len, no_of_features))
    elif no_of_dim == 3:
        features = data_dict['features_per_c']
        adv_x = np.zeros((adv_len, channels, features))
    elif no_of_dim ==4:
        height = data_dict['height']
        width = data_dict['width']
        adv_x = np.zeros((adv_len, channels, height, width))

    o_list = []
    mag_count = 0
    for dev_mag in dev_list:
        adv_x.fill(0)
        start_time = time.time()
        batch_len = 1000
        b_c = 0
        for batch in iterate_minibatches(X_test, y_test, batch_len):
            x_curr, y_curr = batch
            if model_dict['attack'] == 'fgs':
                fgs(x_curr, y_curr, adv_x, dev_mag, b_c, gradient, rd, rev)
            elif model_dict['attack'] == 'fg':
                fg(x_curr, y_curr, adv_x, dev_mag, b_c, gradient, rd, rev)
            b_c += 1
        # Accuracy vs. true labels. Confidence on mismatched predictions
        o_list.append(acc_calc_all(adv_x, y_test, X_test, i_c, validator,
                                   indexer, predictor, confidence))
        # Saving adversarial examples
        adv_x_all[:,:,mag_count] = adv_x.reshape((adv_len, no_of_features))
        mag_count += 1
        print('Finished adv. samples with magnitude {:.3f}: took {:.3f}s'
              .format(dev_mag, time.time() - start_time))

    return adv_x_all, o_list
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
