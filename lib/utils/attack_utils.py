import numpy as np
from scipy.misc import imsave

from ..utils.theano_utils import *
from ..utils.lasagne_utils import *
from ..utils.data_utils import *

#------------------------------------------------------------------------------#
def class_means(X, y):

    """Return a list of means of each class in (X,y)"""

    classes = np.unique(y)
    no_of_classes = len(classes)
    means = []
    class_frac = []
    for item in classes:
        indices = np.where(y == item)[0]
        class_items = X[indices, :]
        class_frac.append(float(len(class_items))/float(len(X)))
        mean = np.mean(class_items, axis=0)
        means.append(mean)
    return means, class_frac
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def length_scales(X, y):

    """Find distances from each class mean to means of the other classes"""

    means, class_frac = class_means(X, y)
    no_of_classes = len(means)
    scales = []
    for i in range(no_of_classes):
        mean_diff = 0.0
        curr_mean = means[i]
        mean_not_i = 0.0
        curr_frac = class_frac[i]
        for j in range(no_of_classes):
            if i == j: continue
            else:
                mean_not_i = mean_not_i + means[j]
        mean_diff = curr_frac*curr_mean - (1-curr_frac)*(mean_not_i/(no_of_classes-1))
        scales.append(np.linalg.norm(mean_diff))
    return scales
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def naive_untargeted_attack(X, y):

    """
    Returns a minimum distance required to move a sample to a different class
    """

    scales = length_scales(X, y)
    print scales
    data_len = len(X)
    classes = np.unique(y)
    distances = []
    for i in range(100):
        curr_data = X[i,:]
        curr_distances = []
        for j in range(100):
            if i == j: continue
            else:
                # if y[i]==y[j]:
                #     continue
                if y[i] != y[j]:
                    data_diff = curr_data - X[j, :]
                    data_dist = np.linalg.norm(data_diff)
                    print data_dist
                    curr_distances.append(data_dist/scales[y[i]])
        distances.append(min(curr_distances))
    return distances
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def local_fns(input_var, target_var, test_prediction):

    """Returns necessary Theano functions"""

    predictor = predict_fn(input_var, test_prediction)
    confidence = conf_fn(input_var, test_prediction)
    gradient = grad_fn(input_var, target_var, test_prediction)
    test_loss = loss_fn(test_prediction, target_var)
    test_acc = acc_fn(test_prediction, target_var)
    validator = val_fn(input_var, target_var, test_loss, test_acc)
    indexer = index_fn(test_prediction, input_var, target_var)
    return validator, indexer, predictor, confidence
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def acc_calc(X_adv, y, validator, indexer, confidence):

    """
    Calculate attack success and average confidence on (X_adv, y) where X_adv is
    adv. examples and y is true labels
    """

    loss_i, acc_i = validator(X_adv, y)
    c_i = 100 - acc_i*100
    indices_i = indexer(X_adv, y)
    i_i = np.where(indices_i == 0)[0]
    conf_i = np.float64(confidence(X_adv[i_i]))
    return [c_i, conf_i]
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def acc_calc_all(X_adv, y_test, X_test_mod, i_c, validator, indexer, predictor,
                 confidence):

    """
    Calculate attack success and average confidence based w.r.t. true labels,
    originally predicted labels and correctly classified labels, respectively
    """

    o_list = []
    # Accuracy vs. true labels. Confidence on mismatched predictions
    c_w, conf_w = acc_calc(X_adv, y_test, validator, indexer, confidence)
    o_list.extend([c_w, conf_w])
    # Accuracy vs. predicted labels
    c_a, conf_a = acc_calc(X_adv, predictor(X_test_mod), validator, indexer,
                           confidence)
    o_list.extend([c_a, conf_a])
    # Accuracy for adv. examples generated from correctly classified examples
    c_p, conf_p = acc_calc(X_adv[i_c], y_test[i_c], validator, indexer,
                           confidence)
    o_list.extend([c_p, conf_p])
    return o_list
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def avg_grad_calc(input_var, target_var, test_prediction, X_test, y_test):

    """Return average gradient of model's loss evaluated on (X_test, y_test)"""

    gradient = grad_fn(input_var, target_var, test_prediction)
    delta_x = gradient(X_test, y_test)
    delta_x_abs = np.abs(delta_x)
    delta_x_avg_abs = np.mean(delta_x_abs, axis=0)
    return delta_x_avg_abs
#------------------------------------------------------------------------------#
