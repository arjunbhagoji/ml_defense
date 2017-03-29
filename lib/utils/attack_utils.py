import numpy as np
from scipy.misc import imsave

from ..utils.theano_utils import *
from ..utils.lasagne_utils import *
from ..utils.data_utils import *

#------------------------------------------------------------------------------#
def class_means(X, y):
    classes = np.unique(y)
    no_of_classes = len(classes)
    means = []
    for item in classes:
        indices = np.where(y == item)[0]
        class_items = X[indices, :]
        mean = np.mean(class_items, axis=0)
        means.append(mean)
    return means
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def length_scales(X, y):
    means = class_means(X, y)
    no_of_classes = len(means)
    scales = []
    for i in range(no_of_classes):
        curr_mean = means[i]
        curr_scales = []
        for j in range(no_of_classes):
            if i == j: continue
            else:
                mean_diff = curr_mean - means[j]
                curr_scales.append(np.linalg.norm(mean_diff))
        scales.append(np.amin(curr_scales))
    return scales
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def naive_untargeted_attack(X, y):
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
    gradient = grad_fn(input_var, target_var, test_prediction)
    delta_x = gradient(X_test, y_test)
    delta_x_abs = np.abs(delta_x)
    delta_x_avg_abs = np.mean(delta_x_abs, axis=0)
    return delta_x_avg_abs
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def file_create(model_dict, is_defense, rev=None, strat_flag=None):
    """
    Creates and returns a file descriptor, named corresponding to model,
    attack type, strat, and rev
    """
    # Resolve absolute path to output directory
    abs_path_o = resolve_path_o(model_dict)

    model_name = model_dict['model_name']
    fname = model_dict['attack']
    # MLP model
    if model_name in ('mlp', 'custom'):
        depth = model_dict['depth']
        width = model_dict['width']
        fname += '_nn_{}_{}'.format(depth, width)
    # CNN model
    elif model_name == 'cnn':
        fname += '_cnn_papernot'

    if strat_flag != None: fname += '_strat'
    if rev != None: fname += '_rev'
    if is_defense: fname += ('_' + model_dict['defense'])
    plotfile = open(abs_path_o + fname + '.txt', 'a')
    return plotfile
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def print_output(model_dict, output_list, dev_list, is_defense=False, rd=None,
                 rev=None, strat_flag=None):
    """
    Creates an output file reporting accuracy and confidence of attack
    """
    plotfile = file_create(model_dict, is_defense, rev, strat_flag)
    plotfile.write('Reduced dimensions: {}\n'.format(rd))
    plotfile.write('Mag.   True            Predicted       Correct Class\n')
    for i in range(len(dev_list)):
        plotfile.write('{0:<7.3f}'.format(dev_list[i]))
        for item in output_list[i]:
            plotfile.write('{0:<8.3f}'.format(item))
        plotfile.write("\n")
    plotfile.close()
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Saves first 10 images from the test set and their adv. samples
def save_images(model_dict, X_test, adv_x, dev_mag):
    indices = range(10)
    channels = X_test.shape[1]
    height = X_test.shape[2]
    width = X_test.shape[3]
    atk = model_dict['attack']
    dataset = model_dict['dataset']
    for i in indices:
        if channels == 1:
            adv = adv_x[i].reshape((height, width))
            orig = X_test[i].reshape((height, width))
        else:
            adv = adv_x[i].swapaxes(0, 2).swapaxes(0, 1)
            orig = X_test[i].swapaxes(0, 2).swapaxes(0, 1)
        imsave('{}_{}_{}_mag{}.jpg'.format(atk, dataset, i, dev_mag), adv)
        imsave('{}_{}_{}_orig.jpg'.format(atk, dataset, i), orig)
#------------------------------------------------------------------------------#
