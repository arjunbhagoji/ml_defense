import sys
import os
import numpy as np
from os.path import dirname
from sklearn import svm
from sklearn.externals import joblib

#------------------------------------------------------------------------------#
def resolve_path_m(model_dict):

    """
    Resolve absolute paths of models for different datasets

    Parameters
    ----------
    model_dict : dictionary
                 contains model's parameters

    Returns
    -------
    absolute path to models directory
    """

    dataset = model_dict['dataset']
    channels = model_dict['channels']
    script_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
    rel_path_m = 'svm_models/' + dataset
    if dataset == 'GTSRB': rel_path_m += str(channels)
    abs_path_m = os.path.join(script_dir, rel_path_m + '/')
    if not os.path.exists(abs_path_m): os.makedirs(abs_path_m)
    return abs_path_m
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def resolve_path_o(model_dict):

    """
    Resolve absolute paths of output data for different datasets

    Parameters
    ----------
    model_dict : dictionary
                 contains model's parameters

    Returns
    -------
    absolute path to output directory
    """
    
    dataset = model_dict['dataset']
    channels = model_dict['channels']
    script_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
    rel_path_o = 'svm_output_data/' + dataset
    if dataset == 'GTSRB': rel_path_o += str(channels)
    abs_path_o = os.path.join(script_dir, rel_path_o + '/')
    if not os.path.exists(abs_path_o): os.makedirs(abs_path_o)
    return abs_path_o
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def get_model_name(model_dict, rd=None):

    """
    Helper function to get model name from <model_dict> and <rd>
    """

    if rd == None:
        model_name = 'svm_cls{}_{}_C{}'.format(model_dict['classes'],
                                               model_dict['penalty'],
                                               model_dict['penconst'])
    else:
        model_name = 'svm_cls{}_{}{}_{}_C{}'.format(model_dict['classes'],
                                                    model_dict['dr'], rd,
                                                    model_dict['penalty'],
                                                    model_dict['penconst'])
    return model_name
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_loader(model_dict, rd=None):

    """
    Returns a classifier object if it already exists. Returns None, otherwise.
    """

    abs_path_m = resolve_path_m(model_dict)
    try:
        clf = joblib.load(abs_path_m + get_model_name(model_dict, rd) + '.pkl')
    except:
        clf = None
    return clf
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_trainer(model_dict, X_train, y_train, rd=None):

    """
    Trains and returns SVM. Also save SVM to file.
    """

    abs_path_m = resolve_path_m(model_dict)
    C = model_dict['penconst']
    penalty = model_dict['penalty']
    clf = svm.LinearSVC(C=C, penalty=penalty)
    clf.fit(X_train, y_train)
    # Save model
    joblib.dump(clf, abs_path_m + get_model_name(model_dict, rd) + '.pkl')
    return clf
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_creator(model_dict, X_train, y_train, rd=None):

    """
    Returns a SVM classifier
    """

    # Load model based on model_dict
    clf = model_loader(model_dict, rd)
    # If model does not exist, train a new SVM
    if clf == None:
        clf = model_trainer(model_dict, X_train, y_train, rd)
    return clf
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def model_tester(model_dict, clf, X_test, y_test):

    """
    Calculate model's accuracy and average normalized distance from correctly
    classified samples to separating hyperplane of corresponding class
    """

    predicted_labels = clf.predict(X_test)
    # Magnitude of weight vectors for each class
    norm = np.linalg.norm(clf.coef_, axis=1)
    # Distance from each sample to hyperplane of each class
    dist = clf.decision_function(X_test)

    test_len = len(X_test)
    sum_dist = 0
    n_correct = 0
    for i in range(test_len):
        if predicted_labels[i] == y_test[i]:
            n_correct += 1
            # Sum normalized distance to sept. hyperplane
            sum_dist += dist[i,y_test[i]] / norm[y_test[i]]

    # Resolve path to utility output file
    abs_path_o = resolve_path_o(model_dict)
    fname = 'utility_' + get_model_name(model_dict)
    plotfile = open(abs_path_o + fname + '.txt', 'a')
    # Format: <dimensions> <accuracy> <avg. dist.>
    plotfile.write('{} {:.2f} {:.3f}\n'.format(X_test.shape[1],
                                               float(n_correct)/test_len*100,
                                               sum_dist/n_correct))
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def acc_calc_all(clf, X_adv, y_test, y_ini):

    """
    Return attack success rate on <clf> based on initially correctly predicted
    samples
    """

    o_list = []
    y_adv = clf.predict(X_adv)
    # Accuracy vs. true labels
    atk_success = (y_adv != y_test)
    acc_t = np.sum(atk_success)/float(len(X_adv))
    o_list.append(acc_t)
    # Accuracy vs. predicted labels
    atk_success = (y_adv != y_ini)
    acc_p = np.sum(atk_success)/float(len(X_adv))
    o_list.append(acc_p)
    # Accuracy for adv. examples generated from correctly classified examples
    atk_success = np.logical_and((y_adv != y_ini), (y_ini == y_test))
    acc_c = np.sum(atk_success)/float(len(X_adv))
    o_list.append(acc_c)

    return o_list
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def file_create(model_dict, strat_flag=None):

    """
    Creates and returns a file descriptor, named corresponding to model,
    attack type, strat, and rev
    """

    # Resolve absolute path to output directory
    abs_path_o = resolve_path_o(model_dict)
    fname = get_model_name(model_dict)
    if strat_flag != None: fname += '_strat'
    plotfile = open(abs_path_o + fname + '.txt', 'a')
    return plotfile
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def print_output(model_dict, output_list, dev_list, dim, strat_flag=None):

    """
    Creates an output file reporting accuracy and confidence of attack
    """

    plotfile = file_create(model_dict, strat_flag)
    plotfile.write('\\\\small{{{}}}\n'.format(dim))
    for i in range(len(dev_list)):
        plotfile.write('{0:.3f}'.format(dev_list[i]))
        for item in output_list[i]:
            plotfile.write(' {0:.3f}'.format(item))
        plotfile.write('\n')
    plotfile.close()
#------------------------------------------------------------------------------#
