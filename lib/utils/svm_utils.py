import sys
import os
import argparse
import time
import numpy as np
from os.path import dirname
from sklearn import svm
from sklearn.externals import joblib
from matplotlib import image as img

from lib.utils.dr_utils import invert_dr, gradient_transform
from lib.attacks.svm_attacks import min_dist_calc

#------------------------------------------------------------------------------#


def resolve_path():
    return dirname(dirname(dirname(os.path.abspath(__file__))))
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
    script_dir = resolve_path()
    rel_path_m = 'svm_models/' + dataset
    if dataset == 'GTSRB':
        rel_path_m += str(channels)
    abs_path_m = os.path.join(script_dir, rel_path_m + '/')
    if not os.path.exists(abs_path_m):
        os.makedirs(abs_path_m)
    return abs_path_m
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
    script_dir = resolve_path()
    rel_path_o = 'svm_output_data/' + dataset
    if dataset == 'GTSRB':
        rel_path_o += str(channels)
    abs_path_o = os.path.join(script_dir, rel_path_o + '/')
    if not os.path.exists(abs_path_o):
        os.makedirs(abs_path_o)
    return abs_path_o
#------------------------------------------------------------------------------#


def resolve_path_v(model_dict):
    """
    Resolve absolute paths of visual data for different datasets

    Parameters
    ----------
    model_dict : dictionary
                 contains model's parameters

    Returns
    -------
    absolute path to output directory
    """

    model_name = get_svm_model_name(model_dict)
    dataset = model_dict['dataset']
    channels = model_dict['channels']
    script_dir = resolve_path()
    rel_path_v = 'svm_visual_data/' + dataset
    if dataset == 'GTSRB':
        rel_path_v += str(channels)
    rel_path_v += '/' + model_name
    abs_path_v = os.path.join(script_dir, rel_path_v + '/')
    if not os.path.exists(abs_path_v):
        os.makedirs(abs_path_v)
    return abs_path_v
#------------------------------------------------------------------------------#


def svm_model_dict_create():
    """
    Parse arguments to strategic_svm.py and create model_dict containing the
    parameters
    """

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-st',
        '--svm_type',
        default='linear',
        type=str,
        help='Specify type of SVM to be used (default: linear)')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='Specify dataset (default: MNIST)')
    parser.add_argument(
        '-c',
        '--channels',
        default=1,
        type=int,
        help='Specify number of input channels (1 (default) or 3)')
    parser.add_argument(
        '--two_classes',
        action='store_true',
        help='Train SVM on two classes instead of all available classes')
    parser.add_argument('-dr', '--dim_red', default='pca', type=str,
                        help='Specify dimension reduction (default: pca)')
    parser.add_argument(
        '--rev',
        action='store_true',
        help='Train SVM and attack on DR sample reverted to original space')
    parser.add_argument('-C', '--penconst', default=1.0, type=float,
                        help='Specify penalty parameter C (default: 1.0)')
    parser.add_argument(
        '-p',
        '--penalty',
        default='l2',
        type=str,
        help='Specify norm to use in penalization (l1 or l2 (default))')
    parser.add_argument(
        '-pp',
        '--preprocess',
        default=None,
        type=str,
        help='Specify preprocessing on dataset (std, whiten, antiwhiten(*)) \
             (default: None) \n (*) is degree of covariance (>= -1)')

    args = parser.parse_args()

    # Create and update model_dict
    model_dict = {}
    model_dict.update({'svm_type': args.svm_type})
    model_dict.update({'dataset': args.dataset})
    model_dict.update({'channels': args.channels})
    model_dict.update({'dim_red': args.dim_red})
    model_dict.update({'penconst': args.penconst})
    model_dict.update({'penalty': args.penalty})
    model_dict.update({'preprocess': args.preprocess})
    if args.rev:
        model_dict.update({'rev': 1})
    else:
        model_dict.update({'rev': None})
    if args.two_classes:
        model_dict.update({'classes': 2})
    else:
        # TODO: preferrably put this somehere else
        dataset = model_dict['dataset']
        if (dataset == 'MNIST'):
            model_dict.update({'classes': 10})
        elif (dataset == 'GTSRB'):
            model_dict.update({'classes': 43})
        elif (dataset == 'HAR'):
            model_dict.update({'classes': 6})

    return model_dict
#------------------------------------------------------------------------------#


def get_svm_model_name(model_dict, rd=None, rev=None):
    """
    Helper function to get model name from <model_dict>, <rd> and <rev>
    """

    model_name = 'svm_{}_cls{}'.format(
        model_dict['svm_type'], model_dict['classes'])

    if model_dict['preprocess'] is not None:
        model_name += ('_' + model_dict['preprocess'])

    if rd is not None:
        model_name += '_{}_{}'.format(model_dict['dim_red'], rd)
        if rev is not None:
            model_name += '_rev'

    model_name += '_{}_C{:.0e}'.format(model_dict['penalty'],
                                       model_dict['penconst'])

    return model_name
#------------------------------------------------------------------------------#


def model_loader(model_dict, rd=None, rev=None):
    """
    Returns a classifier object if it already exists. Returns None, otherwise.
    """

    print('Loading model...')
    abs_path_m = resolve_path_m(model_dict)
    try:
        clf = joblib.load(abs_path_m + get_svm_model_name(model_dict, rd, rev) +
                          '.pkl')
    except BaseException:
        clf = None

    return clf
#------------------------------------------------------------------------------#


def model_trainer(model_dict, X_train, y_train, rd=None, rev=None):
    """Trains and returns SVM. Also save SVM to file."""

    print('Training model...')
    start_time = time.time()
    abs_path_m = resolve_path_m(model_dict)
    svm_model = model_dict['svm_type']
    C = model_dict['penconst']
    penalty = model_dict['penalty']

    # Create model based on parameters
    if svm_model == 'linear':
        dual = True
        if penalty == 'l1':
            dual = False
        clf = svm.LinearSVC(C=C, penalty=penalty, dual=dual)
    elif svm_model != 'linear':
        clf = svm.SVC(C=C, kernel=svm_model)

    # Train model
    clf.fit(X_train, y_train)
    print('Finish training in {:d}s'.format(int(time.time() - start_time)))

    # Save model
    joblib.dump(clf, abs_path_m +
                get_svm_model_name(model_dict, rd, rev) + '.pkl')
    return clf
#------------------------------------------------------------------------------#


def model_creator(model_dict, X_train, y_train, rd=None, rev=None):
    """Returns a SVM classifier"""

    # Load model based on model_dict
    clf = model_loader(model_dict, rd, rev)
    # If model does not exist, train a new SVM
    if clf is None:
        clf = model_trainer(model_dict, X_train, y_train, rd, rev)
    return clf
#------------------------------------------------------------------------------#


def model_transform(model_dict, clf, dr_alg=None, M=None):
    """
    Modify SVM's decision function to take into account transformation
    matrix to transform input data in original space

    Parameters
    ----------
    model_dict :
                 dictionary containing model param
    clf        :
                 sklearn SVM classifier object
    dr_alg     :
                 dim. red. object used on data
    M          :
                 transformation matrix used in preprocessing (optional)

    Returns
    -------
    clf        :
                 same SVM object with weights (coef_) modified
    """

    if dr_alg is not None:
        A = gradient_transform(model_dict, dr_alg)
        if M is not None:
            A = np.dot(A, M.T)
    elif M is not None:
        A = M.T
    else:
        return clf

    clf.coef_ = np.dot(clf.coef_, A)

    return clf
#------------------------------------------------------------------------------#


def model_tester(model_dict, clf, X_test, y_test, rd=None, rev=None):
    """
    Calculate model's accuracy and average normalized distance from correctly
    classified samples to separating hyperplane of corresponding class
    """

    predicted_labels = clf.predict(X_test)
    if model_dict['svm_type'] == 'linear':
        # Magnitude of weight vectors for each class
        norm = np.linalg.norm(clf.coef_, axis=1)
    else:
        # norm is arbritarily set to one for kernel SVM
        norm = np.ones(model_dict['classes'])

    test_len = len(X_test)
    sum_dist = 0
    n_correct = 0
    for i in range(test_len):
        if predicted_labels[i] == y_test[i]:
            n_correct += 1
            # Sum normalized distance to sept. hyperplane
            _, min_dist = min_dist_calc(X_test[i], clf)
            sum_dist += min_dist

    DR = model_dict['dim_red']
    # Resolve path to utility output file
    abs_path_o = resolve_path_o(model_dict)
    fname = 'utility_' + get_svm_model_name(model_dict)
    if rd != None:
        fname += '_' + DR
    if rev != None:
        fname += '_rev'
    ofile = open(abs_path_o + fname + '.txt', 'a')
    DR = model_dict['dim_red']
    if rd == None:
        ofile.write('No_' + DR + ' ')
    else:
        ofile.write(str(rd) + ' ')
    # Format: <dimensions> <accuracy> <avg. dist.>
    ofile.write('{:.2f} {:.3f} \n'.format(clf.score(X_test, y_test),
                                          sum_dist / n_correct))
    ofile.write('\n\n')
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
    acc_t = np.sum(atk_success) / float(len(X_adv))
    o_list.append(acc_t)
    # Accuracy vs. predicted labels
    atk_success = (y_adv != y_ini)
    acc_p = np.sum(atk_success) / float(len(X_adv))
    o_list.append(acc_p)
    # Accuracy for adv. examples generated from correctly classified examples
    i_c = np. where(y_ini == y_test)
    atk_success = (y_adv[i_c] != y_test[i_c])
    acc_c = np.sum(atk_success) / float(len(y_adv[i_c]))
    o_list.append(acc_c)

    return o_list
#------------------------------------------------------------------------------#


def file_create(model_dict, rd=None, strat=None, rev=None):
    """
    Creates and returns a file descriptor, named corresponding to model,
    attack type, strat, and rev
    """

    # Resolve absolute path to output directory
    abs_path_o = resolve_path_o(model_dict)
    fname = get_svm_model_name(model_dict)
    if strat is not None:
        fname += '_strat'
    if rd is not None:
        fname += '_' + model_dict['dim_red']
    if rev is not None:
        fname += '_rev'
    plotfile = open(abs_path_o + fname + '.txt', 'a')
    return plotfile, fname
#------------------------------------------------------------------------------#


def print_svm_output(model_dict, output_list, dev_list, rd=None, strat=None,
                     rev=None):
    """
    Creates an output file reporting accuracy and confidence of attack
    """

    plotfile, fname = file_create(model_dict, rd, strat, rev)
    plotfile.write('\\\\small{{{}}}\n'.format(rd))
    for i in range(len(dev_list)):
        plotfile.write('{0:.3f}'.format(dev_list[i]))
        for item in output_list[i]:
            plotfile.write(' {0:.3f}'.format(item))
        plotfile.write('\n')
    plotfile.write('\n\n')
    plotfile.close()
    return fname
#------------------------------------------------------------------------------#


def save_svm_images(model_dict, data_dict, X_test, adv_x, dev_mag, rd=None,
                    dr_alg=None, rev=None):
    """
    Save <no_of_img> adv. samples as image files in visual_data folder
    """

    no_of_img = 1    # Number of images to save
    indices = range(no_of_img)

    dataset = model_dict['dataset']
    DR = model_dict['dim_red']
    channels = model_dict['channels']
    abs_path_v = resolve_path_v(model_dict)

    no_of_features = data_dict['no_of_features']
    height = data_dict['height']
    width = data_dict['width']

    for i in indices:
        if channels == 1:
            adv = adv_x[i].reshape((height, width))
            orig = X_test[i].reshape((height, width))
            cmap = 'gray'
        else:
            adv = adv_x[i].reshape((channels, height, width))
            adv = adv.swapaxes(0, 2).swapaxes(0, 1)
            orig = X_test[i].reshape((channels, height, width))
            orig = orig.swapaxes(0, 2).swapaxes(0, 1)
            cmap = None

        fname = abs_path_v
        if (rd is not None) and (rev is None):
            fname += '{}_{}_{}'.format(i, DR, rd)
        elif rd is not None:
            fname += '{}_{}_rev_{}'.format(i, DR, rd)
        elif rd is None:
            fname += '{}'.format(i)

        img.imsave(fname + '_mag{}.png'.format(dev_mag), adv, vmin=0, vmax=1,
                   cmap=cmap)
        img.imsave(fname + '_orig.png', orig, vmin=0, vmax=1, cmap=cmap)
#------------------------------------------------------------------------------#


# def plotter(acc_def, acc, dev_list, rd_list, recons_flag=0, strat_flag=0):
#
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     from matplotlib.lines import Line2D
#     import glob as glob
#     import os
#     from matplotlib.pyplot import cm
#     from cycler import cycler
#
#     if strat_flag == 1: title = 'Strategic gradient '
#     elif strat_flag == 0: title = 'Gradient '
#     title += 'on DCA reduced dimensions for MNIST data with '
#     fname ='MNIST_svm_dca'
#     if recons_flag == 1:
#         title += 'recons defense'
#         fname += '_recon.png'
#     elif recons_flag == 0:
#         title += 'retrain defense'
#         fname += '_retrain'
#         if strat_flag == 1: fname += '_strat'
#         fname += '.png'
#
#     font = {'size': 17}
#     matplotlib.rc('font', **font)
#     cm = plt.get_cmap('gist_rainbow')
#     fig, ax = plt.subplots(1, 1, figsize=(12,9))
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
#     colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
#     markers = ('o', '^', 'x', 'D', 's', '|', 'v')
#     handle_list = []
#     count = 0
#     for item in rd_list:
#         count += 1
#         color = colors[count % len(colors)]
#         style = markers[count % len(markers)]
#         handle_list.append(plt.plot(dev_list, np.multiply(100, acc_def[count-1, :]),
#         linestyle='-', marker=style, color=color, markersize=10, label=item))
#     handle_list.append(plt.plot(dev_list, np.multiply(100, acc),
#     linestyle='-', marker='o', color='b', markersize=10, label='No defense'))
#
#     plt.xlabel(r'Adversarial perturbation')
#     plt.ylabel('Adversarial success')
#     plt.title(title)
#     plt.xticks()
#     plt.legend(loc=2, fontsize=14)
#     plt.ylim(0, 100)
#     plt.savefig(fname, bbox_inches='tight')
#     plt.show()
# #------------------------------------------------------------------------------#
