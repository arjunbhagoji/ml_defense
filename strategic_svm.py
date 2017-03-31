import numpy as np
import argparse
from sklearn import svm
from sklearn.decomposition import PCA

from lib.utils.data_utils import *
from lib.utils.svm_utils import *
#from lib.utils.dr_utils import *

# TODO: integrate with dr_utils, dr_wrapper()
#------------------------------------------------------------------------------#
def pca_dr(X_train, X_test, rd, recons_flag=None):
    train_len = len(X_train)
    test_len = len(X_test)
    height = X_train.shape[2]
    width = X_train.shape[3]
    n_features = height*width
    # Reshaping for PCA function
    PCA_in_train = X_train.reshape(train_len, n_features)
    PCA_in_test = X_test.reshape(test_len, n_features)
    # Fitting the PCA model on training data
    pca = PCA(n_components=rd)
    pca_train = pca.fit(PCA_in_train)
    # Reconstructing training and test data
    X_train_dr = pca.transform(PCA_in_train)
    X_test_dr = pca.transform(PCA_in_test)

    if recons_flag != None:
        X_train_rev = pca.inverse_transform(X_train_dr)
        X_train_rev = X_train_rev.reshape((train_len, 1, height, width))
        X_test_rev = pca.inverse_transform(X_test_dr)
        X_test_rev = X_test_rev.reshape((test_len, 1, height, width))
        return X_test_rev, pca
    elif recons_flag == None:
        X_train_dr = X_train_dr.reshape((train_len, 1, rd))
        X_test_dr = X_test_dr.reshape((test_len, 1, rd))
        return X_train_dr, X_test_dr, pca
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def mult_cls_atk(clf, X_test, dev_mag):

    """
    Returns
    (1) Adversarial samples generated from <X_test> for linear SVM <clf>
        with perturbation magnitude <dev_mag>
    (2) Predicted labels of <X_test> by <clf>
    """

    test_len = len(X_test)
    X_adv = np.zeros((test_len, X_test.shape[1]))
    y_ini = np.zeros(test_len)
    classes = clf.intercept_.shape[0]

    for i in range(test_len):
        x_ini = X_test[i,:].reshape(1, -1)
        ini_class = clf.predict(x_ini)
        w = clf.coef_[ini_class[0],:]
        d_list = []
        i_list = []
        distances = clf.decision_function(x_ini)
        for j in range(classes):
            if j == ini_class[0]: continue
            w_curr = clf.coef_[j,:]
            d_list.append(abs(distances[0,j] - distances[0,ini_class[0]])
                          /np.linalg.norm(w_curr - w))
            i_list.append(j)
            i_d_list = zip(i_list, d_list)
        i_d_list = sorted(i_d_list, key = lambda x:x[1])
        min_index = i_d_list[0][0]
        min_dist = i_d_list[0][1]
        w_min = clf.coef_[min_index,:]
        x_adv = (x_ini - dev_mag*((w - w_min)/(np.linalg.norm(w - w_min))))
        X_adv[i,:] = x_adv
        y_ini[i] = ini_class

    return X_adv, y_ini
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def plotter(acc_def, acc, dev_list, rd_list, recons_flag=0, strat_flag=0):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import glob as glob
    import os
    from matplotlib.pyplot import cm
    from cycler import cycler

    if strat_flag == 1: title = 'Strategic gradient '
    elif strat_flag == 0: title = 'Gradient '
    title += 'on DCA reduced dimensions for MNIST data with '
    fname ='MNIST_svm_dca'
    if recons_flag == 1:
        title += 'recons defense'
        fname += '_recon.png'
    elif recons_flag == 0:
        title += 'retrain defense'
        fname += '_retrain'
        if strat_flag == 1: fname += '_strat'
        fname += '.png'

    font = {'size': 17}
    matplotlib.rc('font', **font)
    cm = plt.get_cmap('gist_rainbow')
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    markers = ('o', '^', 'x', 'D', 's', '|', 'v')
    handle_list = []
    count = 0
    for item in rd_list:
        count += 1
        color = colors[count % len(colors)]
        style = markers[count % len(markers)]
        handle_list.append(plt.plot(dev_list, np.multiply(100, acc_def[count-1, :]),
        linestyle='-', marker=style, color=color, markersize=10, label=item))
    handle_list.append(plt.plot(dev_list, np.multiply(100, acc),
    linestyle='-', marker='o', color='b', markersize=10, label='No defense'))

    plt.xlabel(r'Adversarial perturbation')
    plt.ylabel('Adversarial success')
    plt.title(title)
    plt.xticks()
    plt.legend(loc=2, fontsize=14)
    plt.ylim(0, 100)
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def main(argv):

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST', type=str,
           help='Specify dataset (default: MNIST)')
    parser.add_argument('-c', '--channels', default=1, type=int,
           help='Specify number of input channels (1 (default) or 3)')
    parser.add_argument('--two_classes',
           help='Train SVM on two classes instead of all available classes')
    parser.add_argument('--dr', default='pca', type=str,
           help='Specify dimension reduction (default: pca)')
    parser.add_argument('-C', '--penconst', default=1.0, type=float,
           help='Specify penalty parameter C (default: 1.0)')
    parser.add_argument('-p', '--penalty', default='l2', type=str,
           help='Specify norm to use in penalization (l1 or l2 (default))')
    args = parser.parse_args()

    # TODO: error when use l1 norm
    # ValueError: Unsupported set of arguments: The combination of penalty='l1'
    # and loss='squared_hinge' are not supported when dual=True,
    # Parameters: penalty='l1', loss='squared_hinge', dual=True

    # Create and update model_dict
    model_dict = {}
    model_dict.update({'dataset':args.dataset})
    model_dict.update({'channels':args.channels})
    model_dict.update({'dr':args.dr})
    model_dict.update({'penconst':args.penconst})
    model_dict.update({'penalty':args.penalty})
    if args.two_classes:
        model_dict.update({'classes':2})
    else:
        # TODO: preferrably put this somehere else
        if (model_dict['dataset'] == 'MNIST'):
            model_dict.update({'classes':10})
        elif (model_dict['dataset'] == 'GTSRB'):
            model_dict.update({'classes':43})

    print('Loading data...')
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(model_dict)
    # TODO: 2 classes case
    # if model_dict['classes'] == 2:
    #     X_train = X_train
    train_len = len(X_train)
    test_len = len(X_test)
    channels = X_test.shape[1]
    height = X_test.shape[2]
    width = X_test.shape[3]
    n_features = channels*height*width

    # Reshape dataset to have dimensions suitable for SVM
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    X_val_flat= X_val.reshape(-1, n_features)

    # Create a new model or load an existing one
    clf = model_creator(model_dict, X_train_flat, y_train)
    model_tester(model_dict, clf, X_test_flat, y_test)

    # Assign parameters
    n_mag = 10                                  # No. of deviations to consider
    dev_list = np.linspace(0.01, 0.1, n_mag)    # A list of deviations mag.
    rd_list = [331, 100, 50, 40, 30, 20, 10]    # Reduced dimensions to use
    n_rd = len(rd_list)
    output_list = []

    # Test clf against adv. samples
    print('Performing attack...')
    if model_dict['classes'] != 2:
        for i in range(n_mag):
            X_adv, y_ini = mult_cls_atk(clf, X_test_flat, dev_list[i])
            output_list.append(acc_calc_all(clf, X_adv, y_test, y_ini))
        print_output(model_dict, output_list, dev_list, n_features)
    else:
        # TODO: 2 classes
        print('TODO')

    # Retrain defense and strategic attack
    print('--------------Retrain Defense & Strategic Attack--------------')
    for j in range(n_rd):
        rd = rd_list[j]
        output_list = []
        print('Reduced dimensions: {}'.format(rd))

        # Dimension reduce dataset and reshape
        X_train_dr, X_test_dr, pca = pca_dr(X_train, X_test, rd)
        X_train_dr = X_train_dr.reshape(-1, rd)
        X_test_dr = X_test_dr.reshape(-1, rd)

        # With dimension reduced dataset, create new model or load existing one
        clf = model_creator(model_dict, X_train_dr, y_train, rd)
        model_tester(model_dict, clf, X_test_dr, y_test)

        # Strategic attack: create new adv samples based on retrained clf
        print('Performing strategic attack...')
        for i in range(n_mag):
            X_adv, y_ini = mult_cls_atk(clf, X_test_dr, dev_list[i])
            output_list.append(acc_calc_all(clf, X_adv, y_test, y_ini))
        print_output(model_dict, output_list, dev_list, rd, strat_flag=1)

    # TODO: save image
    # TODO: Plot
    # plotter(acc_recons, acc_no_def, dev_list, rd_list, recons_flag=1, strat_flag=0)
    # plotter(acc_retrain, acc_no_def, dev_list, rd_list, recons_flag=0, strat_flag=0)
    # plotter(acc_strat, acc_no_def, dev_list, rd_list, recons_flag=0, strat_flag=1)

if __name__ == "__main__":
   main(sys.argv[1:])
