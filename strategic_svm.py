import numpy as np
from sklearn import svm

from lib.utils.data_utils import *
from lib.utils.DCA import *

def mult_cls_atk(clf, X_test, dev_mag, rd=None):
    """
    Returns
    (1) Adversarial samples generated from <X_test> for linear SVM <clf>
        with perturbation magnitude <dev_mag>
    (2) Predicted labels of <X_test> by <clf>
    """

    test_len = len(X_test)
    if rd == None: X_adv = np.zeros((test_len, 784))
    else: X_adv = np.zeros((test_len, rd))
    y_ini = np.zeros(test_len)

    for i in range(test_len):
        x_ini = X_test[i, :].reshape(1, -1)
        ini_class = clf.predict(x_ini)
        w = clf.coef_[ini_class[0], :]
        d_list = []
        i_list = []
        distances = clf.decision_function(x_ini)
        for j in range(10):
            if j == ini_class[0]: continue
            w_curr = clf.coef_[j, :]
            d_list.append(abs(distances[0, j] - distances[0, ini_class[0]])
                          /np.linalg.norm(w_curr - w))
            i_list.append(j)
            i_d_list = zip(i_list, d_list)
        i_d_list = sorted(i_d_list, key = lambda x:x[1])
        min_index = i_d_list[0][0]
        min_dist = i_d_list[0][1]
        w_min = clf.coef_[min_index,:]
        x_adv = (x_ini - dev_mag*((w - w_min)/(np.linalg.norm(w - w_min))))
        X_adv[i, :] = x_adv
        y_ini[i] = ini_class

    return X_adv, y_ini

def acc_adv(clf, X_adv, y_ini, dca=None, rd=None, recons_flag=0):
    """
    Return attack success rate on <clf>
    """

    # Evaluate adv samples directly without DCA
    if dca == None or rd == None:
        final_class = clf.predict(X_adv)
    # Evaluate adv samples after transformed to DCA dimensions
    else:
        X_adv_dca = dca.transform(X_adv, dim=rd)
        if recons_flag == 1:
            X_adv_dca_rev = dca.inverse_transform(X_adv_dca, dim=rd)
            final_class = clf.predict(X_adv_dca_rev)
        elif recons_flag == 0:
            final_class = clf.predict(X_adv_dca)

    return np.sum(final_class != y_ini)/float(len(X_adv))

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

def main(argv):

    print('Loading data...')
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    train_len = len(X_train)
    test_len = len(X_test)
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    # Assign parameters
    n_mag = 5
    dev_list = np.linspace(0.1, 5, n_mag)
    rd_list = [1, 3, 5, 7, 9]
    n_rd = len(rd_list)
    X_adv_mag = np.zeros((n_mag, test_len, 784))
    y_ini = np.zeros(test_len)
    print('Fitting DCA...')
    dca = DCA(rho_p=-0.0001)
    dca.fit(X_train, y_train)

    # For plotting
    acc_no_def = np.zeros(n_mag)
    acc_recons = np.zeros((n_rd, n_mag))
    acc_retrain = np.zeros((n_rd, n_mag))
    acc_strat = np.zeros((n_rd, n_mag))

    # Train and test liner SVM
    print('Training linear SVM...')
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    print('Classifier Score: {:.3f}'.format(clf.score(X_test, y_test)))

    # Test linear SVM against adv samples
    for i in range(n_mag):
        X_adv_mag[i, :, :], y_ini = mult_cls_atk(clf, X_test, dev_list[i])
        acc_no_def[i] = acc_adv(clf, X_adv_mag[i, :, :], y_ini)
        print('Attack Success Rate (mag={:.3f}): {:.3f}'.format(dev_list[i],
              acc_no_def[i]))

    # Recons defense
    print('------------------------Recons Defense------------------------')
    for j in range(n_rd):
        rd = rd_list[j]
        print('Reduced dimensions: {}'.format(rd))

        # Evaluate accuracy
        X_test_dca = dca.transform(X_test, dim=rd)
        X_test_dca_rev = dca.inverse_transform(X_test_dca, dim=rd)
        print('Recons Classifier Score: {:.3f}'.format(clf.score(X_test_dca_rev, y_test)))

        for i in range(n_mag):
            # Evaluate recons defense on original adv samples
            acc_recons[j, i] = acc_adv(clf, X_adv_mag[i, :, :], y_ini, dca, rd,
                                       recons_flag=1)
            print('Attack Success Rate (mag={:.3f}): {:.3f}'.format(dev_list[i],
                  acc_recons[j, i]))

    # Retrain defense and strategic attack
    print('--------------Retrain Defense & Strategic Attack--------------')
    for j in range(n_rd):
        rd = rd_list[j]
        print('Reduced dimensions: {}'.format(rd))

        # Train and test new SVM in DCA dimensions
        # X_train_dca = dca.transform(X_train, dim=rd)
        # X_test_dca = dca.transform(X_test, dim=rd)
        # clf_def = svm.LinearSVC()
        # clf_def.fit(X_train_dca, y_train)
        # print('Retrained Classifier Score: {:.3f}'.format(clf_def.score(X_test_dca, y_test)))
        X_train_dca = dca.transform(X_train, dim=rd)
        X_test_dca = dca.transform(X_test, dim=rd)
        X_train_dca_rev = dca.inverse_transform(X_train_dca, dim=rd)
        X_test_dca_rev = dca.inverse_transform(X_test_dca, dim=rd)
        clf_def = svm.LinearSVC()
        clf_def.fit(X_train_dca_rev, y_train)
        print('Retrained Classifier Score: {:.3f}'.format(clf_def.score(X_test_dca_rev, y_test)))

        for i in range(n_mag):
            # Evaluate retrained clf on original adv samples
            # acc_retrain[j, i] = acc_adv(clf_def, X_adv_mag[i, :, :], y_ini, dca, rd)
            # print('Attack Success Rate (mag={:.3f}): {:.3f}'.format(dev_list[i],
            #       acc_retrain[j, i]))
            X_adv_dca = dca.transform(X_adv_mag[i, :, :], dim=rd)
            X_adv_dca_rev = dca.inverse_transform(X_adv_dca, dim=rd)
            acc_retrain[j, i] = acc_adv(clf_def, X_adv_dca_rev, y_ini)
            print('Attack Success Rate (mag={:.3f}): {:.3f}'.format(dev_list[i],
                  acc_retrain[j, i]))

            # Strategic attack: create new adv samples based on retrained clf
            # X_adv_strat, y_ini_strat = mult_cls_atk(clf_def, X_test_dca, dev_list[i], rd)
            # acc_strat[j, i] = acc_adv(clf_def, X_adv_strat, y_ini_strat)
            # print('Strategic Attack Success Rate (mag={:.3f}): {:.3f}'.format(dev_list[i],
            #       acc_strat[j, i]))
            X_adv_strat, y_ini_strat = mult_cls_atk(clf_def, X_test_dca_rev, dev_list[i])
            acc_strat[j, i] = acc_adv(clf_def, X_adv_strat, y_ini_strat)
            print('Strategic Attack Success Rate (mag={:.3f}): {:.3f}'.format(dev_list[i],
                  acc_strat[j, i]))

    # Plot
    # plotter(acc_recons, acc_no_def, dev_list, rd_list, recons_flag=1, strat_flag=0)
    # plotter(acc_retrain, acc_no_def, dev_list, rd_list, recons_flag=0, strat_flag=0)
    # plotter(acc_strat, acc_no_def, dev_list, rd_list, recons_flag=0, strat_flag=1)

if __name__ == "__main__":
   main(sys.argv[1:])
