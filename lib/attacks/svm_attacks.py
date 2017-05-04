"""Utility file contains attack algorithm for SVM"""

import numpy as np

#------------------------------------------------------------------------------#


def min_dist_calc(x, clf):
    """
    Find the class that <x> is closest to its hyperplane and calculate that
    distance
    """

    x_ini = x.reshape(1, -1)
    ini_class = clf.predict(x_ini)
    w = clf.coef_[int(ini_class[0]), :]
    d_list = []
    i_list = []
    distances = clf.decision_function(x_ini)
    classes = clf.intercept_.shape[0]
    for j in range(classes):
        if j == ini_class[0]:
            continue
        w_curr = clf.coef_[j, :]
        d_list.append(abs(distances[0, j] - distances[0, int(ini_class[0])])
                      / np.linalg.norm(w_curr - w))
        i_list.append(j)
        i_d_list = zip(i_list, d_list)
    i_d_list = sorted(i_d_list, key=lambda x: x[1])
    min_index = i_d_list[0][0]
    min_dist = i_d_list[0][1]
    return min_index, min_dist
#------------------------------------------------------------------------------#


def mult_cls_atk(clf, X_test, mean, dev_mag):
    """
    Returns
    (1) Adversarial samples generated from <X_test> for linear SVM <clf>
        with perturbation magnitude <dev_mag>
    (2) Predicted labels of <X_test> by <clf>
    """

    test_len = len(X_test)
    X_adv = np.zeros((test_len, X_test.shape[1]))
    y_ini = np.zeros(test_len)

    for i in range(test_len):
        x_ini = X_test[i, :].reshape(1, -1)
        ini_class = clf.predict(x_ini)
        min_index, min_dist = min_dist_calc(x_ini, clf)
        w = clf.coef_[int(ini_class[0]), :]
        w_min = clf.coef_[min_index, :]
        x_adv = (x_ini - dev_mag * ((w - w_min) / (np.linalg.norm(w - w_min))))
        X_adv[i, :] = x_adv
        y_ini[i] = ini_class

    # Clip adv. examples if its values exceed original input range
    X_adv += mean
    np.clip(X_adv, 0, 1)
    X_adv -= mean

    return X_adv, y_ini
#------------------------------------------------------------------------------#
