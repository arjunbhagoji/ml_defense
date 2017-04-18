import os
import numpy as np
import theano
import theano.tensor as T

from lib.utils.theano_utils import *
from lib.utils.lasagne_utils import *
from lib.utils.data_utils import *
from lib.utils.dr_utils import *
from lib.utils.attack_utils import *
from lib.utils.plot_utils import *
from lib.utils.model_utils import *

def gradient_calc(rd, model_dict, X_train, y_train, X_test, y_test, X_val, y_val):

    # Parameters
    rev_flag = None
    dim_red = model_dict['dim_red']

    data_dict, test_prediction, dr_alg, X_test, input_var, target_var = \
        model_setup(model_dict, X_train, y_train, X_test, y_test, X_val, y_val,
                    rd, rev=rev_flag)

    test_len = data_dict['test_len']
    no_of_features = data_dict['no_of_features']
    X_test_dr = X_test.reshape((test_len, no_of_features))

    var_array = np.sqrt(np.var(X_test, axis=0))
    var_list = list(var_array)
    gradient_comp = avg_grad_calc(input_var, target_var, test_prediction,
                                  X_test, y_test)
    gradient_list = list(gradient_comp)

    return zip(var_list, gradient_list)

def main():

    # Create model_dict from arguments
    model_dict = model_dict_create()

    # Reduced dimensions used
    # rd_list = [784, 331, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    rd_list = [784, 331]

    # Load dataset specified in model_dict
    print('Loading data...')
    dataset = model_dict['dataset']
    if (dataset == 'MNIST') or (dataset == 'GTSRB'):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(model_dict)
    elif dataset == 'HAR':
        X_train, y_train, X_test, y_test = load_dataset(model_dict)

    no_of_dims=len(rd_list)

    gradient_var_list=[]

    for rd in rd_list:
        gradient_var_list.append(gradient_calc(rd, model_dict, X_train, y_train,
                                 X_test, y_test, X_val, y_val))
    mag_var_scatter(gradient_var_list, no_of_dims)

if __name__ == "__main__":
    main()
