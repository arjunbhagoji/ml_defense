from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

from lib.utils.svm_utils import *

#------------------------------------------------------------------------------#
def mag_var_scatter(model_dict, gradient_var_list, no_of_dims, rd = None, rev = None):

    """
    Create a scatter plot of gradient of model vs. variance for each dimension
    of the data
    """

    f, axarr = plt.subplots(no_of_dims, 1, sharex=True, figsize=(12,4))
    grad_v_curr = gradient_var_list[0]
    axarr.plot(zip(*grad_v_curr)[0], zip(*grad_v_curr)[1], 'bo')
    # for i in range(no_of_dims):
    #     grad_v_curr = gradient_var_list[i]
    #     axarr[i].plot(zip(*grad_v_curr)[0], zip(*grad_v_curr)[1], 'bo')
    ax = f.add_subplot(111, frameon=False)
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='None', top='off', bottom='off', left='off',
                   right='off')
    plt.xlabel('Standard deviation of component')
    plt.ylabel(r'Magnitude of $i^{th}$' + '\n' + r'component of $\mathbf{w}$', labelpad = 20)
    axarr.set_ylim([0,3.0])
    # plt.title('Magnitude-variance scatter plot for MNIST data')
    abs_path = resolve_path()
    rel_path_p = 'plots/'
    abs_path_p = os.path.join(abs_path, rel_path_p)
    fname = get_svm_model_name(model_dict, None, None, rd, rev) + '_scatter'
    plt.savefig(abs_path_p + fname, bbox_inches = 'tight')

#------------------------------------------------------------------------------#
