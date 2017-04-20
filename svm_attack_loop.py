import subprocess
import numpy as py
import multiprocessing

def reg_param(C):
    print "Running attack for C={}".format(C)
    subprocess.call(["python strategic_svm.py -C {} -p l1".format(C)], shell = True)
    # subprocess.call(["gnuplot -e \"mname='svm_linear_cls10_l2_C{}_strat_pca_rev'\" gnu_in_loop.plg".format(C)], shell=True)

C_list=['1e-04', '1e-03', '1e-02','1e-01', '1e+00']
pool=multiprocessing.Pool(processes=5)
pool.map(reg_param, C_list)
pool.close()
pool.join()
