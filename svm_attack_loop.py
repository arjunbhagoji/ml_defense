import subprocess
import numpy as py
import multiprocessing

def reg_param(C):
    for whiten in [1,2,3]:
        print "Running attack for C={}".format(C)
        subprocess.call(["python strategic_svm.py -dr antiwhiten"+str(whiten)+" -C {}".format(C)], shell = True)
    # subprocess.call(["gnuplot -e \"mname='svm_linear_cls6_l2_C{}_strat_pca'\" gnu_in_loop.plg".format(C)], shell=True)

C_list=['1e-05', '1e-04', '1e-03', '1e-02','1e-01', '1e+00', '1e+01']
pool=multiprocessing.Pool(processes=5)
pool.map(reg_param, C_list)
pool.close()
pool.join()
