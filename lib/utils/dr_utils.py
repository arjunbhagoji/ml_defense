"""
Utility file containing helper functions that perform various dimensionality
reduction technique.
"""

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection as GRP

from lib.utils.data_utils import *
from lib.utils.DCA import DCA

#------------------------------------------------------------------------------#
def pca_dr(X_train, X_test, rd, X_val=None, rev=None):

    """
    Perform PCA on X_train then transform X_train, X_test (and X_val).
    Return transformed data in original space if rev is True; otherwise, return
    transformed data in PCA space.
    """

    # Fit PCA model on training data, random_state is specified to make sure
    # result is reproducible
    pca = PCA(n_components=rd, random_state=10)
    pca.fit(X_train)

    # Transforming training and test data
    X_train_dr = pca.fit_transform(X_train)
    X_test_dr = pca.transform(X_test)
    if X_val is not None: X_val_dr = pca.transform(X_val)

    if rev != None:
        X_train_rev = pca.inverse_transform(X_train_dr)
        X_test_rev = pca.inverse_transform(X_test_dr)
        if X_val is not None:
            X_val_rev = pca.inverse_transform(X_val_dr)
            return X_train_rev, X_test_rev, X_val_rev, pca
        else: return X_train_rev, X_test_rev, pca
    elif rev == None:
        if X_val is not None: return X_train_dr, X_test_dr, X_val_dr, pca
        else: return X_train_dr, X_test_dr, pca
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def random_proj_dr(X_train, X_test, rd, rev=None):

    """
    Perform Gaussian Random Projection on X_train then transform X_train, X_test
    (and X_val). Return transformed data in original space if rev is True;
    otherwise, return transformed data in PCA space.
    """

    # Generating random matrix for projection
    grp = GRP(n_components=rd, random_state=10)
    X_train_dr = grp.fit_transform(PCA_in_train)
    X_test_dr = grp.transform(PCA_in_test)

    X_train_dr = X_train_dr.reshape((train_len,1,rd))
    X_test_dr = X_test_dr.reshape((test_len,1,rd))

    return X_train_dr, X_test_dr, grp
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def dca_dr(X_train, X_test, rd, X_val=None, rev=None):

    """
    Perform DCA on X_train then transform X_train, X_test (and X_val).
    Return transformed data in original space if rev is True; otherwise, return
    transformed data in DCA space.
    """

    # Fit DCA model on training data, random_state is specified to make sure
    # result is reproducible
    dca = DCA(n_components=rd, rho=None, rho_p=None)
    dca.fit(X_train)

    # Transforming training and test data
    dca.fit(X_train, y_train)
    X_train_dr = dca.transform(X_train)
    X_test_dr = dca.transform(X_test)
    if X_val is not None: X_val_dr = dca.transform(X_val)

    if rev != None:
        X_train_rev = dca.inverse_transform(X_train_dr)
        X_test_rev = dca.inverse_transform(X_test_dr)
        if X_val is not None:
            X_val_rev = dca.inverse_transform(X_val_dr)
            return X_train_rev, X_test_rev, X_val_rev, pca
        else: return X_train_rev, X_test_rev, pca
    elif rev == None:
        if X_val is not None: return X_train_dr, X_test_dr, X_val_dr, pca
        else: return X_train_dr, X_test_dr, pca
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def invert_dr(X, dr_alg, DR):

    """
    Inverse transform data <X> in reduced dimension back to its full dimension
    """

    if (DR == 'pca') or (DR == 'dca'):
        X_rev = dr_alg.inverse_transform(X)

    return X_rev
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def dr_wrapper(X_train, X_test, DR, rd, X_val=None, rev=None):

    """
    A wrapper function for dimensionality reduction functions.
    """

    data_dict = get_data_shape(X_train, X_test, X_val)
    no_of_dim = data_dict['no_of_dim']

    train_len = data_dict['train_len']
    test_len = data_dict['test_len']
    no_of_features = data_dict['no_of_features']

    # Reshape for PCA function
    DR_in_train = X_train.reshape(train_len, no_of_features)
    DR_in_test = X_test.reshape(test_len, no_of_features)
    if X_val is not None:
        val_len = data_dict['val_len']
        DR_in_val = X_val.reshape(val_len, no_of_features)
    else:
        DR_in_val = None

    #----------------------------------- PCA ----------------------------------#
    if DR == 'pca':
        if X_val is not None:
            X_train, X_test, X_val, dr_alg = pca_dr(DR_in_train, DR_in_test, rd,
                                                    DR_in_val, rev)
        else:
            X_train, X_test, dr_alg = pca_dr(DR_in_train, DR_in_test, rd,
                                             DR_in_val, rev)

    #----------------------------- Random Projection --------------------------#
    elif DR == 'rp':
        if X_val is not None:
            X_train, X_test, X_val, dr_alg = pca_dr(DR_in_train, DR_in_test, rd,
                                                    DR_in_val, rev)
        else:
            X_train, X_test, dr_alg = pca_dr(DR_in_train, DR_in_test, rd,
                                             DR_in_val, rev)

    #----------------------------------- DCA ----------------------------------#
    if DR == 'dca':
        if X_val is not None:
            X_train, X_test, X_val, dr_alg = pca_dr(DR_in_train, DR_in_test, rd,
                                                    DR_in_val, rev)
        else:
            X_train, X_test, dr_alg = pca_dr(DR_in_train, DR_in_test, rd,
                                             DR_in_val, rev)

    if (no_of_dim == 3) or ((no_of_dim == 4) and (rev == None)):
        channels = data_dict['channels']
        X_train = X_train.reshape((train_len, channels, rd))
        X_test = X_test.reshape((test_len, channels, rd))
        if X_val is not None:
            X_val = X_val.reshape((val_len, channels, rd))
    elif (no_of_dim == 4) and (rev != None):
        channels = data_dict['channels']
        height = data_dict['height']
        width = data_dict['width']
        X_train = X_train.reshape((train_len, channels, height, width))
        X_test = X_test.reshape((test_len, channels, height, width))
        if X_val is not None:
            X_val = X_val.reshape((val_len, channels, height, width))

    if X_val is not None: return X_train, X_test, X_val, dr_alg
    else: return X_train, X_test, dr_alg
#------------------------------------------------------------------------------#
