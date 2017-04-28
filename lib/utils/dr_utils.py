"""
Utility file containing helper functions that perform various dimensionality
reduction technique.
"""

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection as GRP

from lib.utils.data_utils import get_data_shape
from lib.utils.DCA import DCA
from lib.utils.AntiWhiten import AntiWhiten

#------------------------------------------------------------------------------#


def pca_dr(X_train, X_test, rd, X_val=None, rev=None, **kwargs):
    """
    Perform PCA on X_train then transform X_train, X_test (and X_val).
    Return transformed data in original space if rev is True; otherwise, return
    transformed data in PCA space.
    """

    whiten = kwargs['whiten']
    # Fit PCA model on training data, random_state is specified to make sure
    # result is reproducible
    pca = PCA(n_components=rd, whiten=whiten, random_state=10)
    pca.fit(X_train)

    # Transforming training and test data
    X_train_dr = pca.transform(X_train)
    X_test_dr = pca.transform(X_test)
    if X_val is not None:
        X_val_dr = pca.transform(X_val)

    if rev != None:
        X_train_rev = pca.inverse_transform(X_train_dr)
        X_test_rev = pca.inverse_transform(X_test_dr)
        if X_val is not None:
            X_val_rev = pca.inverse_transform(X_val_dr)
            return X_train_rev, X_test_rev, X_val_rev, pca
        else:
            return X_train_rev, X_test_rev, pca
    elif rev == None:
        if X_val is not None:
            return X_train_dr, X_test_dr, X_val_dr, pca
        else:
            return X_train_dr, X_test_dr, pca
#------------------------------------------------------------------------------#


def random_proj_dr(X_train, X_test, rd, X_val=None, rev=None, **kwargs):
    """
    Perform Gaussian Random Projection on X_train then transform X_train, X_test
    (and X_val). Return transformed data in original space if rev is True;
    otherwise, return transformed data in PCA space.
    """

    # Generating random matrix for projection
    grp = GRP(n_components=rd, random_state=10)
    X_train_dr = grp.fit_transform(PCA_in_train)
    X_test_dr = grp.transform(PCA_in_test)

    X_train_dr = X_train_dr.reshape((train_len, 1, rd))
    X_test_dr = X_test_dr.reshape((test_len, 1, rd))

    return X_train_dr, X_test_dr, grp
#------------------------------------------------------------------------------#


def dca_dr(X_train, X_test, rd, X_val=None, rev=None, **kwargs):
    """
    Perform DCA on X_train then transform X_train, X_test (and X_val).
    Return transformed data in original space if rev is True; otherwise, return
    transformed data in DCA space.
    """

    y_train = kwargs['y_train']
    if y_train is None:
        raise ValueError('y_train is required for DCA')

    # Fit DCA model on training data, random_state is specified to make sure
    # result is reproducible
    dca = DCA(n_components=rd, rho=None, rho_p=None)
    dca.fit(X_train, y_train)

    # Transforming training and test data
    X_train_dr = dca.transform(X_train)
    X_test_dr = dca.transform(X_test)
    if X_val is not None:
        X_val_dr = dca.transform(X_val)

    if rev != None:
        X_train_rev = dca.inverse_transform(X_train_dr)
        X_test_rev = dca.inverse_transform(X_test_dr)
        if X_val is not None:
            X_val_rev = dca.inverse_transform(X_val_dr)
            return X_train_rev, X_test_rev, X_val_rev, dca
        else:
            return X_train_rev, X_test_rev, dca
    elif rev == None:
        if X_val is not None:
            return X_train_dr, X_test_dr, X_val_dr, dca
        else:
            return X_train_dr, X_test_dr, dca
#------------------------------------------------------------------------------#


def anti_whiten_dr(X_train, X_test, rd, X_val=None, rev=None, **kwargs):
    """
    Perform dimensionality reduction with eigen-based whitening preprocessing
    """

    deg = kwargs['deg']
    # Fit X_train
    anti_whiten = AntiWhiten(n_components=rd, deg=deg)
    anti_whiten.fit(X_train)

    # Transform X_train and X_test
    X_train_dr = anti_whiten.transform(X_train)
    X_test_dr = anti_whiten.transform(X_test)
    if X_val is not None:
        # Transform X_Val
        X_val_dr = anti_whiten.transform(X_val)

    if rev != None:
        X_train_rev = anti_whiten.inverse_transform(X_train_dr, inv_option=1)
        X_test_rev = anti_whiten.inverse_transform(X_test_dr, inv_option=1)
        if X_val is not None:
            X_val_rev = anti_whiten.inverse_transform(X_val_dr, inv_option=1)
            return X_train_rev, X_test_rev, X_val_rev, anti_whiten
        else:
            return X_train_rev, X_test_rev, anti_whiten
    elif rev == None:
        if X_val is not None:
            return X_train_dr, X_test_dr, X_val_dr, anti_whiten
        else:
            return X_train_dr, X_test_dr, anti_whiten
#------------------------------------------------------------------------------#


def invert_dr(X, dr_alg, DR):
    """
    Inverse transform data <X> in reduced dimension back to its full dimension
    """

    inv_list = ['pca', 'pca-whiten', 'dca']

    if (DR in inv_list) or ('antiwhiten' in DR):
        X_rev = dr_alg.inverse_transform(X)
    else:
        raise ValueError('Cannot invert specified DR')

    return X_rev
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def dr_wrapper(X_train, X_test, DR, rd, y_train, X_val=None, rev=None):
    """
    A wrapper function for dimensionality reduction functions.
    """

    data_dict = get_data_shape(X_train, X_test, X_val)
    no_of_dim = data_dict['no_of_dim']

    train_len = data_dict['train_len']
    test_len = data_dict['test_len']
    no_of_features = data_dict['no_of_features']

    # Reshape for dimension reduction function
    DR_in_train = X_train.reshape(train_len, no_of_features)
    DR_in_test = X_test.reshape(test_len, no_of_features)
    if X_val is not None:
        val_len = data_dict['val_len']
        DR_in_val = X_val.reshape(val_len, no_of_features)
    else:
        DR_in_val = None

    whiten = None
    deg = None

    # Assign corresponding DR function
    if 'pca' in DR:
        dr_func = pca_dr
        if DR == 'pca-whiten':
            whiten = True
        else:
            whiten = False
    elif DR == 'rp':
        dr_func = random_proj_dr
    elif DR == 'dca':
        dr_func = dca_dr
    elif 'antiwhiten' in DR:
        dr_func = anti_whiten_dr
        deg = int(DR.split('antiwhiten', 1)[1])

    # Perform DR
    if X_val is not None:
        X_train, X_test, X_val, dr_alg = dr_func(DR_in_train, DR_in_test, rd,
                                                 X_val=DR_in_val, rev=rev,
                                                 y_train=y_train, whiten=whiten,
                                                 deg=deg)
    else:
        X_train, X_test, dr_alg = dr_func(DR_in_train, DR_in_test, rd,
                                          X_val=DR_in_val, rev=rev,
                                          y_train=y_train, whiten=whiten,
                                          deg=deg)

    # Reshape DR data to appropriate shape (original shape if rev)
    if (no_of_dim == 3) or ((no_of_dim == 4) and (rev is None)):
        channels = data_dict['channels']
        X_train = X_train.reshape((train_len, channels, rd))
        X_test = X_test.reshape((test_len, channels, rd))
        if X_val is not None:
            X_val = X_val.reshape((val_len, channels, rd))
    elif (no_of_dim == 4) and (rev is not None):
        channels = data_dict['channels']
        height = data_dict['height']
        width = data_dict['width']
        X_train = X_train.reshape((train_len, channels, height, width))
        X_test = X_test.reshape((test_len, channels, height, width))
        if X_val is not None:
            X_val = X_val.reshape((val_len, channels, height, width))

    # X_train = reshape_data(X_train, data_dict, rd=rd, rev=rev)
    # X_test = reshape_data(X_test, data_dict, rd=rd, rev=rev)
    # X_val = reshape_data(X_val, data_dict, rd=rd, rev=rev)

    if X_val is not None:
        return X_train, X_test, X_val, dr_alg
    else:
        return X_train, X_test, dr_alg
#------------------------------------------------------------------------------#
