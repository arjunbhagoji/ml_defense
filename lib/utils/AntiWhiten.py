"""
An implementation of data whitening based on eigenvalues of the covariance
matrix. It can be used as dimensionality reduction or as data preprocessing.
Usage is similar to that of sklearn's decomposition module (i.e. PCA).
Please note that this implementation is not at all optimized for speed or memory
usage.
"""

import numpy as np

#------------------------------------------------------------------------------#


class AntiWhiten:

    """
    Perform linear projection on data with options to manipulate covariance
    of projected data. Can also be used as dimension reduction.

    whiten
     -1        whiten (covariance matrix is identity matrix)
      0		   normal linear PCA
     >= 1 	   "anti-whiten" (covariance is mutiplied)
    """

    def __init__(self, n_components=None, whiten=None):
        """
        Init function, specify n_components here if the object will be used
        only with one number of dimensions
        """

        self.n_components = n_components
        self.whiten = whiten

    def _check_dim(self, dim, data_len):
        """Return valid dimensions"""

        if dim is None:
            dim = self.n_components
            if dim is None:
                dim = data_len
        if isinstance(dim, (int, long)) and (dim > 0) and (dim <= len(self._V)):
            return dim
        else:
            raise ValueError('Invalid number of dimensions.')

    def _check_whiten(self, whiten):
        """Return valid whiten degree"""

        if whiten is None:
            whiten = self.whiten
            if whiten is None:
                whiten = 0
        if isinstance(whiten, (int, long)) and (whiten >= -1):
            return whiten
        else:
            raise ValueError(
                'Invalid number of whitening degree (must >= -1).')

    def fit(self, X):
        """Fit X. Assume that X has shape (n_samples, n_features)"""

        self.n_samples = X.shape[0]
        # Center data
        self._mean = np.mean(X, axis=0)
        X = np.copy(X) - self._mean

        _, self._S, self._V = np.linalg.svd(X, full_matrices=False)

    def transform(self, X, whiten=None, dim=None):
        """Transform X"""

        # TODO: check if fit has been called

        n_samples = self.n_samples
        # Center data
        X = np.copy(X) - self._mean

        # Reduce dimensions if specified
        dim = self._check_dim(dim, X.shape[1])
        S = np.diag(self._S[:dim])
        V = self._V.T[:, :dim]

        X_pca = np.dot(X, V)
        whiten = self._check_whiten(whiten)
        if whiten == -1:
            # X_whiten = X * V / S * sqrt(n_samples) = U * sqrt(n_samples)
            X_proj = np.dot(X_pca, np.linalg.inv(S)) * np.sqrt(n_samples)
        elif whiten == 0:
            # X_pca = X * V = U * S * V^T * V = U * S
            X_proj = X_pca
        elif whiten >= 1:
            # X_antiwhite = X * V * (S / sqrt(n_samples))^d
            X_proj = X_pca
            for i in range(whiten):
                X_proj = np.dot(X_proj, S) / np.sqrt(n_samples)

        return X_proj

    def inverse_transform(self, X, whiten=None, inv_option=1):
        """
        Inverse transform projects X back to its original space based on
        inv_option:
        1 : multiplying with both S and components matrix corresponding to
                whitening
        2 : only multiplying with transpose of components matrix regardless of
                whitening
        """

        n_samples = self.n_samples
        # Reduce dimensions if specified
        dim = self._check_dim(X.shape[1], None)
        S = np.diag(self._S[:dim])
        V = self._V.T[:, :dim]

        if inv_option == 1:
            whiten = self._check_whiten(whiten)
            if whiten == -1:
                # X_whiten = X * V / S * sqrt(n_samples)
                temp = np.dot(X / np.sqrt(n_samples), S)
                X_inv = np.dot(temp, V.T) + self._mean
            elif whiten == 0:
                # X_pca = X * V
                X_inv = np.dot(X, V.T) + self._mean
            elif whiten >= 1:
                # X_antiwhite = X * V * (S / sqrt(n_samples))^d
                temp = X
                for i in range(whiten):
                    temp = np.dot(temp * np.sqrt(n_samples), np.linalg.inv(S))
                X_inv = np.dot(temp, V.T) + self._mean

        elif inv_option == 2:
            X_inv = np.dot(X, V.T) + self._mean

        return X_inv
