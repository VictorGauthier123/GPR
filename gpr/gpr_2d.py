"""
gpr_2d.py

Gaussian Process Regression in 2D (latitude, longitude).
"""

import numpy as np
from gpr.kernels import rbf_kernel, covariance_matrix


def gpr_predict(X_train, y_train, X_test, length_scale=1.0):
    """
    Perform Gaussian Process Regression prediction in 2D.

    Parameters
    ----------
    X_train : ndarray, shape (n_samples, 2)
        Training input points (lat, lon).
    y_train : ndarray, shape (n_samples,)
        Training temperature values.
    X_test : ndarray, shape (n_test, 2)
        Test input points (lat, lon).
    length_scale : float
        Kernel length scale.

    Returns
    -------
    mu : ndarray, shape (n_test,)
        Predicted mean temperatures.
    """
    K = covariance_matrix(X_train, X_train, rbf_kernel, length_scale=length_scale) + 1e-5 * np.eye(len(X_train))
    K_inv = np.linalg.inv(K)
    K_test = covariance_matrix(X_test, X_train, rbf_kernel, length_scale=length_scale)

    mu = np.dot(K_test, np.dot(K_inv, y_train))
    return mu
