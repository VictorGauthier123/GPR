"""
gpr_time.py

Gaussian Process Regression for time series (1D: day).
"""

import numpy as np
from gpr.kernels import matern_kernel, covariance_matrix


def gpr_predict(X_train, y_train, X_test, l, nu=1.5, sigma=1.0):
    """
    Gaussian Process Regression prediction with a Matern kernel.

    Parameters
    ----------
    X_train : ndarray, shape (n_samples,)
        Training inputs (days).
    y_train : ndarray, shape (n_samples,)
        Training outputs (temperatures).
    X_test : ndarray, shape (n_test,)
        Test inputs (days).
    l : float
        Length scale.
    nu : float, optional
        Smoothness parameter for the Matern kernel (1.5 or 2.5 supported).
    sigma : float, optional
        Variance scale.

    Returns
    -------
    mu : ndarray
        Predicted mean.
    cov : ndarray
        Predictive covariance.
    """
    K = covariance_matrix(X_train, X_train, kernel_func=matern_kernel,
                          length_scale=l, nu=nu, sigma=sigma) + 1e-5 * np.eye(len(X_train))
    K_inv = np.linalg.inv(K)

    K_s = covariance_matrix(X_test, X_train, kernel_func=matern_kernel,
                            length_scale=l, nu=nu, sigma=sigma)
    K_ss = covariance_matrix(X_test, X_test, kernel_func=matern_kernel,
                             length_scale=l, nu=nu, sigma=sigma) + 1e-5 * np.eye(len(X_test))

    mu = np.dot(K_s, np.dot(K_inv, y_train))
    cov = K_ss - np.dot(K_s, np.dot(K_inv, K_s.T))
    return mu, cov


def mse(y_true, y_pred):
    """
    Mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)

