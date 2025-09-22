"""
kernels.py

Common kernel functions for Gaussian Process Regression.
Supports spatial (RBF) and temporal (Matern) kernels.
"""

import numpy as np


# --- RBF Kernel ---
def rbf_kernel(x1, x2, length_scale=1.0):
    """
    Radial Basis Function (Gaussian) kernel.

    Parameters
    ----------
    x1, x2 : array-like
        Input points.
    length_scale : float
        Length scale of the kernel.

    Returns
    -------
    float
        Kernel value.
    """
    diff = np.linalg.norm(x1 - x2)
    return np.exp(-0.5 * (diff / length_scale) ** 2)


# --- Matern Kernel (nu=1.5 or 2.5) ---
def matern_kernel(x1, x2, length_scale=1.0, nu=1.5, sigma=1.0):
    """
    Matern kernel (for temporal or rougher processes).

    Parameters
    ----------
    x1, x2 : float or array-like
        Input points.
    length_scale : float
        Length scale.
    nu : float
        Smoothness parameter (1.5 or 2.5 supported).
    sigma : float
        Scale parameter.

    Returns
    -------
    float
        Kernel value.
    """
    r = np.abs(np.linalg.norm(x1 - x2)) / length_scale
    if nu == 1.5:
        return sigma**2 * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    elif nu == 2.5:
        return sigma**2 * (1 + np.sqrt(5) * r + 5.0 * r**2 / 3.0) * np.exp(-np.sqrt(5) * r)
    else:
        raise ValueError("Only nu=1.5 or nu=2.5 are supported for the Matern kernel.")


# --- Linear Kernel ---
def linear_kernel(x1, x2, c=0.0):
    """
    Linear kernel.

    Parameters
    ----------
    x1, x2 : array-like
        Input points.
    c : float
        Offset.

    Returns
    -------
    float
        Kernel value.
    """
    return np.dot(x1, x2) + c


# --- Covariance Matrix ---
def covariance_matrix(X1, X2, kernel_func=rbf_kernel, **kernel_params):
    """
    Compute covariance matrix between two sets of points.

    Parameters
    ----------
    X1, X2 : ndarray
        Input arrays of shape (n_samples, n_features).
    kernel_func : callable
        Kernel function to use.
    kernel_params : dict
        Additional parameters for the kernel function.

    Returns
    -------
    K : ndarray
        Covariance matrix of shape (len(X1), len(X2)).
    """
    K = np.zeros((len(X1), len(X2)))
    for i in range(len(X1)):
        for j in range(len(X2)):
            K[i, j] = kernel_func(X1[i], X2[j], **kernel_params)
    return K
