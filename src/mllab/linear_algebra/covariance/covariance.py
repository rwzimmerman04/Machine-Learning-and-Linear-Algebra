"""Mean, centering, covariance for data matrices.

Convention used throughout the repository: X has shape (m, n) with m observations
as rows and n features as columns.
"""

import numpy as np

def feature_means(X):
    """Return the feature-wise mean vector mu, shape (n,)."""
    return np.mean(X, axis=0)


def center(X, mean=None):
    """Return X - 1 mu^T. Provide `mean` to reuse a mean computed beforehand, 
    otherwise compute the mean of X."""
    if mean is None:
        # COmpute mean of X
        mean = feature_means(X)
    return X - mean


def covariance_matrix(Xc, sample=1):
    """Return the (n, n) covariance matrix (1 / (m - sample)) * Xc^T Xc.

    sample=1 gives the sample covariance, sample=0 gives the population covariance.
    """
    m = Xc.shape[0]     # Get number of observations in Xc
    return (Xc.T @ Xc) / (m - sample)

