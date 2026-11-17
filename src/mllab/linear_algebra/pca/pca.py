"""Principal Component Analysis built from first principles.

Notes
-----
- X has shape (m, n): observations are the rows, features are the columns.
- Eigenvectors are returned as COLUMNS, sorted by largest eigenvalue first,
  so eigenvectors[:, 0] is PC1.
- The pipeline is: fit_pca -> project_pca -> reconstruct_pca.
"""

import numpy as np

# Import my covariance functions
from mllab.linear_algebra.covariance.covariance import (
    feature_means,
    center,
    covariance_matrix,
)


def fit_pca(X):
    """Fit PCA to a dataset.

    Param
    ----------
    X : Dataset with m observations (rows) and n features (columns).
    """

    # Compute feature-wise mean (mu)
    mean = feature_means(X)
    # Center X
    Xc = X - mean
    # Compute sample covariance matrix, C = Xc^T Xc / (m - 1), shape (n, n)
    C = covariance_matrix(Xc)

    # eigendecompose C and sort largest eigenvalues first
    # TODO: Switch this out for self-written eigenvalue/vector computations once
    # A1's functions have been expanded to large matrices than 2x2
    eigenvalues, eigenvectors = _eigendecompose(C)

    # return mean, eigenvalues, eigenvectors
    return mean, eigenvalues, eigenvectors


def project_pca(X, mean, eigenvectors, p):
    """Project data onto the first p principal components.
    Computes: scores Z = (X - 1 mu^T) V_p.

    Param
    ----------
    X :             Dataset to project.
    mean :          Feature means returned by fit_pca.
    eigenvectors :  Principal components.
    p :             Number of principal components to keep, 1 <= p <= n.
    """
    _check_p(p, eigenvectors)

    # keep only first p principal components: V_p, shape (n, p)
    V_p = eigenvectors[:, :p]

    # Center X and then project each row onto V_p
    return center(X, mean) @ V_p


def reconstruct_pca(Z, mean, eigenvectors, p):
    """Reconstruct data in the original feature space from PCA scores.
    Computes: X_hat = Z V_p^T + 1 mu^T. 

    Param
    ----------
    Z :             Scores returned by project_pca.
    mean :          Feature means.
    eigenvectors :  Principal components.
    p :             Number of principal components used to make Z.
    """
    _check_p(p, eigenvectors)

    V_p = eigenvectors[:, :p]
    return Z @ V_p.T + mean


def explained_variance_ratio(eigenvalues, p=None):
    """Share of total variance captured by the principal components.

    Param
    ----------
    eigenvalues : Eigenvalues from fit_pca.
    p : If given, return the cumulative share for the first p components.
    """
    # Compute each value's contribution to the total reation
    ratios = eigenvalues / np.sum(eigenvalues)
    # Accumulate ratios as the EVR and return it
    if p is None:
        return ratios
    return float(np.sum(ratios[:p]))


def mse(X, X_hat):
    """
    Mean squared error over every entry of the data.
 
    Params
    ----------
    X :         Original data.
    X_hat :     Reconstructed data.
 
    Returns
    -------
    mean((X - X_hat)^2) over all m * n entries.
    
    """
    return float(np.mean((X - X_hat)**2))



def _eigendecompose(C):
    """Eigenvalues and eigenvectors of a symmetric matrix, largest first."""
    
    # eigh returns eigenvalues in ASCENDING order, so reverse both outputs.
    values, vectors = np.linalg.eigh(C)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def _check_p(p, eigenvectors):
    """Raise ValueError unless 1 <= p <= number of components."""
    n = eigenvectors.shape[1]
    if not 1 <= p <= n:
        raise ValueError(f"p must be between 1 and {n}, got {p}.")