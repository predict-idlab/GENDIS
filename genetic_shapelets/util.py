import numpy as np
from scipy.spatial.distance import euclidean

def z_norm(x):
    """Normalize time series such that it has zero mean and unit variance
    >>> list(np.around(z_norm([0, 3, 6]), 4))
    [-1.2247, 0.0, 1.2247]
    """
    mu_x = np.mean(x)
    sigma_x = np.std(x)
    if sigma_x == 0: sigma_x = 1
    return (x - mu_x) / sigma_x


def sdist_no_norm(x, y):
    """No value normalization and no length normalization is applied
    >>> np.around(sdist_no_norm([1, 1, 1, 1, 1], [0, 0, 0]), 5)
    1.73205
    """
    if len(y) < len(x): return sdist_no_norm(y, x)
    min_dist = np.inf
    for j in range(len(y) - len(x) + 1):
        dist = euclidean(x, y[j:j+len(x)])
        min_dist = min(dist, min_dist)
    return min_dist 


def sdist(x, y):
    """A distance metric, where each subseries and timeseries are first
    z-normalized before calculating euclidean distance. We not apply
    length normalization.
    """
    if len(y) < len(x): return sdist(y, x)
    min_dist = np.inf
    norm_x = z_norm(x)
    for j in range(len(y) - len(x) + 1):
        norm_y = z_norm(y[j:j+len(x)])
        dist = euclidean(norm_x, norm_y)
        min_dist = min(dist, min_dist)
    return min_dist