import numpy as np
import pandas as pd
from scipy.stats import (zscore, pearsonr, entropy,
                         kruskal, f_oneway, median_test)
import time
from collections import Counter, defaultdict
from sklearn.feature_selection import mutual_info_classif
import math

from sklearn.neighbors import KDTree, BallTree
from scipy.spatial.distance import pdist, euclidean


def z_norm(x):
    """Normalize time series such that it has zero mean and unit variance
    >>> list(np.around(z_norm([0, 3, 6]), 4))
    [-1.2247, 0.0, 1.2247]
    """
    mu_x = np.mean(x)
    sigma_x = np.std(x)
    if sigma_x == 0: sigma_x = 1
    return (x - mu_x) / sigma_x


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


def kruskal_score(L):
    score = kruskal(*list(get_distances_per_class(L).values()))[0]
    if not pd.isnull(score):
        return (score,)
    else:
        return (float('-inf'),)


def f_score(L):
    score = f_oneway(*list(get_distances_per_class(L).values()))[0]
    if not pd.isnull(score):
        return (score,)
    else:
        return (float('-inf'),)


def mood_median(L):
    score = median_test(*list(get_distances_per_class(L).values()))[0]
    if not pd.isnull(score):
        return (score,)
    else:
        return (float('-inf'),)


def get_distances_per_class(L):
    distances_per_class = defaultdict(list)
    for dist, label in L:
        distances_per_class[label].append(dist)
    return distances_per_class


def calculate_metric_arrays(x, y):
    """Calculate five statistic arrays:
        * S_x:  contains the cumulative sum of elements of x
        * S_x2: contains the cumulative sum of squared elements of x
        * S_y:  contains the cumulative sum of elements of y
        * S_y2: contains the cumulative sum of squared elements of y
        * M:    stores the sum of products of subsequences of x and y
    """
    S_x = np.append([0], np.cumsum(x))
    S_x2 = np.append([0], np.cumsum(np.power(x, 2)))
    S_y = np.append([0], np.cumsum(y))
    S_y2 = np.append([0], np.cumsum(np.power(y, 2)))

    # TODO: can we calculate M more efficiently (numpy or scipy)??
    M = np.zeros((len(x) + 1, len(y) + 1))
    for u in range(len(x)):
        for v in range(len(y)):
            t = abs(u-v)
            if u > v:
                M[u+1, v+1] = M[u, v] + x[v+t]*y[v]
            else:
                M[u+1, v+1] = M[u, v] + x[u]*y[u+t]

    return S_x, S_x2, S_y, S_y2, M


def pearson_metrics(u, v, l, S_x, S_x2, S_y, S_y2, M):
    """Calculate the correlation between two time series. Calculate
    the mean and standard deviations by using the statistic arrays."""
    mu_x = (S_x[u + l] - S_x[u]) / l
    mu_y = (S_y[v + l] - S_y[v]) / l
    sigma_x = np.sqrt((S_x2[u + l] - S_x2[u]) / l - mu_x ** 2)
    sigma_y = np.sqrt((S_y2[v + l] - S_y2[v]) / l - mu_y ** 2)
    xy = M[u + l, v + l] - M[u, v]
    if sigma_x == 0 or pd.isnull(sigma_x): sigma_x = 1
    if sigma_y == 0 or pd.isnull(sigma_y): sigma_y = 1
    return min(1, (xy - (l * mu_x * mu_y)) / (l * sigma_x * sigma_y))


def pearson_dist_metrics(u, v, l, S_x, S_x2, S_y, S_y2, M):
    return np.sqrt(2 * (1 - pearson_metrics(u, v, l, S_x, S_x2, S_y, S_y2, M)))


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
    

def sdist_metrics(u, l, S_x, S_x2, S_y, S_y2, M):
    min_dist = np.inf
    for v in range(len(S_y) - l):
        dist = pearson_dist_metrics(u, v, l, S_x, S_x2, S_y, S_y2, M)
        min_dist = min(dist, min_dist)
    return min_dist


def sdist_with_pos(x, y):
    if len(y) < len(x): return sdist_with_pos(y, x)
    min_dist = np.inf
    norm_x = z_norm(x)
    best_pos = 0
    for j in range(len(y) - len(x) + 1):
        norm_y = z_norm(y[j:j+len(x)])
        dist = euclidean(norm_x, norm_y)
        if dist < min_dist:
            min_dist = dist
            best_pos = j
    return min_dist, best_pos


def information_gain(prior_entropy, left_counts, right_counts):
    N_left = sum(left_counts)
    N_right = sum(right_counts)
    N = N_left + N_right
    left_entropy = N_left/N * entropy(left_counts)
    right_entropy = N_right/N * entropy(right_counts)
    return prior_entropy - left_entropy - right_entropy


def calculate_ig(L):
    L = sorted(L, key=lambda x: x[0])
    all_labels = [x[1] for x in L]
    classes = set(all_labels)

    left_counts, right_counts, all_counts = {}, {}, {}
    for c in classes: all_counts[c] = 0

    for label in all_labels: all_counts[label] += 1
    prior_entropy = entropy(list(all_counts.values()))

    max_tau = (L[0][0] + L[1][0]) / 2
    max_gain, max_gap = float('-inf'), float('-inf')
    updated = False
    for k in range(len(L) - 1):
        for c in classes: 
            left_counts[c] = 0
            right_counts[c] = 0

        if L[k][0] == L[k+1][0]: continue
        tau = (L[k][0] + L[k + 1][0]) / 2
        
        left_labels = all_labels[:k+1]
        right_labels = all_labels[k+1:]

        for label in left_labels: left_counts[label] += 1
        for label in right_labels: right_counts[label] += 1

        ig = information_gain(
            prior_entropy, 
            list(left_counts.values()), 
            list(right_counts.values())
        )
        g = np.mean([x[0] for x in L[k+1:]]) - np.mean([x[0] for x in L[:k+1]])
        
        if ig > max_gain or (ig == max_gain and g > max_gap):
            max_tau, max_gain, max_gap = tau, ig, g

    return (max_gain, max_gap)


def class_scatter_matrix(X, y):
    # Works faster than Linear Regression and correlates well with predictive performance (e.g. accuracy)
    # FROM: https://datascience.stackexchange.com/questions/11554/varying-results-when-calculating-scatter-matrices-for-lda
    # Construct a mean vector per class
    mean_vecs = {}
    for label in set(y):
        mean_vecs[label] = np.mean(X[y==label], axis=0)
        
    # Construct the within class matrix (S_w)
    d = X.shape[1]
    S_w = np.zeros((d, d))
    for label, mv in zip(set(y), mean_vecs):
        class_scatter = np.cov(X[y==label].T)
        S_w += class_scatter
        
    # Construct an overall mean vector
    mean_overall = np.mean(X, axis=0)
    
    # Construct the between class matrix (S_b)
    S_b = np.zeros((d, d))
    for i in mean_vecs:
        mean_vec = mean_vecs[i]
        n = X[y==i, :].shape[0]
        mean_vec = mean_vec.reshape(d, 1)
        mean_overall = mean_overall.reshape(d, 1)
        S_b += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
        
    return np.trace(S_b) / np.trace(S_w + S_b)


def bhattacharyya(X, y, cells_per_dim=10):
    # Calculate lower and upper bound for each dimension
    bounds = {}
    widths = {}
    col_to_del = []
    cntr = 0
    for d in range(X.shape[1]):
        mi, ma = min(X[:, d]), max(X[:, d])
        if (ma - mi) < 1e-5:
            col_to_del.append(d)
        else:
            bounds[cntr] = (mi, ma)
            widths[cntr] = (ma - mi) / cells_per_dim
            cntr += 1
    
    X = np.delete(X, col_to_del, axis=1)
    #print(X, col_to_del)

    if X.shape[1] == 0:
        return 1
            
    # For each datapoint, calculate its cell in the hypercube
    cell_assignment_counts = []
    label_to_idx = {}
    for i, l in enumerate(set(y)): 
        cell_assignment_counts.append(defaultdict(int))
        label_to_idx[l] = i
        
    unique_assignments = set()
    for point_idx, l in zip(range(X.shape[0]), y):
        assignment = []
        for dim_idx in range(X.shape[1]):
            val = X[point_idx, dim_idx]
            cell = (val - bounds[dim_idx][0])//widths[dim_idx]
            assignment.append(cell)
        cell_assignment_counts[label_to_idx[l]][tuple(assignment)] += 1
        unique_assignments.add(tuple(assignment))
        
    totals = {}
    for l in set(y):
        totals[l] = sum(cell_assignment_counts[label_to_idx[l]].values())
    
    dist = 0
    for assign in unique_assignments:
        temp = 1
        for l in set(y):
            temp *= cell_assignment_counts[label_to_idx[l]][assign] / totals[l]
        dist += (temp * (temp != 1)) ** (1 / len(totals))
    
    return dist

def get_threshold(L):
    L = sorted(L, key=lambda x: x[0])
    all_labels = [x[1] for x in L]
    classes = set(all_labels)

    left_counts, right_counts, all_counts = {}, {}, {}
    for c in classes: all_counts[c] = 0

    for label in all_labels: all_counts[label] += 1
    prior_entropy = entropy(list(all_counts.values()))

    max_tau = (L[0][0] + L[1][0]) / 2
    max_gain, max_gap = float('-inf'), float('-inf')
    updated = False
    for k in range(len(L) - 1):
        for c in classes: 
            left_counts[c] = 0
            right_counts[c] = 0

        if L[k][0] == L[k+1][0]: continue
        tau = (L[k][0] + L[k + 1][0]) / 2
        
        left_labels = all_labels[:k+1]
        right_labels = all_labels[k+1:]

        for label in left_labels: left_counts[label] += 1
        for label in right_labels: right_counts[label] += 1

        ig = information_gain(
            prior_entropy, 
            list(left_counts.values()), 
            list(right_counts.values())
        )
        g = np.mean([x[0] for x in L[k+1:]]) - np.mean([x[0] for x in L[:k+1]])
        
        if ig > max_gain or (ig == max_gain and g > max_gap):
            max_tau, max_gain, max_gap = tau, ig, g

    return max_tau