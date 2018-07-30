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


def sdist_metrics(u, l, S_x, S_x2, S_y, S_y2, M):
    min_dist = np.inf
    for v in range(len(S_y) - l):
        dist = pearson_dist_metrics(u, v, l, S_x, S_x2, S_y, S_y2, M)
        min_dist = min(dist, min_dist)
    return min_dist


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