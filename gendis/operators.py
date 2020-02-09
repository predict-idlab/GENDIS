import numpy as np

from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import sigma_gak, cdist_gak, dtw_subsequence_path, cdist_dtw
from tslearn.clustering import GlobalAlignmentKernelKMeans
from tslearn.barycenters import euclidean_barycenter

from scipy.spatial.distance import euclidean

from deap import tools

##########################################################################
#                       Initialization operators                         #
##########################################################################
# Interface
# ---------
# INPUT: 
#    - X (np.array)
#    - n_shapelets (int)
# OUTPUT: 
#    - shapelets (np.array)
def random_shapelet(X, n_shapelets, min_len, max_len):
    """Extract a random subseries from the training set"""
    shaps = []
    for _ in range(n_shapelets):
        rand_row = np.random.randint(X.shape[0])
        rand_length = np.random.randint(4, min(min_len, max_len))
        rand_col = np.random.randint(min_len - rand_length)
        shaps.append(X[rand_row][rand_col:rand_col+rand_length])
    if n_shapelets > 1:
        return np.array(shaps)
    else:
        return np.array(shaps[0])


def kmeans(X, n_shapelets, min_len, max_len, n_draw=None):
    """Sample subseries from the timeseries and apply K-Means on them"""
    # Sample `n_draw` subseries of length `shp_len`
    if n_shapelets == 1:
        return random_shapelet(X, n_shapelets, min_len, max_len)
    if n_draw is None:
        n_draw = max(n_shapelets, int(np.sqrt(len(X))))
    shp_len = np.random.randint(4, min(min_len, max_len))
    indices_ts = np.random.choice(len(X), size=n_draw, replace=True)
    start_idx = np.random.choice(min_len - shp_len, size=n_draw, 
                                 replace=True)
    end_idx = start_idx + shp_len

    subseries = np.zeros((n_draw, shp_len))
    for i in range(n_draw):
        subseries[i] = X[indices_ts[i]][start_idx[i]:end_idx[i]]

    tskm = TimeSeriesKMeans(n_clusters=n_shapelets, metric="euclidean", 
                            verbose=False)
    return tskm.fit(subseries).cluster_centers_

##########################################################################
#                         Mutatation operators                           #
##########################################################################
# Interface
# ---------
# INPUT: 
#    - shapelets (np.array)
# OUTPUT: 
#    - new_shapelets (np.array)
def add_noise(shapelets, toolbox):
    """Add random noise to a random shapelet"""
    rand_shapelet = np.random.randint(len(shapelets))
    tools.mutGaussian(shapelets[rand_shapelet], 
                      mu=0, sigma=0.1, indpb=0.15)

    return shapelets,


def add_shapelet(shapelets, toolbox):
    """Add a shapelet to the individual"""
    shapelets.append(toolbox.create(n_shapelets=1))

    return shapelets,


def remove_shapelet(shapelets, toolbox):
    """Remove a random shapelet from the individual"""
    if len(shapelets) > 1:
        rand_shapelet = np.random.randint(len(shapelets))
        shapelets.pop(rand_shapelet)

    return shapelets,


def mask_shapelet(shapelets, toolbox):
    """Mask part of a random shapelet from the individual"""
    rand_shapelet = np.random.randint(len(shapelets))
    if len(shapelets[rand_shapelet]) > 4:
        rand_start = np.random.randint(len(shapelets[rand_shapelet]) - 4)
        rand_end = np.random.randint(rand_start + 4, len(shapelets[rand_shapelet]))
        shapelets[rand_shapelet] = shapelets[rand_shapelet][rand_start:rand_end]

    return shapelets,

##########################################################################
#                         Crossover operators                            #
##########################################################################
# Interface
# ---------
# INPUT: 
#    - ind1 (np.array)
#    - ind2 (np.array)
# OUTPUT: 
#    - new_ind1 (np.array)
#    - new_ind2 (np.array)


def merge(ts1, ts2):
    if len(ts1) > len(ts2):
        start = np.random.randint(len(ts1) - len(ts2))
        centroid = euclidean_barycenter([ts1[start:start+len(ts2):], ts2]).flatten()
    elif len(ts2) > len(ts1):
        start = np.random.randint(len(ts2) - len(ts1))
        centroid = euclidean_barycenter([ts2[start:start+len(ts1):], ts1]).flatten()
    else:
        start = 0
        centroid = euclidean_barycenter([ts1, ts2]).flatten()

    return centroid, start

def merge_crossover(ind1, ind2, p=0.25):
    """Merge shapelets from one set with shapelets from the other"""
    # Construct a pairwise similarity matrix using GAK
    _all = list(ind1) + list(ind2)
    similarity_matrix = cdist_gak(ind1, ind2, sigma=sigma_gak(_all))

    # Iterate over shapelets in `ind1` and merge them with shapelets
    # from `ind2`
    for row_idx in range(similarity_matrix.shape[0]):
        # Remove all elements equal to 1.0
        mask = similarity_matrix[row_idx, :] != 1.0
        non_equals = similarity_matrix[row_idx, :][mask]
        if len(non_equals):
            # Get the timeseries most similar to the one at row_idx
            max_col_idx = np.argmax(non_equals)
            ts1 = list(ind1[row_idx]).copy()
            ts2 = list(ind2[max_col_idx]).copy()
            # Merge them and remove nans
            ind1[row_idx] = euclidean_barycenter([ts1, ts2])
            ind1[row_idx] = ind1[row_idx][~np.isnan(ind1[row_idx])]

    # Apply the same for the elements in ind2
    for col_idx in range(similarity_matrix.shape[1]):
        mask = similarity_matrix[:, col_idx] != 1.0
        non_equals = similarity_matrix[:, col_idx][mask]
        if len(non_equals):
            max_row_idx = np.argmax(non_equals)
            ts1 = list(ind1[max_row_idx]).copy()
            ts2 = list(ind2[col_idx]).copy()
            ind2[col_idx] = euclidean_barycenter([ts1, ts2])
            ind2[col_idx] = ind2[col_idx][~np.isnan(ind2[col_idx])]

    return ind1, ind2


import matplotlib.pyplot as plt

def random_merge_crossover(ind1, ind2, p=0.25):
    """Merge shapelets from one set with shapelets from the other"""
    # Construct a pairwise similarity matrix using GAK
    new_ind1, new_ind2 = [], []
    np.random.shuffle(ind1)
    np.random.shuffle(ind2)
    for shap1, shap2 in zip(ind1, ind2):
        if len(shap1) > 4 and len(shap2) > 4 and np.random.random() < p:
            max_size = min(len(shap1), len(shap2))
            merge_len = np.random.randint(1, max_size)
            shap1_start = np.random.randint(len(shap1) - merge_len)
            shap2_start = np.random.randint(len(shap2) - merge_len)

            shap1 = np.concatenate((
                shap1[:shap1_start].flatten(), 
                euclidean_barycenter([
                    shap1[shap1_start:shap1_start+merge_len], 
                    shap2[shap2_start:shap2_start+merge_len]
                ]).flatten(), 
                shap1[shap1_start+merge_len:].flatten()
            ))

            shap2 = np.concatenate((
                shap2[:shap2_start].flatten(), 
                euclidean_barycenter([
                    shap1[shap1_start:shap1_start+merge_len], 
                    shap2[shap2_start:shap2_start+merge_len]
                ]).flatten(), 
                shap2[shap2_start+merge_len:].flatten()
            ))

        new_ind1.append(shap1)
        new_ind2.append(shap2)

    return new_ind1, new_ind2

    # other_shaps = list(ind2[:])
    # for ix, shap in enumerate(ind1):
    #     if np.random.random() < p:
    #         rand_other_shap_ix = np.random.choice(range(len(other_shaps)))
    #         rand_other_shap = other_shaps[rand_other_shap_ix].flatten()
    #         shap = shap.flatten()

    #         # plt.figure()
    #         # plt.plot(shap)
    #         # plt.plot(rand_other_shap)
    #         # plt.title('Before Random Merge')
    #         # plt.show()

    #         centroid, start = merge(shap, rand_other_shap)
    #         ind1[ix] = np.concatenate((shap[:start], centroid, 
    #         						   shap[start+len(centroid):]))

    #         # plt.figure()
    #         # plt.plot(ind1[ix])
    #         # plt.title('After Random Merge')
    #         # plt.show()
    #         # input()

    # other_shaps = list(ind1[:])
    # for ix, shap in enumerate(ind2):
    #     if np.random.random() < p:
    #         rand_other_shap_ix = np.random.choice(range(len(other_shaps)))
    #         rand_other_shap = other_shaps[rand_other_shap_ix].flatten()
    #         shap = shap.flatten()
    #         centroid, start = merge(shap, rand_other_shap)
    #         ind2[ix] = np.concatenate((shap[:start], centroid, 
    #         						   shap[start+len(centroid):]))

    # return ind1, ind2


def point_crossover(ind1, ind2):
    """Apply one- or two-point crossover on the shapelet sets"""
    if len(ind1) > 1 and len(ind2) > 1:
        if np.random.random() < 0.5:
            ind1, ind2 = tools.cxOnePoint(list(ind1), list(ind2))
        else:
            ind1, ind2 = tools.cxTwoPoint(list(ind1), list(ind2))
    
    return ind1, ind2

def shap_point_crossover(ind1, ind2, p=0.25):
    """Apply one--point crossover on pairs of shapelets from the sets"""
    new_ind1, new_ind2 = [], []
    np.random.shuffle(ind1)
    np.random.shuffle(ind2)

    for shap1, shap2 in zip(ind1, ind2):

        if len(shap1) > 4 and len(shap2) > 4 and np.random.random() < p:

            # plt.figure()
            # plt.plot(shap1)
            # plt.plot(shap2)
            # plt.title('Before Point CX')
            # plt.show()
            shap1, shap2 = tools.cxOnePoint(list(shap1), list(shap2))

            # plt.figure()
            # plt.plot(shap1)
            # plt.plot(shap2)
            # plt.title('After Point CX')
            # plt.show()
            # input()

        new_ind1.append(shap1)
        new_ind2.append(shap2)

    if len(ind1) < len(ind2):
        new_ind2 += ind2[len(ind1):]
    else:
        new_ind1 += ind1[len(ind2):]

    return new_ind1, new_ind2