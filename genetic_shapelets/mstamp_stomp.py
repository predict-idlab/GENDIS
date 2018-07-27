# -*- coding: utf-8 -*-
"""
@author: Michael Yeh
"""

# TAKEN FROM: https://github.com/mcyeh/mstamp/blob/master/Python/mstamp_stomp.py

from __future__ import print_function
import time
import numpy as np


_EPS = 1e-14


def mstamp(seq, sub_len, return_dimension=False):
    """ multidimensional matrix profile with mSTAMP (stomp based)

    Parameters
    ----------
    seq : numpy matrix, shape (n_dim, seq_len)
        input sequence
    sub_len : int
        subsequence length
    return_dimension : bool
        if True, also return the matrix profile dimension. It takses O(d^2 n)
        to store and O(d^2 n^2) to compute. (default is False)

    Returns
    -------
    matrix_profile : numpy matrix, shape (n_dim, sub_num)
        matrix profile
    profile_index : numpy matrix, shape (n_dim, sub_num)
        matrix profile index
    profile_dimension : list, optional, shape (n_dim)
        matrix profile dimension, this is only returned when return_dimension
        is True

    Notes
    -----
    C.-C. M. Yeh, N. Kavantzas, and E. Keogh, "Matrix Profile VI: Meaningful
    Multidimensional Motif Discovery," IEEE ICDM 2017.
    https://sites.google.com/view/mstamp/
    http://www.cs.ucr.edu/~eamonn/MatrixProfile.html
    """
    if sub_len < 4:
        raise RuntimeError('Subsequence length (sub_len) must be at least 4')
    exc_zone = sub_len // 2
    seq = np.array(seq, dtype=float, copy=True)

    if seq.ndim == 1:
        seq = np.expand_dims(seq, axis=0)

    seq_len = seq.shape[1]
    sub_num = seq.shape[1] - sub_len + 1
    n_dim = seq.shape[0]
    skip_loc = np.zeros(sub_num, dtype=bool)
    for i in range(sub_num):
        if not np.all(np.isfinite(seq[:, i:i + sub_len])):
            skip_loc[i] = True
    seq[~np.isfinite(seq)] = 0

    drop_val = 0
    matrix_profile = np.empty((n_dim, sub_num))
    matrix_profile[:] = np.inf
    profile_index = -np.ones((n_dim, sub_num), dtype=int)
    seq_freq = np.empty((n_dim, seq_len * 2), dtype=np.complex128)
    seq_mu = np.empty((n_dim, sub_num))
    seq_sig = np.empty((n_dim, sub_num))
    if return_dimension:
        profile_dimension = []
        for i in range(n_dim):
            profile_dimension.append(np.empty((i + 1, sub_num), dtype=int))
    for i in range(n_dim):
        seq_freq[i, :], seq_mu[i, :], seq_sig[i, :] = \
            _mass_pre(seq[i, :], sub_len)

    dist_profile = np.empty((n_dim, sub_num))
    last_product = np.empty((n_dim, sub_num))
    first_product = np.empty((n_dim, sub_num))
    drop_val = np.empty(n_dim)
    que_sum = np.empty(n_dim)
    que_sq_sum = np.empty(n_dim)
    que_sig = np.empty(n_dim)
    tic = time.time()
    for i in range(sub_num):
        cur_prog = (i + 1) / sub_num
        time_left = ((time.time() - tic) / (i + 1)) * (sub_num - i - 1)
        #print('\rProgress [{0:<50s}] {1:5.1f}% {2:8.1f} sec'
        #      .format('#' * int(cur_prog * 50),
        #              cur_prog * 100, time_left), end="")
        for j in range(n_dim):
            que = seq[j, i:i + sub_len]
            if i == 0:
                (dist_profile[j, :], last_product[j, :],
                 que_sum[j], que_sq_sum[j], que_sig[j]) = \
                    _mass(seq_freq[j, :], que, seq_len, sub_len,
                          seq_mu[j, :], seq_sig[j, :])
                first_product[j, :] = last_product[j, :].copy()
            else:
                que_sum[j] = que_sum[j] - drop_val[j] + que[-1]
                que_sq_sum[j] = que_sq_sum[j] - drop_val[j]**2 + que[-1]**2
                que_mu = que_sum[j] / sub_len
                que_sig_sq = que_sq_sum[j] / sub_len - que_mu**2
                if que_sig_sq < _EPS:
                    que_sig_sq = _EPS
                que_sig[j] = np.sqrt(que_sig_sq)
                last_product[j, 1:] = (last_product[j, 0:-1] -
                                       seq[j, 0:seq_len - sub_len] *
                                       drop_val[j] +
                                       seq[j, sub_len:seq_len] * que[-1])
                last_product[j, 0] = first_product[j, i]
                dist_profile[j, :] = \
                    (2 * (sub_len - (last_product[j, :] -
                                     sub_len * seq_mu[j, :] * que_mu) /
                          (seq_sig[j, :] * que_sig[j])))
                dist_profile[j, dist_profile[j, :] < _EPS] = 0
            drop_val[j] = que[0]

        if skip_loc[i] or np.any(que_sig < _EPS):
            continue

        exc_zone_st = max(0, i - exc_zone)
        exc_zone_ed = min(sub_num, i + exc_zone)
        dist_profile[:, exc_zone_st:exc_zone_ed] = np.inf
        dist_profile[:, skip_loc] = np.inf
        dist_profile[seq_sig < _EPS] = np.inf

        dist_profile_dim = np.argsort(dist_profile, axis=0)
        dist_profile_sort = np.sort(dist_profile, axis=0)
        dist_profile_cumsum = np.zeros(sub_num)
        for j in range(n_dim):
            dist_profile_cumsum += dist_profile_sort[j, :]
            dist_profile_mean = dist_profile_cumsum / (j + 1)
            update_pos = dist_profile_mean < matrix_profile[j, :]
            profile_index[j, update_pos] = i
            matrix_profile[j, update_pos] = dist_profile_mean[update_pos]
            if return_dimension:
                profile_dimension[j][:, update_pos] = \
                    dist_profile_dim[:j + 1, update_pos]

    matrix_profile = np.sqrt(matrix_profile)
    if return_dimension:
        return matrix_profile, profile_index, profile_dimension
    else:
        return matrix_profile, profile_index,


def _mass_pre(seq, sub_len):
    """ pre-computation for iterative call to MASS

    Parameters
    ----------
    seq : numpy array
        input sequence
    sub_len : int
        subsequence length

    Returns
    -------
    seq_freq : numpy array
        sequence in frequency domain
    seq_mu : numpy array
        each subsequence's mu (mean)
    seq_sig : numpy array
        each subsequence's sigma (standard deviation)

    Notes
    -----
    This functions is modified from the code provided in the following URL
    http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
    """
    seq_len = len(seq)
    seq_pad = np.zeros(seq_len * 2)
    seq_pad[0:seq_len] = seq
    seq_freq = np.fft.fft(seq_pad)
    seq_cum = np.cumsum(seq_pad)
    seq_sq_cum = np.cumsum(np.square(seq_pad))
    seq_sum = (seq_cum[sub_len - 1:seq_len] -
               np.concatenate(([0], seq_cum[0:seq_len - sub_len])))
    seq_sq_sum = (seq_sq_cum[sub_len - 1:seq_len] -
                  np.concatenate(([0], seq_sq_cum[0:seq_len - sub_len])))
    seq_mu = seq_sum / sub_len
    seq_sig_sq = seq_sq_sum / sub_len - np.square(seq_mu)
    seq_sig = np.sqrt(seq_sig_sq)
    return seq_freq, seq_mu, seq_sig


def _mass(seq_freq, que, seq_len, sub_len, seq_mu, seq_sig):
    """ iterative call of MASS

    Parameters
    ----------
    seq_freq : numpy array
        sequence in frequency domain
    que : numpy array
        query
    seq_len : int
        sequence length
    sub_len : int
        subsequence length
    seq_mu : numpy array
        each subsequence's mu (mean)
    seq_sig : numpy array
        each subsequence's sigma (standard deviation)

    Returns
    -------
    dist_profile : numpy array
        distance profile
    last_product : numpy array
        cross term
    que_sum : float64
        query's sum
    que_sq_sum : float64
        query's squre sum
    que_sig : float64
        query's sigma (standard deviation)

    Notes
    -----
    This functions is modified from the code provided in the following URL
    http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
    """
    que = que[::-1]
    que_pad = np.zeros(seq_len * 2)
    que_pad[0:sub_len] = que
    que_freq = np.fft.fft(que_pad)
    product_freq = seq_freq * que_freq
    product = np.fft.ifft(product_freq)
    product = np.real(product)

    que_sum = np.sum(que)
    que_sq_sum = np.sum(np.square(que))
    que_mu = que_sum / sub_len
    que_sig_sq = que_sq_sum / sub_len - que_mu**2
    if que_sig_sq < _EPS:
        que_sig_sq = _EPS
    que_sig = np.sqrt(que_sig_sq)

    dist_profile = (2 * (sub_len - (product[sub_len - 1:seq_len] -
                                    sub_len * seq_mu * que_mu) /
                         (seq_sig * que_sig)))
    last_product = product[sub_len - 1:seq_len]
    return dist_profile, last_product, que_sum, que_sq_sum, que_sig
