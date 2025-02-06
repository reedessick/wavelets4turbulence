"""a module that houses basic utilities, like estimating moments from monte carlo samples
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np
from scipy.special import comb # comb(n, k) = "n choose k" = n! / ((n-k)! k!)

#-------------------------------------------------

def moments(samples, index):
    """estimate moments of samples for each value in index (which should be an iterable). For example, index=[1,2] will compute the 1st and second moment of samples. Also estimates the covariance matrix between the estimators for the requested moments.
    """
    index = np.array(sorted(index), dtype=int)

    num_index = len(index)
    num_samples = len(samples)

    # compute point estimates
    m = np.array([np.sum(samples**ind)/num_samples for ind in index], dtype=float)

    # compute covariance matrix
    c = np.empty((num_index, num_index), dtype=float)
    for i in range(num_index):
        for j in range(i+1):
            # note, this may repeat some sums, but that shouldn't be much extra overhead
            c[i,j] = c[j,i] = (np.sum(samples**(index[i]+index[j]))/num_samples - m[i]*m[j]) / num_samples

    # return
    return index, m, c

def central_moments(samples, index):
    """returns the central moments instead of just the moments
    """
    index = np.array(sorted(index), dtype=int)
    num_index = len(index)

    # compute all moments up to the maximum index requested
    i, m, c = moments(samples, range(1,index[-1]+1))

    # compute point estimates
    mom = np.zeros(num_index, dtype=float)
    for j, ind in enumerate(index):
        for k in range(0, ind+1): # iterate over terms
            mom[ind] += m[k] * (-m[0])**(ind-k) * comb(ind, k)

    # compute covariance matrix
    raise NotImplementedError

    # return
    return index, mom, cov
