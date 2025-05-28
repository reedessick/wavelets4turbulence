"""a module that houses utilities for computing moments from samples
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
            # compute second moments carefully to try to avoid overflows

            ### compute the second moment
            if np.any(samples>0):
                log_samples = (index[i]+index[j]) * np.log(samples[samples>0]) # only include non-zero samples
                max_samples = np.max(log_samples)
                m2 = np.exp(np.log(np.sum(np.exp(log_samples-max_samples))) + max_samples - np.log(num_samples))
            else:
                m2 = m[i]*m[j] # zeroes the variance

            # now assemble the variances
            c[i,j] = c[j,i] = (m2 - m[i]*m[j]) / num_samples

    # return
    return index, m, c

#------------------------

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
    raise NotImplementedError('compute covariance matrix!')

    # return
    return index, mom, cov
