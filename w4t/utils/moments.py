"""a module that houses utilities for computing moments from samples
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np
from scipy.special import comb # comb(n, k) = "n choose k" = n! / ((n-k)! k!)

#-------------------------------------------------

def moments(samples, index, central=False):
    """estimate moments of samples for each value in index (which should be an iterable). For example, index=[1,2] will compute the 1st and second moment of samples. Also estimates the covariance matrix between the estimators for the requested moments.
    """
    index = np.array(index, dtype=int)

    num_index = len(index)
    num_samples = len(samples)

    if central: # compute central moments; uncertainty estimates do not include uncertainty in the mean
        mean = np.mean(samples)
    else:
        mean = 0

    # compute point estimates
    m = np.array([np.sum((samples-mean)**ind)/num_samples for ind in index], dtype=float)

    # compute covariance matrix
    c = np.empty((num_index, num_index), dtype=float)
    for i in range(num_index):
        for j in range(i+1):

            ### compute the second moment
            # compute second moments carefully to try to avoid overflows
            if np.any(samples!=samples[0]): # there is more than 1 unique value
                log_samples = (index[i]+index[j]) * np.log(samples[samples>0]) # only include non-zero samples
                max_samples = np.max(log_samples)
                m2 = np.exp(np.log(np.sum(np.exp(log_samples-max_samples))) + max_samples - np.log(num_samples))

                # now assemble the variances
                c[i,j] = c[j,i] = (m2 - m[i]*m[j]) / num_samples

            else:
                c[i,j] = c[j,i] = 0

    # return
    return index, m, c

#------------------------

def scaling_exponent(scales, mom, std):
    """perform a linear fit of log(mom) as a function of log(scales) with uncertainties in mom given by stdv
    """
    p = np.polyfit(np.log(scales), np.log(mom), deg=deg, weights=mom/std)
    return p[0] # return linear coefficient
