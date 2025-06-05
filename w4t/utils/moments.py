"""a module that houses utilities for computing moments from samples
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np
from scipy.special import comb # comb(n, k) = "n choose k" = n! / ((n-k)! k!)

#-------------------------------------------------

def direct_isotropic_structure_function(array, scale, index, verbose=False, Verbose=False):
    """average over cartesian directions to estimate isotropic structure function
    """
    verbose |= Verbose

    # iterate over dimensions
    ndim = len(array.shape)

    mom = []
    cov = []
    for dim in range(ndim):
        if verbose:
            print('estimating moments directly along dim=%d' % dim)

        index, m, c = direct_structure_function(array, dim, scale, index, verbose=Verbose)
        mom.append(m)
        cov.append(c)

    # average
    if verbose:
        print('averaging moments over dimensions')

    mom = np.mean(mom, axis=0)
    cov = np.sum(cov, axis=0) / ndim**2

    # return
    return index, mom, cov

#------------------------

def direct_structure_function(array, dim, scale, index, verbose=False):
    """directly estimate the structure function along dimension "dim" at length "scale"
    """
    assert (0 <= dim) and (dim < len(array.shape)), 'bad dimension (dim=%d) for ndim=%d' % (dim, len(array.shape))

    if verbose:
        print('computing moments for scale: %d' % scale)

    # figure out the relevant indexes
    inds = np.arange(array.shape[dim]-scale)

    # compute the differences with step size "scale", take moments, and return
    return moments(np.abs(np.take(array, inds+scale, axis=dim) - np.take(array, inds, axis=dim)), index, central=False)

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

            if np.any(samples!=samples[0]): # there is more than 1 unique value
                c[i,j] = c[j,i] = np.sum((samples**index[i]-m[i]) * (samples**index[j]-m[j])) / (num_samples-1)

            else:
                c[i,j] = c[j,i] = 0

    if np.any(np.diag(c) < 0):
        raise RuntimeError('''\
bad covariance matrix!
num_samples = %d
samples = %s
index = %s
mom = %s
cov = %s''' % (num_samples, samples, str(index), str(m), str(c)))

    # return
    return index, m, c

#------------------------

def scaling_exponent(scales, mom, std, deg=1):
    """perform a linear fit of log(mom) as a function of log(scales) with uncertainties in mom given by stdv
    """
    return np.polyfit(np.log(scales), np.log(mom), deg=deg, w=mom/std)
