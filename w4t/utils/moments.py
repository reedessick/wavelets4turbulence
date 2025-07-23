"""a module that houses utilities for computing moments from samples
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np
from scipy.special import comb # comb(n, k) = "n choose k" = n! / ((n-k)! k!)

### non-standard libraries
from w4t.utils.utils import default_map2scalar

#-------------------------------------------------

def direct_isotropic_structure_function(
        array,
        scale,
        index,
        map2scalar=default_map2scalar,
        use_abs=True,
        increment=1,
        verbose=False,
        Verbose=False,
    ):
    """average over cartesian directions to estimate isotropic structure function
    """
    verbose |= Verbose

    # iterate over dimensions
    ndim = len(array.shape) - 1

    if verbose:
        print('computing isotropic structure function directly')

    mom = []
    cov = []
    for dim in range(ndim):
        index, m, c = direct_structure_function(
            array,
            dim,
            scale,
            index,
            map2scalar=map2scalar,
            use_abs=use_abs,
            increment=increment,
            verbose=Verbose,
        )
        mom.append(m)
        cov.append(c)

    # average
    if verbose:
        print('averaging moments over dimensions')

    mom, cov = average_moments(mom, cov)

    # return
    return index, mom, cov

#------------------------

def direct_structure_function(
        array,
        dim,
        scale,
        index,
        map2scalar=default_map2scalar,
        use_abs=True,
        increment=1,
        verbose=False,
    ):
    """directly estimate the structure function along dimension "dim" at length "scale"
    """
    assert (0 <= dim) and (dim < len(array.shape)), 'bad dimension (dim=%d) for ndim=%d' % (dim, len(array.shape))

    if verbose:
        print('computing moments directly for dim=%d for scale=%d with increment=%d' % (dim, scale, increment))

    # figure out the relevant indexes
    inds = np.arange(0, array.shape[dim+1]-scale, increment)

    # compute the differences with step size "scale", take moments, and return
    samples = map2scalar(np.take(array, inds+scale, axis=dim+1) - np.take(array, inds, axis=dim+1)).flatten()
    if use_abs:
        samples = np.abs(samples)

    return moments(
        samples,
        index,
        central=False,
        verbose=verbose,
    )

#-------------------------------------------------

def average_moments(mom, cov):
    """average moments and update covariance
    """
    # average individual measurement undertainties and add the variance between estimates
    cov = (np.mean(cov, axis=0) + np.cov(np.transpose(mom))) / len(mom)
    # take the average of the means
    mom = np.mean(mom, axis=0)

    return mom, cov

#-------------------------------------------------

def moments(samples, index, central=False, verbose=False):
    """estimate moments of samples for each value in index (which should be an iterable). For example, index=[1,2] will compute the 1st and second moment of samples. Also estimates the covariance matrix between the estimators for the requested moments.
    """
    index = np.array(index, dtype=int)

    num_index = len(index)
    num_samples = len(samples)

    if central: # compute central moments; uncertainty estimates do not include uncertainty in the mean
        samples = samples - np.mean(samples)

    # compute point estimates
    m = np.array([np.sum(samples**ind)/num_samples for ind in index], dtype=float)

    # compute covariance matrix
    c = np.empty((num_index, num_index), dtype=float)
    for i in range(num_index):
        for j in range(i+1):

            if np.any(samples!=samples[0]): # there is more than 1 unique value
                # compute this in a silly way to avoid overflow errors
                c[i,j] = c[j,i] = m[i]*m[j]*np.sum((1 - samples**index[i]/m[i]) * (1 - samples**index[j]/m[j])) / (num_samples-1) / num_samples

            else:
                c[i,j] = c[j,i] = 0

    if np.any(np.diag(c) < 0) or np.any(np.isnan(c)):
        raise RuntimeError('''\
bad covariance matrix!
num_samples = %d
samples = %s
index = %s
mom = %s
cov = %s''' % (num_samples, samples, str(index), str(m), str(c)))

    if verbose:
        print('    index :', index)
        print('    mom   :', m)
        print('    std   :', np.diag(c)**0.5)

    # return
    return index, m, c

#------------------------

def scaling_exponent(scales, mom, std, deg=1):
    """perform a linear fit of log(mom) as a function of log(scales) with uncertainties in mom given by stdv
    """
    return np.polyfit(np.log(scales), np.log(mom), deg=deg, w=mom/std)
