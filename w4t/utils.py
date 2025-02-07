"""a module that houses basic utilities, like estimating moments from monte carlo samples
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np
from scipy.special import comb # comb(n, k) = "n choose k" = n! / ((n-k)! k!)

### non-standard libraries
try:
    from PLASMAtools.read_funcs.read import Fields
except ImportError:
    Fields = None

#-------------------------------------------------

DEFAULT_NUM_GRID = 32
DEFAULT_NUM_DIM = 3

#-------------------------------------------------

def seed(num=None, verbose=False):
    if num is not None:
        if verbose:
            print('setting numpy.random.seed=%d' % num)
        np.random.seed(num)

#-------------------------------------------------

def load(fields, path=None, num_grid=DEFAULT_NUM_GRID, num_dim=DEFAULT_NUM_DIM, max_edgelength=None, verbose=False, Verbose=False):
    """standardize logic for loading data and/or generating synthetic data
    """
    verbose |= Verbose
    data = dict()

    if path is not None: # read data from file
        if verbose:
            print('loading: '+path)

        if Fields is None:
            raise ImportError('could not import PLASMAtools.read_funcs.read.Fields')

        turb = Fields(path, reformat=True)

        # read the fields
        for field in fields:
            turb.read(field, verbose=Verbose)
            data[field] = getattr(turb, field) # replacement for this syntax: turb.vel

        del turb # get rid of this object to save memory

    else: # generate random data on a big-ish 3D array

        shape = (1,)+(num_grid,)*num_dim
        if verbose:
            print('generating randomized data with shape: %s' % (shape,))

        # use grid to compute coherent structure
        x = np.arange(num_grid) / num_grid
        if num_dim > 1:
            xs = np.meshgrid(*(x for x in range(num_dim)), indexing='ij')
            coherent = 0.5*np.exp(-0.5*np.sum((xs[:-1]-xs[-1])**2)/0.1**2) ### a tube

        else:
            coherent = 0.5*np.exp(-0.5*(x-0.5)**2/0.1**2) ### a bump

        # iterate through fields and add Gaussia noise
        for field in fields:
            data[field] = coherent + np.random.normal(size=shape)

    #---

    if verbose:
        for field in fields:
            print('    '+field, data[field].shape) # expect [num_dim, num_x, num_y, num_z]

    #---

    if max_edgelength is not None:
        if verbose:
            print('limiting data size by selecting the first max(edgelength)=%d samples' % max_edgelength)

        for key in data.keys():
            data[key] = data[key][:, :max_edgelength, :max_edgelength, :max_edgelength]

    #---

    return data

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
    raise NotImplementedError('compute covariance matrix!')

    # return
    return index, mom, cov
