"""a module that houses basic utilities
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

def seed(num=None, verbose=False):
    if num is not None:
        if verbose:
            print('setting numpy.random.seed=%d' % num)
        np.random.seed(num)

#-------------------------------------------------

def default_map2scalar(array):
    """a default map from a vector to a scalar quantitity: sqrt of quadrature sum of components (euclidean vector magnitude)
    """
    return np.sum(array**2, axis=0)**0.5 # take the quadrature sum of vector components by default
