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
    """a default map from a vector to a scalar quantitity:
    if there is only as single vector component, return that component
    otherwise, return the (euclidean) vector magnitude
    """
    if len(array) == 1: # there is only a single component, so just return that
        return array[0]

    else: # otherwise, return euclidean norm of vector
        return np.sum(array**2, axis=0)**0.5
