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
