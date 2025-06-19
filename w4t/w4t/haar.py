"""a module for custom Haar decompositions
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from .w4t import WaveletArray

#-------------------------------------------------

__SCALE__ = 1./2**0.5 # used to normalize sums

def haar(array, axis=None):
    """take the Haar decomposition of array along axis. If axis is None, use the last axis of array (axis=-1)
    approx = SCALE * (array[i+1] + array[i])
    detail = SCALE * (array[i+1] - array[i])
    return approx, detail
    """
    if axis is None:
        axis = -1

    # figure out the indexes for Haar transform
    num = np.shape(array)[axis]
    assert num % 2 == 0, 'axes length (axis=%d --> len=%d) must be a multiple of 2' % (axis, num)

    inds = np.arange(num)[::2]
    jnds = inds + 1

    # compute Haar transform
    inds = np.take(array, inds, axis=axis)
    jnds = np.take(array, jnds, axis=axis)

    # return
    return __SCALE__*(jnds+inds), __SCALE__*(jnds-inds) ### FIXME: include a normalization to keep stdv of Gaussian coeffs constant?

#---

def ihaar(approx, detail, axis=None):
    """invert the Haar decomposition along an axis. If axis is None, use the last axis of the array (axis=-1)
    array[i]   = 0.5 * (approx - detail) / SCALE
    array[i+1] = 0.5 * (approx + detail) / SCALE
    return array
    """
    if axis is None:
        axis = -1

    # set up the output array
    shape = np.shape(approx)
    assert shape == np.shape(detail), 'mismatch in shape between approx and detail'

    shape = list(shape)
    num = shape[axis]
    shape[axis] *= 2

    array = np.empty(shape, dtype=float)

    # put the data into the array
    inds = np.arange(num)*2
    jnds = inds + 1

    shape = tuple(num if a==axis else 1 for a in range(len(shape)))
    np.put_along_axis(array, inds.reshape(shape), 0.5*(approx-detail)/__SCALE__, axis)
    np.put_along_axis(array, jnds.reshape(shape), 0.5*(approx+detail)/__SCALE__, axis)

    # return
    return array

#------------------------

class HaarArray(WaveletArray):
    """an object that manages storage and Haar decompositions of ND arrays
    """

    def _dwtn(self, array, axis):
        # perform decomposition
        return haar(array, axis=axis)

    def _idwtn(self, a, d, axis):
        # perform inverse decomposition
        return ihaar(a, d, axis=axis)
