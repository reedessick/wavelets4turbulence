"""a module for Haar decompositions
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

def haar(array, axis=None):
    """take the Haar decomposition of array along axis. If axis is None, use the last axis of array (axis=-1)
    approx = array[i+1] + array[i]
    detail = array[i+1] - array[i]
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
    return jnds+inds, jnds-inds

#---

def ihaar(approx, detail, axis=None):
    """invert the Haar decomposition along an axis. If axis is None, use the last axis of the array (axis=-1)
    array[i+1] = 0.5*(approx + detail)
    array[i]   = 0.5*(approx - detail)
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

    np.put_along_axis(array, inds, 0.5*(approx+detail), axis)
    np.put_along_axis(array, jnds, 0.5*(approx-detail), axis)

    # return
    return array

#------------------------

# FIXME
# write method to compute detail and approx coefficients in an ND cube
# it will probably be ok if I lose information during this (i.e., if the transform is not invertible)
# since we will primarily be interested in decomposing a field and examining it at different scales without necessarily
# reassembling it after the fact
