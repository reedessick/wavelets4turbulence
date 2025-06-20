"""a module that houses logic about plotting flows
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from . import (dim1, dim2, dim3)

#-------------------------------------------------

def plot(array, **kwargs):
    """a simple routing function that controls workflow based on data dimensionality
    """
    ndim = len(array.shape)
    if ndim == 1:
        extent = kwargs.pop('extent', [(0,1)])[0]
        return dim1.plot(array, extent=extent, **kwargs)

    elif ndim == 2:
        extent = kwargs.pop('extent', [(0,1)]*2)
        extent = (extent[0][0], extent[0][1], extent[1][0], extent[1][1])
        return dim2.plot(array, extent=extent, **kwargs)

    elif ndim == 3:
        return dim3.plot(array, **kwargs)

    else:
        raise RuntimeError('do not know how to plot approx for ndim=%d' % ndim)

#-----------

def plot_coeff(waveletarray, **kwargs):
    """a simple routing function that controls workflow based on data dimensionality
    """
    if waveletarray.ndim == 1:
        extent = kwargs.pop('extent', [(0, 1)])[0]
        return dim1.plot_coeff(*waveletarray.coeffset, extent=extent, **kwargs)

    elif waveletarray.ndim == 2:
        extent = kwargs.pop('extent', [(0,1)]*2)
        extent = (extent[0][0], extent[0][1], extent[1][0], extent[1][1])
        return dim2.plot_coeff(*waveletarray.coeffset, extent=extent, **kwargs)

    elif waveletarray.ndim == 3:
        return dim3.plot_coeff(*waveletarray.coeffset, **kwargs)

    else:
        raise RuntimeError('do not know how to plot wavelet coefficients for ndim=%d' % waveletarray.ndim)

#------------------------

def hist(array, **kwargs):
    """a simple routing function that controls workflow based on data dimensionality
    """
    ndim = len(array.shape)
    if ndim == 1:
        return dim1.hist(array, **kwargs)

    elif ndim == 2:
        return dim2.hist(array, **kwargs)

    elif ndim == 3:
        return dim3.hist(array, **kwargs)

    else:
        raise RuntimeError('do not know how histogram approx for ndim=%d' % ndim)

#-----------

def hist_coeff(waveletarray, **kwargs):
    """a simple routing function that controls workflow based on data dimensionality
    """
    if waveletarray.ndim == 1:
        foo = dim1.hist_coeff

    elif waveletarray.ndim == 2:
        foo = dim2.hist_coeff

    elif waveletarray.ndim == 3:
        foo = dim3.hist_coeff

    else:
        raise RuntimeError('do not know how to plot wavelet coefficients for ndim=%d' % waveletarray.ndim)

    return foo(*waveletarray.coeffset, **kwargs)

#------------------------

def grand_tour(array, **kwargs):
    """a simple routing function that controls workflow based on data dimensionality
    """
    ndim = len(array.shape)
    if ndim == 2:
        return dim2.grand_tour(array, **kwargs)

    elif ndim == 3:
        return dim3.grand_tour(array, **kwargs)

    else:
        raise RuntimeError('do not know how to grand_tour for ndim=%d' % ndim)
