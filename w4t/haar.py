"""a module for Haar decompositions
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np
import copy

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

class HaarArray(object):
    """an object that manages storage and Haar decompositions of ND arrays
    """

    def __init__(self, array):
        self._array = copy.deepcopy(array) # make a copy of the array
        self._shape = self.array.shape
        self._ndim = len(self.shape)
        self._levels = [0]*self.ndim

    #--------------------

    @property
    def array(self):
        return self._array

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

    #--------------------

    @property
    def levels(self):
        return self._levels

    @property
    def active(self):
        """return the indexes of the lowest level of the decomposition
        """
        return tuple(n//s for n, s in zip(self.shape, self.scales))

    #---

    @property
    def scales(self):
        return tuple(2**l for l in self.levels)

    @ property
    def wavenumbers(self):
        return tuple(1./scale for scale in self.scales)

    #--------------------

    @property
    def approx(self):
        """grab the current approximate coefficients
        """
        inds = self.active
        return self.array[tuple(slice(ind) for ind in inds)]

    @property
    def detail(self):
        """grab the current detail coefficients
        """
        inds = self.active
        return self.array[tuple(slice(ind, 2*ind) for ind in inds)]

    #--------------------

    def haar(self, axis=None):
        """apply the Haar decomposition to axis. If axis=None, apply it to all axes
        """
        if axis is not None:
            # determine appropriate indexes
            if axis < 0:
                axis = self.ndim + axis

            assert 2**self.levels[axis] < self.shape[axis], 'cannot decompose axis=%d further!' % axis

            ind = self.active[axis]
            inds = np.arange(ind)

            # perform decomposition
            a, d = haar(np.take(self.array, inds, axis=axis), axis=axis)

            # update in place
            num = ind//2
            shape = tuple(num if a==axis else 1 for a in range(self.ndim))

            np.put_along_axis(self.array, inds[:num].reshape(shape), a, axis=axis)
            np.put_along_axis(self.array, inds[num:].reshape(shape), d, axis=axis)

            # increment level
            self.levels[axis] += 1

        else:
            for axis in range(self.ndim):
                self.haar(axis=axis)

    def ihaar(self, axis=None):
        """apply the inverse Haar decomposition to axis. If axis=None, apply it to all axes
        """
        if axis is not None:
            # determine appropriate indexes
            if axis < 0:
                axis = self.ndim + axis

            assert self.levels[axis] > 0, 'cannot inverse-decompose axis=%d further!' % axis

            ind = self.active[axis]

            # perform inverse decomposition
            x = ihaar(
                np.take(self.array, np.arange(ind), axis=axis),
                np.take(self.array, np.arange(ind, 2*ind), axis=axis),
                axis=axis,
            )

            # update in place
            np.put_along_axis(
                self.array,
                np.arange(2*ind).reshape(tuple(2*ind if a==axis else 1 for a in range(self.ndim))),
                x,
                axis=axis,
            )

            # increment level
            self.levels[axis] -= 1

        else:
            for axis in range(self.ndim):
                self.ihaar(axis=axis)

    #-------

    def decompose(self, axis=None):
        """completely decompose a particular axis as far as it will go
        """
        if axis is not None:
            while self.active[axis] > 1: ### keep decomposing
                self.haar(axis=axis)

        else:
            for axis in range(self.ndim):
                self.decompose(axis=axis)

    def idecompose(self, axis=None):
        """completely undo decomposition of a particular axis as far as it will go
        """
        if axis is not None:
            while self.levels[axis] > 0: ### keep inverting
                self.ihaar(axis=axis)

        else:
            for axis in range(self.ndim):
                self.idecompose(axis=axis)

    #---

    def set_levels(self, levels):
        """haar and/or ihaar until we reach the target decomposition specified by levels
        """
        assert len(levels) == self.ndim, 'bad shape for levels'
        for axis in range(self.ndim):
            if self.levels[axis] > levels[axis]: # need to ihaar
                while self.levels[axis] > levels[axis]:
                    self.ihaar(axis=axis)

            elif self.levels[axis] < levels[axis]: # need to haar
                while self.levels[axis] < levels[axis]:
                    self.haar(axis=axis)

            else: # levels already match
                pass

    #--------------------

    def denoise(self, thr=None):
        """perform basic "wavelet denoising" by taking the full wavelet decomposition and zeroing all detail coefficients with \
with absolute values less than "thr". If thr=None, automatically selects an appropriate threshold.

        WARNING: this function modifies data in-place. Data will be lost for the coefficients that are set to zero.
        """
        levels = copy.copy(self.levels) # make a copy so we can remember the level of decomposition

        if thr is None:
            thr = self._automatic_denoise_thr() # note, this will change the decomposition level
        else:
            self.decompose() # completely decompose

        self.array[np.abs(self.array) < thr] = 0.0 # apply threshold

        # work back to the level of decomposition we were at initially
        self.set_levels(levels)

        # return the threshold used
        return thr

    #---

    def _automatic_denoise_thr(self, max_iter=1000):
        """compute the appropriate denoising threshold based on Farge & Schneider (2015) arXiv:1508.05650v1
        """
        # grab all the detail coefficients at each level of decomposition

        self.idecompose()
        details = []
        while self.active[0] > 1: # WARNING! implicitly assumes all axes have equal length
            self.haar() # decompose
            details.append(self.detail.flatten()) # grab the details

        details = np.concatenate(tuple(details))

        #---

        N = np.prod(self.shape) # the number of points in the grid
        logN = np.log(N)

        #---

        # initialize loop
        Nnoise = N # initial guess for how many coeffs are noise
        thr = self._var2thr(np.var(details), logN) # threshold for declaring a coeff noise
        sel = np.abs(details) < thr # the identified coeffs
        Nsel = np.sum(sel) # number of identified coeffs

        # iterate until we converge on a number of selected pixels
        i = 0
        while (Nnoise != Nsel):
            if i >= max_iter: # increment counter to break infinite loop
                raise RuntimeError('number of identified coefficients did not converge within %d iterations' % max_iter)
            i += 1

            # update estiamtes
            Nnoise = Nsel
            thr = self._var2thr(np.var(details[sel]), logN)
            sel = np.abs(details) < thr
            Nsel = np.sum(sel)

        #---

        # return thr
        return thr

    @staticmethod
    def _var2thr(var, logN):
        """optimal denoising threshold given a estimate of the wavelet variance (Eq 30 of arXiv:1508.05650v1)
        """
        return (2 * var * logN)**0.5

    #--------------------

    def coefficients(self, levels):
        """extract and return the coefficients corresponding to decomposition at levels
        """
        raise NotImplementedError

    def pixels(self, levels):
        """compute the pixel boundaries corresponding to the coefficients of the decomposition at levels
        """
        raise NotImplementedError
