"""a module for general wavelet decompositions that relies on PyWavelets (https://pywavelets.readthedocs.io/en/latest/index.html)
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np
import copy

### non-standard libraries
import pywt

from . import utils

#-------------------------------------------------

class WaveletArray(object):
    """an object that manages storage and wavelet decompositions of ND arrays
    """
    _mode = 'periodization' # this is important to the memory structure within this array and should not be changed!

    def __init__(self, array, wavelet):
        self._wavelet = wavelet

        self._array = copy.deepcopy(array) # make a copy of the array
        self._shape = self.array.shape
        self._ndim = len(self.shape)
        self._levels = [0]*self.ndim

    #--------------------

    @property
    def wavelet(self):
        return self._wavelet

    @property
    def mode(self):
        return self._mode

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

    def dwt(self, axis=None):
        """apply the Haar decomposition to axis. If axis=None, apply it to all axes
        """
        if axis is not None:
            assert 2**self.levels[axis] < self.shape[axis], 'cannot decompose axis=%d further!' % axis

            ind = self.active[axis]
            inds = np.arange(ind)

            # perform decomposition
            ans = pywt.dwtn(np.take(self.array, inds, axis=axis), self.wavelet, mode=self.mode, axes=[axis])
            a = ans['a']
            d = ans['d']

            # update in place
            num = ind//2
            shape = tuple(num if _==axis else 1 for _ in range(self.ndim))

            np.put_along_axis(self.array, inds[:num].reshape(shape), a, axis=axis)
            np.put_along_axis(self.array, inds[num:].reshape(shape), d, axis=axis)

            # increment level
            self.levels[axis] += 1

        else:
            for axis in range(self.ndim):
                self.dwt(axis=axis)

    def idwt(self, axis=None):
        """apply the inverse Haar decomposition to axis. If axis=None, apply it to all axes
        """
        if axis is not None:
            assert self.levels[axis] > 0, 'cannot inverse-decompose axis=%d further!' % axis

            ind = self.active[axis]

            # perform inverse decomposition
            x = pywt.idwtn(
                dict(a=np.take(self.array, np.arange(ind), axis=axis), d=np.take(self.array, np.arange(ind, 2*ind), axis=axis)),
                self.wavelet,
                mode=self.mode,
                axes=[axis],
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
                self.idwt(axis=axis)

    #-------

    def decompose(self, axis=None):
        """completely decompose a particular axis as far as it will go
        """
        if axis is not None:
            while self.active[axis] > 1: ### keep decomposing
                self.dwt(axis=axis)

        else:
            for axis in range(self.ndim):
                self.decompose(axis=axis)

    def idecompose(self, axis=None):
        """completely undo decomposition of a particular axis as far as it will go
        """
        if axis is not None:
            while self.levels[axis] > 0: ### keep inverting
                self.idwt(axis=axis)

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
                    self.idwt(axis=axis)

            elif self.levels[axis] < levels[axis]: # need to haar
                while self.levels[axis] < levels[axis]:
                    self.dwt(axis=axis)

            else: # levels already match
                pass

    #--------------------

    def denoise(self, num_std, smooth=False):
        """perform basic "wavelet denoising" by taking the full wavelet decomposition and zeroing all detail coefficients with \
with absolute values less than a threshold. This threshold is taken as num_std*std(detail) within each scale separately

        WARNING: this function modifies data in-place. Data will be lost for the coefficients that are set to zero.
        """
        levels = copy.copy(self.levels) # make a copy so we can remember the level of decomposition

        self.decompose() # completely decompose
        while self.scales[0] > 1: # work our way to smaller scales

            # set up all possible combos of scales relevant here
            slices = [tuple()]
            for dim, num in enumerate(self.active):
                slices = [_+(slice(0,num),) for _ in slices] + [_+(slice(num,2*num),) for _ in slices]
            slices = slices[1:] # skip the corner that is only approximants

            # iterate over slices, derive and apply thresholds for each set separately
            for s in slices:
                sel = np.abs(self.array[s]) <= np.std(self.array[s])*num_std # the small-amplitude detail coeffs
                if smooth: # zero the high-amplitude detail coefficients
                    sel = np.logical_not(sel)
                self.array[s] *= np.where(sel, 0.0, 1.0)

            # ihaar to go up to the next scale
            self.idwt()

        # work back to the level of decomposition we were at initially
        self.set_levels(levels)

    #--------------------

    def spectrum(self, index=[2], use_abs=False):
        """compute and return the moments of the detail distributions at each scale in the decomposition
    index should be an iterable corresponding to which moments you want to compute
        """
        index = sorted(index)

        self.idecompose() # start at the top

        scales = []
        moments = []
        covs = []
        while self.active[0] > 1:
            self.dwt() # decompose
            scales.append(self.scales)
            _, m, c = utils.moments(np.abs(self.detail.flatten()) if use_abs else self.detail.flatten(), index)
            moments.append(m)
            covs.append(c)

        return np.array(scales, dtype=float), np.array(moments, dtype=float), np.array(covs, dtype=float)

    #--------------------

#    def coefficients(self, levels):
#        """extract and return the coefficients corresponding to decomposition at levels
#        """
#        raise NotImplementedError

#    def pixels(self, levels):
#        """compute the pixel boundaries corresponding to the coefficients of the decomposition at levels
#        """
#        raise NotImplementedError
