"""a module for general wavelet decompositions
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from abc import (ABC, abstractmethod)

import copy
from collections import defaultdict

import numpy as np

### non-standard libraries
from w4t.utils import moments
from w4t.utils import structures
from w4t.utils.utils import default_map2scalar

from w4t.plot import flow

#-------------------------------------------------

DEFAULT_DENOISE_THRESHOLD = 1.0
DEFAULT_STRUCTURE_THRESHOLD = 1.0

#-------------------------------------------------

class WaveletArray(ABC):
    """an object that manages storage and wavelet decompositions of ND arrays for vector fields.
expect arrays to have shape (nvec, numx, numy, ...) so that the first index is interpreted as the vector componets and the \
rest are interpreted as spatial coordinates.
    """

    def __init__(self, array):
        self._array = copy.deepcopy(array) # make a copy of the array
        self._shape = self.array.shape
        self._nvec = self.shape[0]
        self._ndim = len(self.shape) - 1
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

    @property
    def nvec(self):
        return self._nvec

    #--------------------

    @property
    def levels(self):
        return self._levels

    @property
    def active(self):
        """return the indexes of the lowest level of the decomposition
        """
        return tuple(n//s for n, s in zip(self.shape[1:], self.scales))

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
        return self.array[(slice(self.nvec),) + tuple(slice(ind) for ind in inds)]

    @property
    def detail(self):
        """grab the current detail coefficients. Only grabs details in all dimensions
        """
        inds = self.active
        return self.array[(slice(self.nvec),) + tuple(slice(ind, 2*ind) for ind in inds)]

    def coeffs(self, approx_or_detail):
        """grab the current coefficients in various blocks of the array.
        "approx_or_detail" should be an iterable of length ndim that specifies which block is referenced. True corresponds to approx and False corresponds to detail
        """
        assert len(approx_or_detail) == self.ndim, 'bad length for approx_or_detail'
        inds = self.active
        return self.array[(slice(self.nvec),) + \
            tuple(slice(ind) if aod else slice(ind,2*ind) for ind, aod in zip(inds, approx_or_detail))]

    @property
    def coeffset(self):
        """grab the set of most-relevant approximant and detail coefficients given the current level of decomposition
        """
        # FIXME? can we this in a way that extends to an arbitrary number of dimensions?
        if self.ndim == 1:
            return self.approx, self.detail

        elif self.ndim == 2:
            return (
                self.approx,
                self.coeffs([True, False]), # ad
                self.coeffs([False, True]), # da
                self.detail,
            )

        elif self.ndim == 3:
            return (
                self.approx,
                self.coeffs([True, True, False]), # aad
                self.coeffs([True, False, True]), # ada
                self.coeffs([False, True, True]), # daa
                self.coeffs([True, False, False]), # add
                self.coeffs([False, True, False]), # dad
                self.coeffs([False, False, True]), # dda
                self.detail,
            )

        else:
            raise RuntimeError('do not know how to retrieve coeffset for ndim=%d' % self.ndim)

    #--------------------

    @abstractmethod
    def _dwtn(self, axis):
        pass

    def dwt(self, axis=None):
        """apply the Haar decomposition to axis. If axis=None, apply it to all axes
        """
        if axis is not None:
            assert 2**self.levels[axis] < self.shape[axis+1], 'cannot decompose axis=%d further!' % axis

            ind = self.active[axis]
            inds = np.arange(ind)

            # perform decomposition
            a, d = self._dwtn(np.take(self.array, inds, axis=axis+1), axis+1)

            # update in place
            num = ind//2
            shape = (1,)+tuple(num if _==axis else 1 for _ in range(self.ndim))

            np.put_along_axis(self.array, inds[:num].reshape(shape), a, axis=axis+1)
            np.put_along_axis(self.array, inds[num:].reshape(shape), d, axis=axis+1)

            # increment level
            self.levels[axis] += 1

        else:
            for axis in range(self.ndim):
                self.dwt(axis=axis)

    #---

    @abstractmethod
    def _idwtn(self, a, d, axis):
        pass

    def idwt(self, axis=None):
        """apply the inverse Haar decomposition to axis. If axis=None, apply it to all axes
        """
        if axis is not None:
            assert self.levels[axis] > 0, 'cannot inverse-decompose axis=%d further!' % axis

            ind = self.active[axis]

            # perform inverse decomposition
            x = self._idwtn(
                np.take(self.array, np.arange(ind), axis=axis+1), # a
                np.take(self.array, np.arange(ind, 2*ind), axis=axis+1), # d
                axis+1,
            )

            # update in place
            np.put_along_axis(
                self.array,
                np.arange(2*ind).reshape((1,)+tuple(2*ind if a==axis else 1 for a in range(self.ndim))),
                x,
                axis=axis+1,
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
        """dwt and/or idwt until we reach the target decomposition specified by levels
        """
        assert len(levels) == self.ndim, 'bad shape for levels'
        for axis in range(self.ndim):
            if self.levels[axis] > levels[axis]: # need to idwt
                while self.levels[axis] > levels[axis]:
                    self.idwt(axis=axis)

            elif self.levels[axis] < levels[axis]: # need to dwt
                while self.levels[axis] < levels[axis]:
                    self.dwt(axis=axis)

            else: # levels already match
                pass

    #--------------------

    def isotropic_structure_function(self, map2scalar=default_map2scalar, index=[2], use_abs=True, verbose=False, Verbose=False):
        """compute the structure function for various scales by averaging over all dimensions at each scale
        """
        verbose |= Verbose

        if verbose:
            print('computing isotropic structure function via wavelet decomposition')

        mom_dict = defaultdict(list) # use these to store the result at each scale
        cov_dict = defaultdict(list)

        scales_set = set()

        # compute moments for 1D decompositions along each axis separately
        for dim in range(self.ndim):

            scales, mom, cov = self.structure_function(dim, map2scalar=map2scalar, index=index, use_abs=use_abs, verbose=Verbose)
            for snd, scale in enumerate(scales):
                scales_set.add(scale)
                mom_dict[scale].append(mom[snd])
                cov_dict[scale].append(cov[snd])

        # now average over directions for each scale
        if verbose:
            print('averaging moments over dimensions at each scale')

        scales = np.array(sorted(scales_set), dtype=float)

        num_scales = len(scales)
        num_index = len(index)

        mom = np.empty((num_scales, num_index), dtype=float)
        cov = np.empty((num_scales, num_index, num_index), dtype=float)

        for snd, scale in enumerate(scales):
            mom[snd,:], cov[snd,:] = moments.average_moments(mom_dict[scale], cov_dict[scale])

        # return
        return scales, mom, cov

    #---

    def structure_function(self, dim, map2scalar=default_map2scalar, index=[2], use_abs=True, verbose=False):
        """compute the structure function for various scales in a 1D decomposition along dim
        """
        if verbose:
            print('estimating moments for wavelet decomposition along dim=%d' % dim)

        assert (0 <= dim) and (dim < self.ndim), 'bad dimension (dim=%d) for ndim=%d' % (dim, self.ndim)

        scales = []
        moms = []
        covs = []

        self.idecompose() # start at the top

        approx_or_detail = np.arange(self.ndim) != dim # grab the detail coeffs for just this dimension

        while self.active[dim] > 1: # keep going
            self.dwt(axis=dim)

            if verbose:
                print('computing moments for scales: ' + str(self.scales))

            samples = map2scalar(self.coeffs(approx_or_detail)).flatten()
            if use_abs:
                samples = np.abs(samples)

            s = self.scales[dim] / 2

            _, m, c = moments.moments(
                samples * 2**(1 - 0.5*self.levels[dim]), # correct for wavelet normalization
                index,
                central=False, # do not use central moments for structure functions
                verbose=verbose,
            )

            scales.append(s)
            moms.append(m)
            covs.append(c)

        return np.array(scales, dtype=float), np.array(moms, dtype=float), np.array(covs, dtype=float)

    #--------------------

    def denoise(self, num_std, map2scalar=default_map2scalar, smooth=False, max_scale=None):
        """perform basic "wavelet denoising" by taking the full wavelet decomposition and zeroing all detail coefficients with \
with absolute values less than a threshold. This threshold is taken as num_std*std(detail) within each scale separately

        WARNING: this function modifies data in-place. Data will be lost for the coefficients that are set to zero.
        """
        if max_scale is None:
            max_scale = self.shape[0] # continue to denoise over all scales

        levels = copy.copy(self.levels) # make a copy so we can remember the level of decomposition

        self.idecompose() # start from the top
        max_scale = min(max_scale, min(*self.active)) # limit this to the size of the array

        ones = np.ones(self.nvec, dtype=bool)

        while self.scales[0] < max_scale: # continue to denoise
            self.dwt() # decompose again

            slices = [(slice(self.nvec),)]
            for dim, num in enumerate(self.active):
                slices = [_+(slice(0,num),) for _ in slices] + [_+(slice(num,2*num),) for _ in slices]

            for s in slices[1:]: # only touch the detail coefficients
                scalar = map2scalar(self.array[s])
                sel = np.abs(scalar) <= np.std(scalar)*num_std # the small-amplitude detail coeffs
                if smooth: # zero the high-amplitude detail coefficients
                    sel = np.logical_not(sel)
                sel = np.outer(ones, sel).reshape((self.nvec,)+sel.shape)
                self.array[s] *= np.where(sel, 0.0, 1.0)

        # zero the remaining approx coeffs
        if not smooth:
            self.array[(slice(self.nvec),)+tuple(slice(0,num) for num in self.active)] *= 0

        # put the levels back
        self.set_levels(levels)

    #--------------------

    def structures(self, map2scalar=default_map2scalar, thr=0, num_proc=1, timeit=False):
        """returns a list of sets of pixels corresponding to spatially separate structures at the current scale
        thr sets the threshold for considering a pixel to be "on" and therefore eligible to be included in a structure
        """
        pixels = structures.find_structures(
            np.abs(map2scalar(self.approx)) >= thr*np.std(map2scalar(self.approx).flatten()),
            num_proc=num_proc,
            timeit=timeit,
        )
        return [Structure(pix, self.levels, self.shape[1:]) for pix in pixels]

    #--------------------

    def plot(self, map2scalar=default_map2scalar, **kwargs):
        """make a plot of approx coefficients
        """
        return flow.plot(map2scalar(self.approx), **kwargs)

    #---

    def hist(self, map2scalar=default_map2scalar, **kwargs):
        """make a histogram of approx coefficients
        """
        return flow.hist(map2scalar(self.approx), **kwargs)

    #-------

    def plot_coeff(self, map2scalar=default_map2scalar, **kwargs):
        """make plots of wavelet coefficients
        """
        return flow.plot_coeff(self.ndim, *(map2scalar(cs) for cs in self.coeffset), **kwargs)

    #---

    def hist_coeff(self, map2scalar=default_map2scalar, **kwargs):
        """make histograms of wavelet coefficients
        """
        return flow.hist_coeff(self.ndim, *(map2scalar(cs) for cs in self.coeffset), **kwargs)

    #-------

    def scalogram(self, map2scalar=default_map2scalar, **kwargs):
        """make a scalogram of the data
        """
        return flow.dim1.scalogram(self, map2scalar, **kwargs)

#-------------------------------------------------

class Structure(object):
    """a class representing an identified structure within a flow
    """

    def __init__(self, pixels, levels, shape):
        self._pixels = pixels
        self._levels = levels
        self._shape = shape

    #---

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def pixels(self):
        return self._pixels

    @property
    def levels(self):
        return self._levels

    @property
    def shape(self):
        return self._shape

    #---

    def __len__(self):
        return len(self.pixels)

    @property
    def bounding_box(self):
        return np.array([(np.min(self.pixels[:,dim]), np.max(self.pixels[:,dim])) for dim in range(self.ndim)], dtype=int)

    @property
    def tuple(self):
        return tuple(np.transpose(self.pixels)) # useful when accessing arrays

    #---

    def extract_as_array(self, waveletarray):
        """extract an array with zeros everywhere except for the active pixels
        """
        # sanity-check input
        waveletarray.set_levels(self.levels) # set to the appropriate level of decomposition
        assert np.all(waveletarray.approx.shape[1:] == self.shape), 'shape mismatch'

        # extract data
        array = np.zeros_like(waveletarray.approx, dtype=float) # default is zero everywhere

        tup = (slice(waveletarray.nvec),) + self.tuple
        array[tup] = waveletarray.approx[tup] # fill in the selected pixels

        # return
        return array

    def extract(self, waveletarray):
        # sanity-check input
        assert waveletarray.shape[1:] == self.shape, 'shape mismatch'
        waveletarray.set_levels(self.levels) # set to the appropriate level of decomposition
        return waveletarray.approx[(slice(waveletarray.nvec),) + self.tuple]

    #-------

    def principle_components(self, waveletarray, map2scalar=default_map2scalar, index=1):
        """compute the principle components of a structure with respect to the field contained in waveletarray raised to index
        """
        # figure out the measure with repect to which we compute the principle components
        weights = np.abs(map2scalar(self.extract(waveletarray)))**index
        weights /= np.sum(weights)

        # compute principle components and return
        return structures.principle_components(self.pixels, weights=weights)

    #--------------------

    def plot(self, waveletarray, map2scalar=default_map2scalar, zoom=False, **kwargs):
        """make a plot of approx coefficients
        """
        array = self.extract_as_array(waveletarray)
        return flow.plot(map2scalar(array), **kwargs)

    #-------

    def hist(self, waveletarray, map2scalar=default_map2scalar, **kwargs):
        """make a histogram of approx coefficients
        """
        return flow.hist(map2scalar(self.extract(waveletarray)), **kwargs)
