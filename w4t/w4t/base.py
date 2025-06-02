"""a module for general wavelet decompositions
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import copy
from collections import defaultdict

import numpy as np

### non-standard libraries
from w4t.utils import moments
from w4t.utils import structures

#-------------------------------------------------

class WaveletArray(object):
    """an object that manages storage and wavelet decompositions of ND arrays
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
        """grab the current detail coefficients. Only grabs details in all dimensions
        """
        inds = self.active
        return self.array[tuple(slice(ind, 2*ind) for ind in inds)]

    def coeffs(self, approx_or_detail):
        """grab the current coefficients in various blocks of the array.
        "approx_or_detail" should be an iterable of length ndim that specifies which block is referenced. True corresponds to approx and False corresponds to detail
        """
        assert len(approx_or_detail) == self.ndim, 'bad length for approx_or_detail'
        inds = self.active
        return self.array[tuple(slice(ind) if aod else slice(ind,2*ind) for ind, aod in zip(inds, approx_or_detail))]

    #--------------------

    def _dwtn(self, axis):
        raise NotImplementedError('children need to overwrite this')

    def dwt(self, axis=None):
        """apply the Haar decomposition to axis. If axis=None, apply it to all axes
        """
        if axis is not None:
            assert 2**self.levels[axis] < self.shape[axis], 'cannot decompose axis=%d further!' % axis

            ind = self.active[axis]
            inds = np.arange(ind)

            # perform decomposition
            a, d = self._dwtn(np.take(self.array, inds, axis=axis), axis)

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

    #---

    def _idwtn(self, a, d, axis):
        raise NotImplementedError('children need to overwrite this')

    def idwt(self, axis=None):
        """apply the inverse Haar decomposition to axis. If axis=None, apply it to all axes
        """
        if axis is not None:
            assert self.levels[axis] > 0, 'cannot inverse-decompose axis=%d further!' % axis

            ind = self.active[axis]

            # perform inverse decomposition
            x = self._idwtn(
                np.take(self.array, np.arange(ind), axis=axis), # a
                np.take(self.array, np.arange(ind, 2*ind), axis=axis), # d
                axis,
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

    def isotropic_structure_function(self, index=[2], use_abs=False, verbose=False, Verbose=False):
        """compute the structure function for various scales by averaging over all dimensions at each scale
        """
        verbose |= Verbose

        mom_dict = defaultdict(list) # use these to store the result at each scale
        cov_dict = defaultdict(list)

        scales_set = set()

        # compute moments for 1D decompositions along each axis separately
        for dim in range(self.ndim):
            if verbose:
                print('estimating moments for wavelet decomposition along dim=%d' % dim)

            scales, mom, cov = self.structure_function(dim, index=index, use_abs=use_abs, verbose=Verbose)
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
            mom[snd,:] = np.mean(mom_dict[scale], axis=0) # average over dimensions
            cov[snd,:] = np.sum(cov_dict[scale], axis=0) / len(cov_dict[scale])**2 # update covariance of the average

        # return
        return scales, mom, cov

    #-------

    def structure_function(self, dim, index=[2], use_abs=False, verbose=False):
        """compute the structure function for various scales in a 1D decomposition along dim
        """
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

            samples = self.coeffs(approx_or_detail).flatten()
            if use_abs:
                samples = np.abs(samples)

            s = self.scales[dim] / 2

            _, m, c = moments.moments(
                samples * 2**(1 - 0.5*self.levels[dim]), # correct for wavelet normalization
                index,
                central=False, # do not use central moments for structure functions
            )

            scales.append(s)
            moms.append(m)
            covs.append(c)

        return np.array(scales, dtype=float), np.array(moms, dtype=float), np.array(covs, dtype=float)

    #-------

    def scaling_exponent(self, index=[2], use_abs=False, min_scale=None, max_scale=None, verbose=False, Verbose=False):
        """compute scaling exponent for structure functions
        """
        # compute moments
        scales, moms, covs = self.isotropic_structure_function(index=index, use_abs=use_abs, verbose=verbose, Verbose=Verbose)

        # downselect scales before performing basic fit to extract scaling exponent
        sel = np.ones(len(scale), dtype=bool)

        if min_scale is not None:
            sel *= (min_scale <= scales)

        if max_scale is not None:
            sel *= (scales <= max_scales)

        # perform fit and return
        return [moments.scaling_exponent(scales[sel], mom[sel,ind], 1./covs[sel,ind,ind]**0.5) for ind in range(len(index))]

    #--------------------

    def denoise(self, num_std, smooth=False, max_scale=None):
        """perform basic "wavelet denoising" by taking the full wavelet decomposition and zeroing all detail coefficients with \
with absolute values less than a threshold. This threshold is taken as num_std*std(detail) within each scale separately

        WARNING: this function modifies data in-place. Data will be lost for the coefficients that are set to zero.
        """
        if max_scale is None:
            max_scale = self.shape[0] # continue to denoise over all scales

        levels = copy.copy(self.levels) # make a copy so we can remember the level of decomposition

        self.idecompose() # start from the top
        max_scale = min(max_scale, min(*self.active)) # limit this to the size of the array

        while self.scales[0] < max_scale: # continue to denoise
            self.dwt() # decompose again

            slices = [tuple()]
            for dim, num in enumerate(self.active):
                slices = [_+(slice(0,num),) for _ in slices] + [_+(slice(num,2*num),) for _ in slices]

            for s in slices[1:]: # only touch the detail coefficients
                sel = np.abs(self.array[s]) <= np.std(self.array[s])*num_std # the small-amplitude detail coeffs
                if smooth: # zero the high-amplitude detail coefficients
                    sel = np.logical_not(sel)
                self.array[s] *= np.where(sel, 0.0, 1.0)

        # zero the remaining approx coeffs
        if not smooth:
            self.array[tuple(slice(0,num) for num in self.active)] *= 0

        # put the levels back
        self.set_levels(levels)

    #--------------------

    def structures(self, thr=0, num_proc=1, timeit=False):
        """returns a list of sets of pixels corresponding to spatially separate structures at the current scale
        thr sets the threshold for considering a pixel to be "on" and therefore eligible to be included in a structure
        """
        return structures.find_structures(np.abs(self.approx)>=thr, num_proc=num_proc, timeit=timeit)
