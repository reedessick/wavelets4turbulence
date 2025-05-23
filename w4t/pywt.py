"""a module for general wavelet decompositions that relies on PyWavelets (https://pywavelets.readthedocs.io/en/latest/index.html)
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import copy
import numpy as np

import multiprocessing as mp

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

    def structures(self, thr=0, num_proc=1):
        """returns a list of sets of pixels corresponding to spatially separate structures at the current scale
        thr sets the threshold for considering a pixel to be "on" and therefore eligible to be included in a structure
        """
        sel = np.abs(self.approx) >= thr
        if num_proc == 1:
            return self._structures(sel)

        else:
            # figure out how to slice the array
            num = len(sel)
            n = num // num_proc # the min number per job
            m = num % num_proc # the number of jobs that will have an extra
            bounds = []
            s = 0
            for job in range(num_proc):
                e = s+n + (job<m)
                bounds.append((s,e))
                s = e

            # parallelize and then merge
            with mp.Pool(processes=num_proc) as pool:
                return self._merge_structures(pool.map(self._structures, [sel[s:e] for s, e in bounds]), bounds)

    @staticmethod
    def _merge_structures(strucs, bounds):
        """the additional overhead may be significant
        """
        merged = strucs[0]
        for new, (s, e) in zip(strucs[1:], bounds[1:]): # iterate through remaining slices

            for cluster in new: # for identified clusters in the new slice
                cluster[:,0] += s # bump these to start in the correct place in the overall array
                matches = cluster[:,0] == s

                if np.any(matches): # matches preceeding boundary, so check whether it connects to an existing cluster

                    connected = set() # these are the sets of pixels in merged that are connected to cluster

                    for pix in cluster[matches]:

                        for mnd, existing in enumerate(merged):
                            old_matches = existing[:,0] == s-1
                            # only check those that are on the border and are close enough that they would match
                            if np.any(old_matches) and np.any(np.sum((pix-existing[old_matches])**2, axis=1) <= 2):
                                connected.add(mnd)

                    if connected: # we need to merge something
                        cluster = np.concatenate(tuple([merged[mnd] for mnd in connected])+(cluster,))

                    merged = [merged[mnd] for mnd in range(len(merged)) if (mnd not in connected)] + [cluster]

                else: # no chance this connects to an existing cluster, so just add it
                    merged.append(cluster)

        return merged

    @staticmethod
    def _structures(sel):
        sel = copy.copy(sel) # make a copy so we can update it in-place
        clusters = []
        while np.any(sel): # iterate until there are no more selected pixels
            pix = WaveletArray._sel2pix(sel) # grab a new pixel
            cluster, sel = WaveletArray._pix2cluster(pix, sel) # find everything that belongs in that pixel's cluster
            clusters.append(cluster) # record the cluster

        # return : list of clusters, each of which is a list of pixels
        return clusters

    @staticmethod
    def _sel2pix(sel):
        """pick a pixel from this boolean array
        """
        assert np.any(sel), 'cannot select a pixel from a boolean array with all entries == False'
        return tuple(np.transpose(np.nonzero(sel))[0])

    @staticmethod
    def _pix2cluster(pix, sel):
        """starting at location "pix", identify all contiguous pixels with sel[pix]=True.
        Updates sel in place.
        """
        cluster = [pix]
        tocheck = [pix]
        sel[pix] = False # mark this as checked

        shape = sel.shape

        while len(tocheck):
            pix = tocheck.pop(0) # grab the next pixel

            for neighbor in WaveletArray._pix2neighbors(pix, shape): # iterate over neighbors
                if sel[neighbor]:
                    tocheck.append(neighbor)
                    cluster.append(neighbor)
                    sel[neighbor] = False # mark this as checked

        # return
        return np.array(cluster), sel

    @staticmethod
    def _pix2neighbors(pix, shape):
        """return a list of possible neighbors for this pix
        """
        # check consistency of data
        ndim = len(shape)
        assert len(pix) == ndim

        # get vectors so we can make changes
        pix = np.array(pix, dtype=int)

        # iterate through dimensions to figure out all the possible shifts
        shifts = [()]
        for dim, (ind, num) in enumerate(zip(pix, shape)):
            new = [shift+(0,) for shift in shifts]
            if ind > 0: # there is a pixel at ind - 1
                new += [shift+(-1,) for shift in shifts]
            if ind < num-1: # there is a pixel at ind + 1
                new += [shift+(+1,) for shift in shifts]
            shifts = new
        shifts = [np.array(shift) for shift in shifts if np.any(shift)]

        # define new pixels
        return [tuple(pix+shift) for shift in shifts]

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
