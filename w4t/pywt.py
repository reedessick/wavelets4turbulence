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

    def structures(self, thr=0, num_proc=1, timeit=False):
        """returns a list of sets of pixels corresponding to spatially separate structures at the current scale
        thr sets the threshold for considering a pixel to be "on" and therefore eligible to be included in a structure
        """
        sel = np.abs(self.approx) >= thr
        if num_proc == 1:
            strucs = self._structures(sel)

        else:
            # figure out how to slice the array
            # FIXME! we may be able to slice this up into many smaller pieces by using more than 1 axis?
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
                strucs = pool.map(self._structures, [sel[s:e] for s, e in bounds]) # find structures in separate regions

            ### old merging --> FIXME: remove this once new merging works 
            old_strucs = self._merge_structures(copy.deepcopy(strucs), bounds, self.ndim, timeit=timeit) # merge these

            ### new merging --> FIXME: you should be able to parallelize this
            if timeit:
                import time
                t0 = time.time()

            while len(strucs) > 1:
                struc, bound = self._merge2structures(strucs[0], bounds[0], strucs[1], bounds[1], self.ndim)
                strucs = [struc] + strucs[2:]
                bounds = [bound] + bounds[2:]
            strucs = strucs[0]

            if timeit:
                print('>>> new merging:', time.time()-t0)

        # return
        return strucs

    @staticmethod
    def _merge2structures(struc1, bounds1, struc2, bounds2, ndim):
        """merge 2 sets of structures into a single set
        """
        s1, e1 = bounds1
        s2, e2 = bounds2
        if e1!=s2:
            if e2==s1:
                return WaveletArray._merge2structures(struc2, bounds2, struc1, bounds1, ndim)
            else:
                raise ValueError('regions do not seem to be contiguous')

        # sort the structures into those that might be connected and those that cannot be connected
        unconnected = []

        interesting1 = []
        for struc in struc1:
            struc[:,0] += s1
            if np.max(struc[:,0]) == e1-1:
                interesting1.append(struc)
            else:
                unconnected.append(struc)

        interesting2 = []
        for struc in struc2:
            struc[:,0] += s2
            if np.min(struc[:,0]) == s2:
                interesting2.append(struc)
            else:
                unconnected.append(struc)

        # now compare the structures that are in "interesting"
        ### FIXME switch to first check bounding boxes?

        unconnected2 = []
        for cluster2 in interesting2:
            matches2 = cluster2[:,0] == s2
            connected = set()

            for pix in cluster2[matches2]:
                for mnd, cluster1 in enumerate(interesting1):
                    matches1 = cluster1[:,0] == e1-1
                    if np.any(np.sum((pix-cluster1[matches1])**2, axis=1) <= ndim):
                        connected.add(mnd)

            if connected: # any are connected --> update interesting1
                cluster2 = np.concatenate(tuple([interesting1[mnd] for mnd in connected])+(cluster2,))
                interesting1 = [interesting1[mnd] for mnd in range(len(interesting1)) if (mnd not in connected)] + [cluster2]
            else:
                unconnected2.append(cluster2)

        # now concatenate all clusters
        return unconnected + interesting1 + unconnected2, (s1, e2)













    @staticmethod
    def _merge_structures(strucs, bounds, ndim, timeit=False):
        """merge sets of structures into a single set
        """

        if timeit:### FIXME! this is the bottleneck in parallelization!
            import time
            t0 = time.time()

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
                            if np.any(old_matches) and np.any(np.sum((pix-existing[old_matches])**2, axis=1) <= ndim):
                                connected.add(mnd)

                    if connected: # we need to merge something
                        cluster = np.concatenate(tuple([merged[mnd] for mnd in connected])+(cluster,))

                    merged = [merged[mnd] for mnd in range(len(merged)) if (mnd not in connected)] + [cluster]

                else: # no chance this connects to an existing cluster, so just add it
                    merged.append(cluster)

        if timeit:
            print('>>> merging:', time.time()-t0)

        return merged

    @staticmethod
    def _structures(sel):
        sel = copy.copy(sel) # make a copy so we can update it in-place
        clusters = []

        # define the shift template
        shifts = WaveletArray._shape2shifts(sel.shape)

        while np.any(sel): # iterate until there are no more selected pixels, this loop in necessary
            pix = WaveletArray._sel2pix(sel) # grab a new pixel
            cluster, sel = WaveletArray._pix2cluster(pix, sel, shifts) # find everything that belongs in that pixel's cluster
            clusters.append(cluster) # record the cluster

        # return : list of clusters, each of which is a list of pixels
        return clusters

    @staticmethod
    def _shape2shifts(shape):
        shifts = [()]
        for dim in range(len(shape)): # this loop is necessary, but will only be done once
            shifts = [shift+(0,) for shift in shifts] + [shift+(-1,) for shift in shifts] + [shift+(+1,) for shift in shifts]
        return np.array([np.array(shift) for shift in shifts if np.any(shift)], dtype=int)

    @staticmethod
    def _sel2pix(sel):
        """pick a pixel from this boolean array
        """
#        assert np.any(sel), 'cannot select a pixel from a boolean array with all entries == False'
        return np.transpose(np.nonzero(sel))[0]

    @staticmethod
    def _pix2cluster(pix, sel, shifts):
        """starting at location "pix", identify all contiguous pixels with sel[pix]=True.
        Updates sel in place.
        """
        cluster = [np.array([pix])]
        tocheck = [pix]
        sel[tuple(pix)] = False # mark this as checked

        # iterate through 
        while len(tocheck): # this loop is probably necessary
            pix = tocheck.pop(0) # grab the next pixel

            neighbors = WaveletArray._pix2neighbors(pix, shifts, sel.shape)
            successes = neighbors[sel[tuple(np.transpose(neighbors))]]

            cluster.append(successes)
            tocheck += list(successes)
            sel[tuple(np.transpose(successes))] = False

        # return
        return np.concatenate(tuple(cluster)), sel

    @staticmethod
    def _pix2neighbors(pix, shifts, shape):
        """return a list of possible neighbors for this pix
        """
        neighbors = pix + shifts

        # exclude neighbors that fall outside grid boundaries
        return neighbors[np.all(neighbors >= 0, axis=1)*np.all(neighbors < shape, axis=1)]

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
