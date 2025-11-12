"""a module that holds routines used to identify structures from boolean arrays
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import time
import copy
import numpy as np
import multiprocessing as mp

#-------------------------------------------------

def principle_components(pixels, weights=None):
    """compute the principle components of a set of pixels with respect to the measure defined by weights
    return mean, eigvec, eigval (eigvec, eigval are in the format returned by np.linalg.eig)
    """
    # set up weights
    if weights is None:
        weights = np.ones(len(pixels), dtype=float)
    weights = weights/np.sum(weights) # make sure this is normalized

    # compute 1st moment
    mean = np.sum(weights*pixels, axis=0) # compute the mean position

    # compute 2nd moment
    ndim = pixels.shape[1] # the number of dimensions
    cov = np.empty((ndim, ndim), dtype=float)
    for row in range(ndim):
        cov[row,row] = np.sum(weights*(pixels[:,row]-mean[row])**2)

        for col in range(row):
            cov[row,col] = cov[col,row] = np.sum(weights*(pixels[:,row]-mean[row])*(pixels[:,col]-mean[col]))

    # find eigenvectors of cov to get principle directions
    eigval, eigvec = np.linalg.eig(cov)

    # return
    return mean, eigvev, eigval

#------------------------

def find_structures(sel, num_proc=1, timeit=False):
    """returns a list of sets of pixels corresponding to spatially separate structures at the current scale
    thr sets the threshold for considering a pixel to be "on" and therefore eligible to be included in a structure
    """
    ndim = len(sel.shape)

    if num_proc == 1:
        strucs = _structures(sel)

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

        # identify structures and then merge in parallel
        with mp.Pool(processes=num_proc) as pool:

            # identify structures
            if timeit:
                t0 = time.time()

            strucs = pool.starmap(_structures, [(sel[s:e], s) for s, e in bounds]) # find structures in separate regions

            if timeit:
                print('identify structures: %f sec' % (time.time()-t0))

            # merge
            if timeit:
                t0 = time.time()

            while len(strucs) > 2:
                num_pairs = len(strucs) // 2
                ans = pool.starmap(
                    _merge2structures,
                    [(strucs[2*n], bounds[2*n], strucs[2*n+1], bounds[2*n+1], ndim) for n in range(num_pairs)],
                )

                strucs = [a for a,b in ans] + strucs[2*num_pairs:]
                bounds = [b for a,b in ans] + bounds[2*num_pairs:]

            if timeit:
                print('preliminary merge structures: %f sec' % (time.time()-t0))

        if timeit:
            t0 = time.time()

        # merge the final 2 structures with a single process
        strucs, _ = _merge2structures(strucs[0], bounds[0], strucs[1], bounds[1], ndim)

        if timeit:
            print('final merge structures: %f sec' % (time.time()-t0))

    # return
    return strucs


#------------------------

def _merge2structures(struc1, bounds1, struc2, bounds2, ndim):
    """merge 2 sets of structures into a single set
    """
    s1, e1 = bounds1
    s2, e2 = bounds2
    if e1!=s2:
        if e2==s1:
            return _merge2structures(struc2, bounds2, struc1, bounds1, ndim)
        else:
            raise ValueError('regions do not seem to be contiguous')

    # sort the structures into those that might be connected and those that cannot be connected
    unconnected = []

    interesting1 = []
    for struc in struc1:
        if np.max(struc[:,0]) == e1-1:
            interesting1.append(struc)
        else:
            unconnected.append(struc)

    interesting2 = []
    for struc in struc2:
        if np.min(struc[:,0]) == s2:
            interesting2.append(struc)
        else:
            unconnected.append(struc)

    # now compare the structures that are in "interesting"

    box1 = np.empty((ndim-1, 2), dtype=int) # use to check bounding boxes first
    box2 = np.empty((ndim-1, 2), dtype=int)

    unconnected2 = []

    for cluster2 in interesting2:
        connected = set()
        c2_m2 = cluster2[cluster2[:,0] == s2] # identify which pixels touch the boundary
        box2[:] = np.array([(np.min(c2_m2[:,dim]), np.max(c2_m2[:,dim])) for dim in range(1,ndim)]) # bounding box

        for mnd, cluster1 in enumerate(interesting1): # loop over this first so we can break once we find a connection
            c1_m1 = cluster1[cluster1[:,0] == e1-1]
            box1[:] = np.array([(np.min(c1_m1[:,dim]), np.max(c1_m1[:,dim])) for dim in range(1,ndim)])

            if np.all((box1[:,0] <= box2[:,1]+1)*(box2[:,0] <= box1[:,1]+1)): # bounding boxes overlap within 1 pixel

                start = np.max([box1[:,0], box2[:,0]], axis=0) - 1 # the starting point of overlapping region
                stop = np.min([box1[:,1], box2[:,1]], axis=0) + 1 # the stopping point of overlapping region

                # grab just the pixels in the overlapping region
                c1_m1_o = c1_m1[np.all(start <= c1_m1[:,1:], axis=1)*np.all(c1_m1[:,1:] <= stop, axis=1)]

                if len(c1_m1_o): # there are some pixels in this overlapping region

                    # iterate over only the pixels in the overlapping region
                    for pix in c2_m2[np.all(start <= c2_m2[:,1:], axis=1)*np.all(c2_m2[:,1:] <= stop, axis=1)]:
                        if np.any(np.sum((pix-c1_m1_o)**2, axis=1) <= ndim):
                            connected.add(mnd)
                            break

                else: # there are no pixels in the overlapping region
                    pass

            else: # bounding boxes do not overlap, so no pixels can overlap
                pass

        if connected: # any are connected --> update interesting1
            cluster2 = np.concatenate(tuple([interesting1[mnd] for mnd in connected])+(cluster2,))
            interesting1 = [interesting1[mnd] for mnd in range(len(interesting1)) if (mnd not in connected)] + [cluster2]
        else:
            unconnected2.append(cluster2)

    # now concatenate all clusters
    return unconnected + interesting1 + unconnected2, (s1, e2)

#------------------------

def _structures(sel, start=0):
    """identify structures within a boolean array
    """
    sel = copy.copy(sel) # make a copy so we can update it in-place
    clusters = []

    # define the shift template
    shifts = _shape2shifts(sel.shape)

    while np.any(sel): # iterate until there are no more selected pixels, this loop in necessary
        pix = _sel2pix(sel) # grab a new pixel
        cluster, sel = _pix2cluster(pix, sel, shifts) # find everything that belongs in that pixel's cluster
        clusters.append(cluster) # record the cluster

    # return : list of clusters, each of which is a list of pixels
    if start:
        for cluster in clusters:
            cluster[:,0] += start

    return clusters

def _shape2shifts(shape):
    """figure out an array of possible shifts that correspond to neighbors in the array with shape given by "shape"
    """
    shifts = [()]
    for dim in range(len(shape)): # this loop is necessary, but will only be done once
        shifts = [shift+(0,) for shift in shifts] + [shift+(-1,) for shift in shifts] + [shift+(+1,) for shift in shifts]
    return np.array([np.array(shift) for shift in shifts if np.any(shift)], dtype=int)

def _sel2pix(sel):
    """pick a pixel from this boolean array
    """
    return np.transpose(np.nonzero(sel))[0]

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

        neighbors = _pix2neighbors(pix, shifts, sel.shape)
        successes = neighbors[sel[tuple(np.transpose(neighbors))]]

        cluster.append(successes)
        tocheck += list(successes)
        sel[tuple(np.transpose(successes))] = False

    # return
    return np.concatenate(tuple(cluster)), sel

def _pix2neighbors(pix, shifts, shape):
    """return a list of possible neighbors for this pix
    """
    neighbors = pix + shifts
    # exclude neighbors that fall outside grid boundaries
    return neighbors[np.all(neighbors >= 0, axis=1)*np.all(neighbors < shape, axis=1)]
