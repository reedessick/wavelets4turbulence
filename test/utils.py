"""a module to hold some basic utilities that will appear in multiple test scripts
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

### non-standard libraries
try:
    from PLASMAtools.read_funcs.read import Fields
except ImportError:
    Fields = None

#-------------------------------------------------

DEFAULT_NUM_GRID = 32
DEFAULT_NUM_DIM = 3

#-------------------------------------------------

def seed(num=None, verbose=False):
    if num is not None:
        if verbose:
            print('setting numpy.random.seed=%d' % num)
        np.random.seed(num)

#------------------------

def load(fields, path=None, num_grid=DEFAULT_NUM_GRID, num_dim=DEFAULT_NUM_DIM, max_edgelength=None, verbose=False, Verbose=False):
    """standardize logic for loading data and/or generating synthetic data
    """
    verbose |= Verbose
    data = dict()

    if path is not None: # read data from file
        if verbose:
            print('loading: '+path)

        if Fields is None:
            raise ImportError('could not import PLASMAtools.read_funcs.read.Fields')

        turb = Fields(path, reformat=True)

        # read the fields
        for field in fields:
            turb.read(field, verbose=Verbose)
            data[field] = getattr(turb, field) # replacement for this syntax: turb.vel

        del turb # get rid of this object to save memory

    else: # generate random data on a big-ish 3D array

        shape = (1,)+(num_grid,)*num_dim
        if verbose:
            print('generating randomized data with shape: %s' % (shape,))

        # use grid to compute coherent structure
        x = np.arange(num_grid) / num_grid
        if num_dim > 1:
            xs = np.meshgrid(*(x for x in range(num_dim)), indexing='ij')
            coherent = 0.5*np.exp(-0.5*np.sum((xs[:-1]-xs[-1])**2)/0.1**2) ### a tube

        else:
            coherent = 0.5*np.exp(-0.5*(x-0.5)**2/0.1**2) ### a bump

        # iterate through fields and add Gaussia noise
        for field in fields:
            data[field] = coherent + np.random.normal(size=shape)

    #---

    if verbose:
        for field in fields:
            print('    '+field, data[field].shape) # expect [num_dim, num_x, num_y, num_z]

    #---

    if max_edgelength is not None:
        if verbose:
            print('limiting data size by selecting the first max(edgelength)=%d samples' % max_edgelength)

        for key in data.keys():
            data[key] = data[key][:, :max_edgelength, :max_edgelength, :max_edgelength]

    #---

    return data
