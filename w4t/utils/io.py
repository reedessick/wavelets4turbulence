"""a module that houses basic utilities, like estimating moments from monte carlo samples
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import h5py
import numpy as np

### non-standard libraries
try:
    from PLASMAtools.read_funcs.read import Fields
    from PLASMAtools.aux_funcs import derived_var_funcs as dv
except ImportError:
    Fields = None

from w4t.w4t.w4t import Structure

#-------------------------------------------------

DEFAULT_NUM_GRID = 32
DEFAULT_NUM_DIM = 3

#-------------------------------------------------

def write(data, path, verbose=False, Verbose=False, **kwargs):
    """write data into a standard HDF file
    """
    verbose |= Verbose

    if verbose:
        print('writing: '+path)

    with h5py.File(path, 'w') as obj:
        for key, val in kwargs.items():
            obj.attrs.create(key, data=val)

        for key in data.keys():
            if Verbose:
                print('    writing: %s %s' % (key, str(np.shape(data[key]))))
            obj.create_dataset(key, data=data[key])

#-----------

def load(
        fields,
        path=None,
        flash_format=False,
        num_grid=DEFAULT_NUM_GRID,
        num_dim=DEFAULT_NUM_DIM,
        max_edgelength=None,
        verbose=False,
        Verbose=False,
    ):
    """standardize logic for loading data and/or generating synthetic data
    """
    verbose |= Verbose
    data = dict()

    if path is not None: # read data from file
        if verbose:
            print('loading: '+path)

        if flash_format:
            if Fields is None:
                raise ImportError('could not import PLASMAtools.read_funcs.read.Fields')

            parser = Fields(path, reformat=True)

            for field in fields:
                if field == 'vort':
                    parser.read('vel')
                    data['vel'] = parser.vel 
                elif field == 'curr':
                    parser.read('mag')
                    data['mag'] = parser.mag
                else:
                    parser.read(field)
                    data[field] = getattr(parser, field)

            del parser

        else:
            with h5py.File(path, 'r') as obj:
                for field in fields:
                    if field in obj.keys():
                        data[field] = obj[field][:]
                    else:
                        if field == 'vort':
                            data['vel'] = obj['vel'][:]
                        elif field == 'curr':
                            data['mag'] = obj['mag'][:]
                        else:
                            raise RuntimeError('could not load field='+field)

        # read the fields
        compute_vort = ('vort' in fields) and ('vort' not in data)
        compute_curr = ('curr' in fields) and ('curr' not in data)

        if compute_vort or compute_curr:
            dvf = dv.DerivedVars()

            if compute_vort:
                if Verbose:
                    print('computing vort as curl(vel)')
                data['vort'] = dvf.vector_curl(data['vel'])

                if 'vel' not in fields:
                    data.pop('vel')

            if compute_curr:
                if Verbose:
                    print('computing curr as curl(mag)')
                data['curr'] = dvf.vector_curl(data['mag'])

                if 'mag' not in fields:
                    data.pop('mag')

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

#------------------------

def simplify(data, field, component=None, max_edgelength=None, verbose=False):
    """further extract specific components of an array and standardize truncation
    """
    if component is None:
        if data.shape[0] > 1: # take the magnitude
            if verbose:
                print('extracting magnitude')
            data = np.sum(data**2, axis=0)**0.5
            field = field+"_mag"
        else:
            data = data[0]
            field = field

    else: # take a specific component
        if verbose:
            print('extracting component: %d' % component)
        data = data[component]
        field = "%s_%d" % (field, component)

    if max_edgelength is not None:
        if verbose:
            print('retaining the first %d samples in each dimension' % max_edgelength)
        data = data[tuple(slice(max_edgelength) for _ in range(len(data.shape)))]

    if verbose:
        print('    %s %s' % (field, data.shape))

    return data, field

#-------------------------------------------------

def write_structures(structures, path, verbose=False, Verbose=False, **kwargs):
    """write lists of pixels corresponding to separate structures to disk
    """
    verbose |= Verbose

    if verbose:
        print('writing structures: '+path)

    with h5py.File(path, 'w') as obj:
        for key, val in kwargs.items():
            obj.attrs.create(key, data=val)

        for ind, structure in enumerate(structures):
            key = str(ind)
            grp = obj.create_group(key)
            if Verbose:
                print('    writing: %s %s' % (key, len(structure)))

            grp.attrs.create('levels', structure.levels)
            grp.attrs.create('shape', structure.shape)

            grp.create_dataset('pixels', data=structure.pixels)

#-----------

def load_structures(path, verbose=False):
    """load lists of pixels corresponding to separate structures from disk
    """
    if verbose:
        print('loading structures: '+path)

    with h5py.File(path, 'r') as obj:
        keys = list(obj.keys())
        keys.sort(key=lambda x: int(x))
        structures = [Structure(obj[key]['pixels'][:], obj[key].attrs['levels'], obj[key].attrs['shape']) for key in keys]

    return structures

#-------------------------------------------------

def write_structure_function(scales, index, mom, cov, path, verbose=False, **kwargs):
    """write structure functions to disk
    """
    if verbose:
        print('writing structure functions: '+path)

    with h5py.File(path, 'w') as obj:
        for key, val in kwargs.items():
            obj.attrs.create(key, data=val)

        obj.create_dataset('scales', data=scales)
        obj.create_dataset('index', data=index)
        obj.create_dataset('mom', data=mom)
        obj.create_dataset('cov', data=cov)

#-----------

def load_structure_function(path, verbose=False):
    """load structure functions from disk
    """
    if verbose:
        print('loading structure functions: '+path)

    with h5py.File(path, 'r') as obj:
        scales = obj['scales'][:]
        index = obj['index'][:]
        mom = obj['mom'][:]
        cov = obj['cov'][:]

    return scales, index, mom, cov

#-------------------------------------------------

def write_scaling_exponent(poly, index, degree, path, verbose=False, **kwargs):
    """write polynomial fits to disk
    """
    if verbose:
        print('writing scaling exponent: '+path)

    with h5py.File(path, 'w') as obj:
        for key, val in kwargs.items():
            obj.attrs.create(key, data=val)

        obj.attrs.create('degree', data=degree)

        obj.create_dataset('index', data=index)
        obj.create_dataset('poly', data=poly)

#-----------

def load_scaling_exponent(path, verbose=False):
    if verbose:
        print('loading scaling exponent: '+path)

    with h5py.File(path, 'r') as obj:
        m = obj.attrs['min_scale']
        M = obj.attrs['max_scale']
        degree = obj.attrs['degree']
        index = obj['index'][:]
        poly = obj['poly'][:]

    return poly, index, degree, (m, M)

#-------------------------------------------------

def write_structure_function_ansatz_samples(posterior, prior, scales, index, path, verbose=False, **kwargs):
    """write posterior samples for structure function ansatz to disk
    """
    if verbose:
        print('writing scaling exponent samples: '+path)

    with h5py.File(path, 'w') as obj:
        for key, val in kwargs.items():
            obj.attrs.create(key, data=val)

        obj.create_dataset('index', data=index)
        obj.create_dataset('scales', data=scales)

        for label, data in [('posterior', posterior), ('prior', prior)]:
            for ind, val in data.items():
                grp = obj.create_group('%s_%d' % (label, ind))
                for key, val in val.items():
                    grp.create_dataset(key, data=val)

#-----------

def load_structure_function_ansatz_samples(path, verbose=False):
    """load posterior samples for structure function ansatz from disk
    """
    if verbose:
        print('loading scaling exponent samples: '+path)

    with h5py.File(path, 'r') as obj:
        index = obj['index'][:]
        scales = obj['scales'][:]

        posterior = dict()
        prior = dict()
        for ind in index:
            for label, data in [('posterior', posterior), ('prior', prior)]:
                key = '%s_%d' % (label, ind)
                data[ind] = dict((k, obj[key][k][:]) for k in obj[key].keys())

    return posterior, prior, scales, index
