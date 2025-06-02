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

#-------------------------------------------------

DEFAULT_NUM_GRID = 32
DEFAULT_NUM_DIM = 3

#-------------------------------------------------

def write(
        data,
        path,
        verbose=False,
        Verbose=False,
    ):
    """write data into a standard HDF file
    """
    verbose |= Verbose

    if verbose:
        print('writing: '+path)

    with h5py.File(path, 'w') as obj:
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

        if flash:
            if Fields is None:
                raise ImportError('could not import PLASMAtools.read_funcs.read.Fields')

            turb = Fields(path, reformat=True)

            for field in fields:
                if field == 'vort':
                    turb.read('vel')
                    data['vel'] = turb.vel 
                elif field == 'curr':
                    trub.read('mag')
                    data['mag'] = turb.mag
                else:
                    turb.read(field)
                    data[field] = getattr(turb, field)

            del turb

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
