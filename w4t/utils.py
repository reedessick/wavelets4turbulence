"""a module that houses basic utilities, like estimating moments from monte carlo samples
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np
from scipy.special import comb # comb(n, k) = "n choose k" = n! / ((n-k)! k!)

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

#-------------------------------------------------

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
            turb.read(field)
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

#-------------------------------------------------

def load_flash(path, fields):
    """load data from a Flash simulation
    """
    raise NotImplementedError('''
        # simulation attributes
        self.filename           = filename
        self.reformat           = reformat
        self.sim_data_type      = sim_data_type
        self.n_cores            = 0
        self.nxb                = 0
        self.nyb                = 0
        self.nzb                = 0
        self.n_cells            = 0
        self.int_properties     = {}
        self.str_properties     = {}
        self.logic_properties   = {}

        # read in the simulation properties
        if self.sim_data_type == "flash":
            self.__read_sim_properties()

        # grid data attributes
        # if the data is going to be reformated, preallocate the 3D
        # arrays for the grid data
        if self.reformat:
            init_field = np.zeros((self.nyb*self.int_properties["jprocs"],
                                   self.nxb*self.int_properties["iprocs"],
                                   self.nzb*self.int_properties["kprocs"]), dtype=np.float32)
        else:
            # otherwise, preallocate the 1D arrays for the grid data
            init_field = np.zeros(self.n_cells, dtype=np.float32)

        # add a new method here for initialisation 
        # (could speed up IO a bit but haven't properly timed it)

        # read in state (read in true or false if the reader
        # has actually been called -- this is all to save time 
        # with derived vars. Note that the way that it is 
        # written means that it will create new read_... state variables
        # even if one hasn't been initialised here.)
        self.read_dens          = False
        self.read_vel           = False
        self.read_mag           = False
        self.derived_cur        = False

        # use a CGS unit system
        self.mu0                = 4 * np.pi

    def __read_sim_properties(self) -> None:
        """
        This function reads in the FLASH field properties directly from
        the hdf5 file metadata. 
        
        Author: James Beattie
        
        """
        g = File(self.filename, 'r')
        self.int_scalars        = {str(key).split("'")[1].strip(): value for key, value in g["integer scalars"]}
        self.int_properties     = {str(key).split("'")[1].strip(): value for key, value in g["integer runtime parameters"]}
        self.str_properties     = {str(key).split("'")[1].strip(): str(value).split("'")[1].strip() for key, value in g["string runtime parameters"]}
        self.logic_properties   = {str(key).split("'")[1].strip(): value for key, value in g["logical runtime parameters"]}
        g.close()

        # read out properties of the grid
        self.n_cores  = self.int_scalars["globalnumblocks"]
        self.nxb      = self.int_scalars["nxb"]
        self.nyb      = self.int_scalars["nyb"]
        self.nzb      = self.int_scalars["nzb"]
        self.iprocs   = self.int_properties["iprocs"]
        self.jprocs   = self.int_properties["jprocs"]
        self.kprocs   = self.int_properties["kprocs"]
        self.n_cells  = self.n_cores*self.nxb*self.nyb*self.nzb
        self.plot_file_num = self.int_scalars["plotfilenumber"]


    def read(self,
             field_str          : str,
             vector_magnitude   : bool = False,
             debug              : bool = False,
             interpolate        : bool = True,
             N_grid_x           : int  = 256,
             N_grid_y           : int  = 256,
             N_grid_z           : int  = 256,
             verbose = False,
        ) -> None:
        """
        This function reads in grid data.
        
        Args:
            field_str (str):            The field to read in.
            vector_magnitude (bool):    Whether to read in the magnitude of the vector field.
            debug (bool):               Whether to print debug information.
            N_grid_x (int):             The number of grid points to interpolate onto in the x direction.
            N_grid_y (int):             The number of grid points to interpolate onto in the y direction.
            N_grid_z (int):             The number of grid points to interpolate onto in the z direction.
        
        """

        setattr(self,f"read_{field_str}",True)

        if self.sim_data_type == "flash":
            # setting the read in states here (this adds an attribute)
            if field_lookup_type[field_str] == "scalar":
                g = File(self.filename, 'r')
                if verbose:
                    print(f"Reading in grid attribute: {field_str}")

                if self.reformat:
                    if verbose:
                        print(f"Reading in reformatted grid attribute: {field_str}")

                    setattr(self, field_str,
                            np.array([reformat_FLASH_field(g[field_str][:,:,:,:],
                                                 self.nxb,
                                                 self.nyb,
                                                 self.nzb,
                                                 self.iprocs,
                                                 self.jprocs,
                                                 self.kprocs,
                                                 debug)]))
                else:
                    setattr(self, field_str, np.array([g[field_str][:,:,:,:]]))
                g.close()


def reformat_FLASH_field(field  : np.ndarray,
                         nxb    : int,
                         nyb    : int,
                         nzb    : int,
                         iprocs : int,
                         jprocs : int,
                         kprocs : int,
                         debug  : bool) -> np.ndarray:
    """
    This function reformats the FLASH block / core format into
    (x,y,z) format for processing in real-space coordinates utilising
    numba's jit compiler, producing roughly a two-orders of magnitude
    speedup compared to the pure python version.

    INPUTS:
    field   - the FLASH field in (core,block_x,block_y,block_z) coordinates
    iprocs  - the number of cores in the x-direction
    jprocs  - the number of cores in the y-direction
    kprocs  - the number of cores in the z-direction
    debug   - flag to print debug information


    OUTPUTs:
    field_sorted - the organised 3D field in (x,y,z) coordinates

    """

    if debug:
        print(f"reformat_FLASH_field: nxb = {nxb}")
        print(f"reformat_FLASH_field: nyb = {nyb}")
        print(f"reformat_FLASH_field: nzb = {nzb}")
        print(f"reformat_FLASH_field: iprocs = {iprocs}")
        print(f"reformat_FLASH_field: jprocs = {jprocs}")
        print(f"reformat_FLASH_field: kprocs = {kprocs}")

    # Initialise an empty (x,y,z) field
    # has to be the same dtype as input field (single precision)
    # swap axes to get the correct orientation
    # x = 0, y = 1, z = 2
    return np.transpose(sort_flash_field(field,
                                         nxb,
                                         nyb,
                                         nzb,
                                         iprocs,
                                         jprocs,
                                         kprocs),
                        (2,1,0))

def sort_flash_field(field    : np.ndarray,
                     nxb      : int,
                     nyb      : int,
                     nzb      : int,
                     iprocs   : int,
                     jprocs   : int,
                     kprocs   : int) -> np.ndarray:

    # Initialise an empty (x,y,z) field
    field_sorted = np.zeros((nyb*jprocs,
                             nzb*kprocs,
                             nxb*iprocs),
                            dtype=np.float32)

    # The block counter for looping through blocks
    block_counter = 0

    # Sort the unsorted field
    for j in range(jprocs):
        for k in range(kprocs):
            for i in range(iprocs):
                field_sorted[j*nyb:(j+1)*nyb, k*nzb:(k+1)*nzb, i*nxb:(i+1)*nxb] = field[block_counter, :, :, :]
                block_counter += 1
    return field_sorted

''')

#-------------------------------------------------

def moments(samples, index):
    """estimate moments of samples for each value in index (which should be an iterable). For example, index=[1,2] will compute the 1st and second moment of samples. Also estimates the covariance matrix between the estimators for the requested moments.
    """
    index = np.array(sorted(index), dtype=int)

    num_index = len(index)
    num_samples = len(samples)

    # compute point estimates
    m = np.array([np.sum(samples**ind)/num_samples for ind in index], dtype=float)

    # compute covariance matrix
    c = np.empty((num_index, num_index), dtype=float)
    for i in range(num_index):
        for j in range(i+1):
            # note, this may repeat some sums, but that shouldn't be much extra overhead
            c[i,j] = c[j,i] = (np.sum(samples**(index[i]+index[j]))/num_samples - m[i]*m[j]) / num_samples

    # return
    return index, m, c

def central_moments(samples, index):
    """returns the central moments instead of just the moments
    """
    index = np.array(sorted(index), dtype=int)
    num_index = len(index)

    # compute all moments up to the maximum index requested
    i, m, c = moments(samples, range(1,index[-1]+1))

    # compute point estimates
    mom = np.zeros(num_index, dtype=float)
    for j, ind in enumerate(index):
        for k in range(0, ind+1): # iterate over terms
            mom[ind] += m[k] * (-m[0])**(ind-k) * comb(ind, k)

    # compute covariance matrix
    raise NotImplementedError('compute covariance matrix!')

    # return
    return index, mom, cov
