"""a simple module for a few common functions related to flow visualization
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

TICK_PARAMS = dict(
    left=True,
    right=True,
    top=True,
    bottom=True,
    direction='in',
    which='both',
)

#-------------------------------------------------

def hist(ax, data, grid=False, **kwargs):
    """make a standard histogram
    """
    data = np.ravel(data)
    xlim = np.max(np.abs(data))
    bins = np.linspace(-xlim, +xlim, min(1000, max(10, int(num**0.5))))

    ax.hist(data, bins=bins, **kwargs)

    ax.tick_params(**TICK_PARAMS)
    ax.grid(grid, which='both')

    return ax
