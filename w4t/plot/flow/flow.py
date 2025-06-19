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

def hist(ax, data, symmetric_xlim=False, grid=False, histtype='step', log=True, xlabel=None, title=None, **kwargs):
    """make a standard histogram
    """
    data = np.ravel(data)
    num = len(data)

    if symmetric_xlim:
        xlim = np.max(np.abs(data))
        xmin = -xlim
        xmax = +xlim
    else:
        xmin = np.min(data)
        xmax = np.max(data)

    bins = np.linspace(xmin, xmax, min(1000, max(10, int(num**0.5))))
    ax.hist(data, bins=bins, histtype=histtype, log=True, **kwargs)

    ax.tick_params(**TICK_PARAMS)
    ax.grid(grid, which='both')

    ax.set_xlim(xmin=bins[0], xmax=bins[-1])

    ax.set_ylabel('count')

    if xlabel:
        ax.set_xlabel(xlabel)

    if title:
        ax.set_title(title)

    return ax
