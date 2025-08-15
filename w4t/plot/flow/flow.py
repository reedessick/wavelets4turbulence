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

def hist(
        ax,
        data,
        nonzero=False,
        symmetric_xlim=False,
        grid=False,
        histtype='step',
        log=True,
        xlabel=None,
        title=None,
        num_samples=False,
        **kwargs
    ):
    """make a standard histogram
    """
    data = np.ravel(data)

    if nonzero:
        data = data[data != 0] # only consider non-zero values

    data = data[data==data] # only count things that are not nan
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

    diff = (xmax-xmin)*0.05
    ax.set_xlim(xmin=bins[0]-diff, xmax=bins[-1]+diff)

    ax.set_ylabel('count')

    if xlabel:
        ax.set_xlabel(xlabel)

    if title:
        ax.set_title(title)

    if num_samples:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(
            xmax - 0.01*(xmax-xmin),
            ymax / (ymax/ymin)**0.01,
            '%d samples\n$\mu = %s$\n$\sigma = %s$' % (num, np.mean(data), np.std(data)),
            ha='right',
            va='top',
        )

    return ax
