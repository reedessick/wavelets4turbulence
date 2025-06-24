"""a module that houses logic about plotting 1D data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from w4t.plot.plot import plt

from .flow import hist as _hist

#-------------------------------------------------

FIGSIZE = (5.0, 3.0)

#---

TICK_PARAMS = dict(
    left=True,
    right=True,
    top=True,
    bottom=True,
    which='both',
    direction='in',
)

SUBPLOTS_ADJUST = dict(
    left=0.10,
    right=0.90,
    bottom=0.15,
    top=0.875,
    hspace=0.03,
    wspace=0.03,
)

#---

SCALOGRAM_CMAP = 'Reds'

#-------------------------------------------------

def _plot(ax, data, extent=(0, 1), symmetric_ylim=False, grid=False, xlabel=None, ylabel=None, title=None, **kwargs):
    num = len(data)
    dx = (extent[1]-extent[0])/num
    ax.plot(np.arange(extent[0]+0.5*dx, extent[1], dx), data, **kwargs)

    ax.set_xlim(xmin=extent[0], xmax=extent[1])
#    ax.set_xticks(ax.get_xticks()[1:-1])

    ax.tick_params(**TICK_PARAMS)
    ax.grid(grid, which='both')

    if symmetric_ylim:
        ylim = np.max(np.abs(ax.get_ylim()))
        ax.set_ylim(ymin=-ylim, ymax=+ylim)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return ax

#---

def plot(approx, title=None, **kwargs):
    """plot 1D data
    """
    fig = plt.figure(figsize=FIGSIZE)
    _plot(plt.subplot(1,1,1), approx, ylabel=title, **kwargs)
    plt.subplots_adjust(**SUBPLOTS_ADJUST)
    return fig

#-----------

def plot_coeff(approx, detail, title=None, **kwargs):
    """plot 1D data from a decomposed array
    """
    fig = plt.figure(figsize=FIGSIZE)

    #---

    for ind, (label, data) in enumerate([('approx', approx), ('detail', detail)]):

        if np.prod(data.shape) == 0: # no data
            continue

        ax = _plot(plt.subplot(1,2,ind+1), data, symmetric_ylim=(ind>0), **kwargs)

        ax.set_ylabel(label)

        if ind == 1:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

    #---

    if title:
        fig.suptitle(title)

    #---

    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig

#------------------------

def hist(approx, title=None, num_samples=True, **kwargs):
    """histogram 1D data
    """
    fig = plt.figure(figsize=FIGSIZE)
    _hist(plt.subplot(1,1,1), approx, xlabel=title, num_samples=num_samples, **kwargs)
    plt.subplots_adjust(**SUBPLOTS_ADJUST)
    return fig

#-----------

def hist_coeff(approx, detail, title=None, num_samples=True, **kwargs):
    """plot histograms of coefficients from a 1D decomposed array
    """
    fig = plt.figure(figsize=FIGSIZE)

    #---

    for ind, (label, data) in enumerate([('approx', approx), ('detail', detail)]):

        num = np.prod(data.shape)
        if num == 0: # no data
            continue

        ax = _hist(plt.subplot(1,2,ind+1), data, symmetric_xlim=(ind!=0), num_samples=num_samples, **kwargs)

        ax.set_xlabel(label)

        if ind:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

    #---

    if title:
        fig.suptitle(title)

    #---

    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig

#-------------------------------------------------

def scalogram(waveletarray, map2scalar, **kwargs):
    """plot a scalogram of 1D WaveletArray
    """
    assert waveletarray.ndim == 1, 'can only plot scalogram of 1D WaveletArray'

    fig = plt.figure()

    ax1 = fig.add_axes([0.10, 0.10, 0.80, 0.70]) # scalogram
    ax3 = fig.add_axes([0.10, 0.81, 0.80, 0.14]) # raw data
    ax2 = fig.add_axes([0.91, 0.10, 0.01, 0.70]) # colorbar

    #---

    waveletarray.decompose()

    if waveletarray.active[0] == 1: # ignore the lowest order
        waveletarray.idwt()

    X = []
    Y = []
    Z = []
    scales = []

    while waveletarray.scales[0] > 1:
        scales.append(waveletarray.scales[0])

        xs = np.arange(waveletarray.active[0], dtype=float) / waveletarray.active[0]
        xs += 0.5*(xs[1]-xs[0])

        # add to arrays for scatter points
        X.append(xs)
        Y.append(waveletarray.scales[0]*np.ones(waveletarray.active[0]))

        detail = np.array(map2scalar(waveletarray.detail[:])) # make a copy to avoid the fact that ha will edit this in-place
        s = np.std(detail)
        if s > 0:
            detail /= s # only scale this if there is some variation
        Z.append( detail )

        # iterate
        waveletarray.idwt()

    # plot the scalogram as a scatter
    vlim = np.max([np.max(np.abs(z)) for z in Z])

    cmap = plt.get_cmap(SCALOGRAM_CMAP)

    for x, y, z in zip(X, Y, Z): # plot tiles
        ymin = y/2**0.5
        ymax = y*2**0.5

        dx = (x[1]-x[0])/2
        xmin = x-dx
        xmax = x+dx

        for xmin, xmax, ymin, ymax, z in zip(xmin, xmax, ymin, ymax, z):
            color = cmap((z+vlim)/(2*vlim))
            ax1.fill_between([xmin, xmax], [ymin]*2, [ymax]*2, color=color)

    # decorate spectogram

    ax1.set_yscale('log')
    ax1.set_ylim(
        ymin=np.max(scales)*2**0.5,
        ymax=np.min(scales)/2**0.5,
    )

    ax1.set_yticks(scales, minor=False)
    ax1.set_yticks([], minor=True)
    ax1.set_yticklabels(['%d'%_ for _ in scales])

    ax1.set_ylabel('scale')

    ax1.set_xlim(xmin=0, xmax=1)

    ax1.tick_params(**TICK_PARAMS)

    #---

    # add colorbar to ax2

    gradient = np.linspace(-1, 1, 256)*vlim
    gradient = np.transpose(np.vstack((gradient, gradient)))

    ax2.imshow(gradient, aspect='auto', cmap=cmap, origin='lower', extent=(0,1,-vlim,+vlim))

    ax2.set_xticks([])

    ax2.set_ylim(ymin=-vlim, ymax=+vlim)

    ax2.set_ylabel('scaled detail')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')

    ax2.tick_params(**TICK_PARAMS)

    #---

    # plot raw data

    ax3.plot(np.arange(len(waveletarray.array))/len(waveletarray.array), waveletarray.array, )

    ax3.set_xlim(ax1.get_xlim())
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax3.tick_params(**TICK_PARAMS)

    #---

    return fig
