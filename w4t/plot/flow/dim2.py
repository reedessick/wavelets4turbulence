"""utils for plotting 2D data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from w4t.plot.plot import (plt, save, close)

from .flow import hist as _hist
from .flow import TICK_PARAMS as COLORBAR_TICK_PARAMS

from .dim1 import _plot as _dim1_plot
from .dim1 import FIGSIZE as DIM1_FIGSIZE

#-------------------------------------------------

FIGSIZE = (5.0, 5.0)

#---

IMSHOW_TICK_PARAMS = dict(
    left=False,
    right=False,
    top=False,
    bottom=False,
    direction='in',
    which='both',
)

#---

SUBPLOTS_ADJUST = dict(
    left=0.10,
    right=0.90,
    bottom=0.10,
    top=0.90,
    hspace=0.03,
    wspace=0.03,
)

#------------------------

CMAP = 'cividis'

LOG_POS_CMAP = 'YlOrRd'
LOG_NEG_CMAP = 'YlGnBu'

#-------------------------------------------------

def _plot(
        ax,
        data,
        aspect='auto',
        extent=(0, 1, 0, 1),
        log=False,
        grid=False,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        vmin=None,
        vmax=None,
        xlabel=None,
        ylabel=None,
        title=None,
        **kwargs
    ):
    """imshow a 2D array
    """

    if log:

        # limit the range to only a few orders of mag off the maximum
        vmax = np.max(np.log10(np.abs(data)))
        vmin = max(np.min(np.log10(np.abs(data))), vmax-3) 

        # positive values
        ax.imshow(
            np.transpose(np.where(data > 0, np.log10(np.abs(data)), np.nan)),
            cmap=LOG_POS_CMAP,
            vmin=vmin,
            vmax=vmax,
            aspect=aspect,
            origin='lower',
            extent=extent,
        )

        # negative values
        ax.imshow(
            np.transpose(np.where(data < 0, np.log10(np.abs(data)), np.nan)),
            cmap=LOG_NEG_CMAP,
            vmin=vmin,
            vmax=vmax,
            aspect=aspect,
            origin='lower',
            extent=extent,
        )

    else:
        ax.imshow(
            np.transpose(data),
            vmin=vmin,
            vmax=vmax,
            cmap=CMAP,
            aspect=aspect,
            origin='lower',
            extent=extent,
        )

    if xlabel:
        ax.set_xlabel(xlabel)
    if xmin is not None:
        ax.set_xlim(xmin=xmin)
    if xmax is not None:
        ax.set_xlim(xmax=xmax)

    if ylabel:
        ax.set_ylabel(ylabel)
    if ymin is not None:
        ax.set_ylim(ymin=ymin)
    if ymax is not None:
        ax.set_ylim(ymax=ymax)

    if title:
        ax.set_title(title)

    ax.set_xticks(ax.get_xticks()[1:-1])
    ax.set_yticks(ax.get_yticks()[1:-1])

    ax.tick_params(**IMSHOW_TICK_PARAMS)

    return ax

#---

def plot(approx, **kwargs):
    """plot a visualization of the flow
    """
    fig = plt.figure(figsize=FIGSIZE)
    _plot(plt.subplot(1,1,1), approx, **kwargs)
    plt.subplots_adjust(**SUBPLOTS_ADJUST)
    return fig

#-----------

def plot_coeff(aa, ad, da, dd, title=None, **kwargs):
    """plot visualization of wavelet coefficients
    """
    fig = plt.figure(figsize=FIGSIZE)

    #---

    for ind, data in enumerate([aa, da, ad, dd]):

        if np.prod(data.shape) == 0: # no data
            continue

        ax = _plot(plt.subplot(2,2,ind+1), data, **kwargs)

        if ind == 0:
            ax.set_xlabel('approximant')
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_ylabel('approximant')

        elif ind == 1:
            ax.set_xlabel('detail')
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_ylabel('approximant')
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

        elif ind == 2:
            ax.set_xlabel('approximant')
            ax.set_ylabel('detail')

        else:
            ax.set_xlabel('detail')
            ax.set_ylabel('detail')
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

def hist_coeff(aa, ad, da, dd, title=None, num_samples=True, **kwargs):
    """histogram wavelet coefficients
    """
    fig = plt.figure(figsize=FIGSIZE)

    #---

    for ind, (label, data) in enumerate([
            ('approx-approx', aa),
            ('detail-approx', da),
            ('approx-detail', ad),
            ('detail-detail', dd),
        ]):

        num = np.prod(data.shape)
        if num == 0: # no data
            continue

        ax = _hist(plt.subplot(2,2,ind+1), data, symmetric_xlim=(ind!=0), num_samples=num_samples, **kwargs)

        if ind < 2: # top row
            ax.xaxis.tick_top()

        if ind%2 == 1: # right column
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ax.text(xmin + 0.01*(xmax-xmin), ymax / (ymax/ymin)**0.01, label, ha='left', va='top')

    #---

    if title:
        fig.suptitle(title)

    #---

    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig

#-------------------------------------------------

def grand_tour(
        array,
        extent=[(0,1)]*2,
        increment=1,
        title=None,
        verbose=False,
        figtmp="grand_tour",
        figtype=["png"],
        dpi=None,
        **kwargs
    ):
    """make a sequence of plots showing the behavior of the function as we slice through the data
    """
    shape = array.shape
    assert len(shape) == 2, 'bad number of dimensions!'

    figtmp = figtmp + '-dim%d'
    alpha = 0.10

    cmap = plt.get_cmap(CMAP)

    for dim in range(2): # iterate over each dimension, making overlaid 1D plot for each
        fig = plt.figure(figsize=DIM1_FIGSIZE)

        ax = fig.add_axes([0.12, 0.12, 0.78, 0.80]) # scalogram
        cb = fig.add_axes([0.91, 0.12, 0.01, 0.80]) # colorbar

        if dim == 0:
            cblabel = 'x'
            xlabel = 'y'
            _extent = (extent[1][0], extent[1][1])
            cblim = (extent[0][0], extent[0][1])
        else:
            cblabel = 'y'
            xlabel = 'x'
            _extent = (extent[0][0], extent[0][1])
            cblim = (extent[1][0], extent[1][1])

        for ind in range(0, shape[dim], increment): # iterate over slice

            color = cmap((ind+0.5)/shape[dim])

            ax = _dim1_plot(
                ax,
                np.take(array, ind, axis=dim), # should be a 1D array
                extent=_extent,
#                symmetric_ylim=True,
                ylabel=title,
                color=color,
                alpha=alpha,
                **kwargs
            )

        ax.set_xlabel(xlabel)

        # add colorbar

        gradient = np.linspace(0, 1, 256)
        gradient = np.transpose(np.vstack((gradient, gradient)))

        cb.imshow(gradient, aspect='auto', cmap=CMAP, origin='lower', extent=(0,1,0,1))

        cb.set_xticks([])

        cb.set_ylim(*cblim)

        cb.set_ylabel(cblabel)
        cb.yaxis.tick_right()
        cb.yaxis.set_label_position('right')

        cb.tick_params(**COLORBAR_TICK_PARAMS)

        # save figure
        save(fig, (figtmp % dim) + '.%s', figtype, verbose=verbose, dpi=dpi)
        close(fig)
