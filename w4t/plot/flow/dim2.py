"""utils for plotting 2D data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from w4t.plot.plot import plt

from .flow import hist as _hist

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
    left=0.075,
    right=0.925,
    bottom=0.05,
    top=0.90,
    hspace=0.03,
    wspace=0.03,
)

#------------------------

CMAP = 'RdGy'

LOG_POS_CMAP = 'YlOrRd'
LOG_NEG_CMAP = 'YlGnBu'

#-------------------------------------------------

def _plot(
        ax,
        data,
        log=False,
        grid=False,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        vmin=None,
        vmax=None,
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
            origin='lower',
            extent=(0, 1, 0, 1),
        )

        # negative values
        ax.imshow(
            np.transpose(np.where(data < 0, np.log10(np.abs(data)), np.nan)),
            cmap=LOG_NEG_CMAP,
            vmin=vmin,
            vmax=vmax,
            origin='lower',
            extent=(0, 1, 0, 1),
        )

    else:
        ax.imshow(
            np.transpose(data),
            vmin=vmin,
            vmax=vmax,
            cmap=CMAP,
            origin='lower',
            extent=(0, 1, 0, 1),
        )

    if xmin is not None:
        ax.set_xlim(xmin=xmin)
    if xmax is not None:
        ax.set_xlim(xmax=xmax)
    if ymin is not None:
        ax.set_ylim(ymin=ymin)
    if ymax is not None:
        ax.set_ylim(ymax=ymax)

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

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

def plot_coeff(aa, ad, da, dd, **kwargs):
    """plot visualization of wavelet coefficients
    """
    fig = plt.figure(figsize=FIGSIZE)

    #---

    for ind, data in enumerate([aa, ad, da, dd]):

        if np.prod(data.shape) == 0: # no data
            continue

        ax = _plot(plt.subplot(2,2,ind+1), data, **kwargs)

        if ind == 0:
            ax.set_xlabel('approximant')
            ax.xaxis.set_label_position('top')
            ax.set_ylabel('approximant')

        elif ind == 1:
            ax.set_xlabel('detail')
            ax.xaxis.set_label_position('top')
            ax.set_ylabel('approximant')
            ax.yaxis.set_label_position('right')

        elif ind == 2:
            ax.set_xlabel('approximant')
            ax.set_ylabel('detail')

        else:
            ax.set_xlabel('detail')
            ax.set_ylabel('detail')
            ax.yaxis.set_label_position('right')

    #---

    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig

#------------------------

def hist(approx, **kwargs):
    """histogram 1D data
    """
    fig = plt.figure(figsize=FIGSIZE)
    _hist(plt.subplot(1,1,1), approx, **kwargs)
    plt.subplots_adjust(**SUBPLOTS_ADJUST)
    return fig

#-----------

def hist_coeff(aa, ad, da, dd, **kwargs):
    """histogram wavelet coefficients
    """
    fig = plt.figure(figsize=FIGSIZE)

    #---

    for ind, (label, data) in enumerate([
            ('approx-approx', aa)
            ('approx-detail', ad),
            ('detail-approx', da),
            ('detail-detail', dd),
        ]):

        num = np.prod(data.shape)
        if num == 0: # no data
            continue

        ax = _hist(plt.subplot(2,2,ind+1), data, symmetric_xlim=(ind!=0), **kwargs)

        if ind < 2: # top row
            ax.xaxis.tick_top()

        if ind%2 == 0: # right column
            ax.yaxis.tick_right()

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ax.text(xmin + 0.01*(xmax-xmin), ymax / (ymax/ymin)**0.01, '%s\n%d samples' % (label, num), ha='left', va='top')

    #---

    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig
