"""utils for plotting 3D data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from w4t.plot.plot import (plt, save, close)

from .flow import hist as _hist

from .dim2 import _plot as _dim2_plot
from .dim2 import FIGSIZE as DIM2_FIGSIZE
from .dim2 import SUBPLOTS_ADJUST as DIM2_SUBPLOTS_ADJUST

#-------------------------------------------------

FIGSIZE = (5.0, 5.0)

#---

_TICK_PARAMS = dict(
    left=True,
    right=True,
    top=True,
    bottom=True,
    which='both',
    direction='in',
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

#---

CMAP = 'RdGy'

LOG_POS_CMAP = 'YlOrRd'
LOG_NEG_CMAP = 'YlGnBu'

#-------------------------------------------------

def _plot(ax11, ax12, ax22, data, grid=False, **kwargs):
    """plot a visualization of the flow
    """
    assert len(np.shape(data)) == 3, 'data must be 3-dimensional'
 
    for ind, (ax, dim, xlabel, ylabel, transpose) in enumerate([
            (ax11, 1, 'x', 'z', False),
            (ax12, 2, 'x', 'y', False),
            (ax22, 0, 'z', 'y', True),
        ]):
        d = np.mean(data, axis=dim) # average along one dimension
        if transpose: # make sure we have the correct orientation of axes (x-axis is index 0, y-axis is index 1)
            d = np.transpose(d)

        ax = _dim2_plot(ax, d)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if ind == 0:
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')

        elif ind == 2:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

    #---

    return ax11, ax12, ax22

#---

def plot(approx, title=None, **kwargs):
    """plot a visualization of the flow
    """
    fig = plt.figure(figsize=FIGSIZE)
    _plot(
        plt.subplot(2,2,1),
        plt.subplot(2,2,3),
        plt.subplot(2,2,4),
        approx,
        **kwargs
    )
    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    if title:
        fig.text(0.75, 0.75, title, ha='center', va='center')

    return fig

#-----------

def plot_coeff(aaa, aad, ada, daa, add, dad, dda, ddd, **kwargs):
    """plot visualization of wavelet coefficients
    """
    fig = plt.figure(figsize=FIGSIZE)

    #---

    for row, group in enumerate([
            [('approx-approx-approx', aaa)],
            [('detail-approx-approx', daa), ('approx-detail-approx', ada), ('approx-approx-detail', aad)],
            [('approx-detail-detail', add), ('detail-approx-detail', dad), ('detail-detail-approx', dda)],
            [('detail-detail-detail', ddd)],
        ]):

        _ymin = +np.inf
        _ymax = -np.inf
        text = []

        for col, (label, data) in enumerate(group):

            num = np.prod(data.shape)
            if num == 0: # no data
                continue

            ax11, ax12, ax22 = _plot(
                plt.subplot(8,6,row*6 + col + 1),
                plt.subplot(8,6,(row+1)*6 + col + 1),
                plt.subplot(8,6,(row+1)*6 + col + 2),
                data,
                **kwargs
            )

            ax11.set_title(label)
            fig.text(col/3, 1-0.125-row/4, label, ha='center', va='center')

    #---

    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig

#------------------------

def hist(approx, **kwargs):
    """histogram approx
    """
    fig = plt.figure(figsize=FIGSIZE)
    _hist(plt.subplot(1,1,1), approx, **kwargs)
    plt.subplots_adjust(**SUBPLOTS_ADJUST)
    return fig

#-----------

def hist_coeff(aaa, aad, ada, daa, add, dad, dda, ddd, **kwargs):
    """histogram wavelet coefficients
    """
    fig = plt.figure(figsize=FIGSIZE)

    #---

    for row, group in enumerate([
            [('approx-approx-approx', aaa)],
            [('detail-approx-approx', daa), ('approx-detail-approx', ada), ('approx-approx-detail', aad)],
            [('approx-detail-detail', add), ('detail-approx-detail', dad), ('detail-detail-approx', dda)],
            [('detail-detail-detail', ddd)],
        ]):

        _ymin = +np.inf
        _ymax = -np.inf
        text = []

        for col, (label, data) in enumerate(group):

            num = np.prod(data.shape)
            if num == 0: # no data
                continue

            ax = _hist(plt.subplot(4,3,row*3+col+1), data, symmetric_xlim=(ind!=0), **kwargs)

            ymin, ymax = ax.get_ylim()

            _ymin = min(_ymin, ymin)
            _ymax = max(_ymax, ymax)

            text.append((ax, xmin + 0.01*(xmax-xmin), '%s\n%d samples' % (label, num), 'left', 'top'))

        y = ymax / (ymax/ymin)**0.01
        for ax, x, text, ha, va in text:
            ax.set_ylim(_ymin, _ymax)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.text(x, y, text, ha=ha, va=va)

    #---

    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig

#-------------------------------------------------

def grand_tour(array, verbose=False, figtmp="grand_tour", figtype=["png"], dpi=None, **kwargs):
    """make a sequence of plots showing the behavior of the function as we slice through the data
    """
    shape = array.shape
    assert len(shape) == 3, 'bad number of dimensions!'

    figtmp = figtmp + '-%06d-%06d'

    for dim in range(3): # iterate over each dimension, making overlaid 1D plot for each
        for ind in range(shape[dim]): # iterate over slices
            fig = plt.figure(figsize=DIM2_FIGSIZE)
            ax = plt.subplot(1,1,1)

            ax = _dim2_plot(
                ax,
                np.take(data, ind, axis=dim), # should be a 2D array
                **kwargs
            )

            ax.set_title('dim=%d\nind=%d' % (dim, ind))

            plt.subplots_adjust(**DIM2_SUBPLOTS_ADJUST)

            # save figure
            save(fig, (figtmp % (dim, ind)) + '.%s', figtype, verbose=verbose, dpi=dpi)
            close(fig)
