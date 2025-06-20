"""utils for plotting 3D data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from w4t.plot.plot import (plt, save, close)

from .flow import hist as _hist

from .dim2 import _plot as _dim2_plot
from .dim2 import FIGSIZE as DIM2_FIGSIZE
from .dim2 import IMSHOW_TICK_PARAMS as DIM2_TICKPARAMS
from .dim2 import SUBPLOTS_ADJUST as DIM2_SUBPLOTS_ADJUST

#-------------------------------------------------

FIGSIZE = (5.0, 5.0)
BIG_FIGSIZE = (10.0, 12.0)

#---

SUBPLOTS_ADJUST = dict(
    left=0.10,
    right=0.90,
    bottom=0.10,
    top=0.90,
    hspace=0.03,
    wspace=0.03,
)

BIG_SUBPLOTS_ADJUST = dict(
    left=0.05,
    right=0.95,
    bottom=0.05,
    top=0.95,
    hspace=0.10,
    wspace=0.10,
)

#---

CMAP = 'RdGy'

LOG_POS_CMAP = 'YlOrRd'
LOG_NEG_CMAP = 'YlGnBu'

#-------------------------------------------------

def _plot(ax11, ax12, ax22, data, extent=[(0, 1)]*3, grid=False, labels=True, **kwargs):
    """plot a visualization of the flow
    """
    assert len(np.shape(data)) == 3, 'data must be 3-dimensional'
 
    for ind, (ax, dim, xlabel, ylabel, extent, transpose) in enumerate([
            (ax11, 1, 'x', 'z', (extent[0][0], extent[0][1], extent[2][0], extent[2][1]), False),
            (ax12, 2, 'x', 'y', (extent[0][0], extent[0][1], extent[1][0], extent[1][1]), False),
            (ax22, 0, 'z', 'y', (extent[2][0], extent[2][1], extent[1][0], extent[1][1]), True),
        ]):
        d = np.mean(data, axis=dim) # average along one dimension
        if transpose: # make sure we have the correct orientation of axes (x-axis is index 0, y-axis is index 1)
            d = np.transpose(d)

        ax = _dim2_plot(ax, d, extent=extent, **kwargs)

        if ind == 0:
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')

        elif ind == 2:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

        if labels:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)

        ax.tick_params(**DIM2_TICKPARAMS)

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
        fig.text(0.70, 0.70, title, ha='center', va='center')

    return fig

#-----------

def plot_coeff(aaa, aad, ada, daa, add, dad, dda, ddd, title=None, **kwargs):
    """plot visualization of wavelet coefficients
    """
    fig = plt.figure(figsize=BIG_FIGSIZE)

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
                plt.subplot(8, 6, 2*row*6 + 2*col + 1),
                plt.subplot(8, 6, (2*row+1)*6 + 2*col + 1),
                plt.subplot(8, 6, (2*row+1)*6 + 2*col + 2),
                data,
                labels=False,
                **kwargs
            )

#            ax11.set_title(label)
            fig.text(
                BIG_SUBPLOTS_ADJUST['left'] + (col+0.75)*(BIG_SUBPLOTS_ADJUST['right']-BIG_SUBPLOTS_ADJUST['left'])/3,
                BIG_SUBPLOTS_ADJUST['top'] - (row+0.25)*(BIG_SUBPLOTS_ADJUST['top']-BIG_SUBPLOTS_ADJUST['bottom'])/4,
                label,
                ha='center',
                va='center',
            )

    #---

    if title:
        fig.suptitle(title)

    #---

    plt.subplots_adjust(**BIG_SUBPLOTS_ADJUST)

    #---

    return fig

#------------------------

def hist(approx, title=None, num_samples=True, **kwargs):
    """histogram approx
    """
    fig = plt.figure(figsize=FIGSIZE)
    _hist(plt.subplot(1,1,1), approx, xlabel=title, num_samples=num_samples, **kwargs)
    plt.subplots_adjust(**SUBPLOTS_ADJUST)
    return fig

#-----------

def hist_coeff(aaa, aad, ada, daa, add, dad, dda, ddd, title=None, num_samples=True, **kwargs):
    """histogram wavelet coefficients
    """
    fig = plt.figure(figsize=BIG_FIGSIZE)

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

            ax = _hist(plt.subplot(4,3,row*3+col+1), data, symmetric_xlim=(row!=0), num_samples=num_samples, **kwargs)

            if col == 1:
                ax.set_ylabel('')
                plt.setp(ax.get_yticklabels(), visible=False)

            elif col == 2:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position('right')

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            _ymin = min(_ymin, ymin)
            _ymax = max(_ymax, ymax)

            text.append((ax, xmin + 0.02*(xmax-xmin), label, 'left', 'top'))

        y = ymax / (ymax/ymin)**0.02
        for ax, x, text, ha, va in text:
            ax.set_ylim(_ymin, _ymax)
            ax.text(x, y, text, ha=ha, va=va)

    #---

    if title:
        fig.suptitle(title)

    #---

    plt.subplots_adjust(**BIG_SUBPLOTS_ADJUST)

    #---

    return fig

#-------------------------------------------------

def grand_tour(
        array,
        extent=[(0,1)]*3,
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
    assert len(shape) == 3, 'bad number of dimensions!'

    figtmp = figtmp + '-dim%d-ind%06d'

    for dim in range(3): # iterate over each dimension, making overlaid 1D plot for each

        if dim == 0:
            xlabel = 'y'
            ylabel = 'z'
            dlabel = 'x'
            _extent = (extent[1][0], extent[1][1], extent[2][0], extent[2][1])

        elif dim == 1:
            xlabel = 'x'
            ylabel = 'z'
            dlabel = 'y'
            _extent = (extent[0][0], extent[0][1], extent[2][0], extent[2][1])

        elif dim == 2:
            xlabel = 'x'
            ylabel = 'y'
            dlabel = 'z'
            _extent = (extent[0][0], extent[0][1], extent[1][0], extent[1][1])

        for ind in range(0, shape[dim], increment): # iterate over slices
            if verbose:
                print('dim = %d    ind = %d/%d' % (dim, ind, shape[dim]))

            fig = plt.figure(figsize=DIM2_FIGSIZE)
            ax = plt.subplot(1,1,1)

            ax = _dim2_plot(
                ax,
                np.take(array, ind, axis=dim), # should be a 2D array
                xlabel=xlabel,
                ylabel=ylabel,
                extent=_extent,
                **kwargs
            )

            label = '%s=%06d' % (dlabel, ind)
            if title:
                label = '%s\n%s' % (title, label)
            ax.set_title(label)

            plt.subplots_adjust(**DIM2_SUBPLOTS_ADJUST)

            # save figure
            save(fig, (figtmp % (dim, ind)) + '.%s', figtype, verbose=verbose, dpi=dpi)
            close(fig)
