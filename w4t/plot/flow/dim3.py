"""utils for plotting 3D data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

from w4t.plot.plot import plt

from .flow import hist as _hist
from .dim2 import _plot as _dim2_plot

#-------------------------------------------------

FIGSIZE = (5.0, 8.0)

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
    left=0.05,
    right=0.95,
    bottom=0.03,
    top=0.93,
    hspace=0.10,
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
 
    for ax, dim, xlabel, ylabel, transpose in [
            (ax11, 1, None, 'z', False),
            (ax12, 2, 'x', 'y', False),
            (ax22, 0, 'z', None, True),
        ]:
        d = np.mean(data, axis=dim) # average along one dimension
        if transpose: # make sure we have the correct orientation of axes (x-axis is index 0, y-axis is index 1)
            d = np.transpose(d)

        ax = _dim2_plot(ax, d)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    #---

    return ax

#---

def plot(approx, **kwargs):
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

        for col, label, data in enumerate(group):

            num = np.prod(data.shape)
            if num == 0: # no data
                continue

            ax = _plt(
                plt.subplot(8,6,row*6 + col + 1),
                plt.subplot(8,6,(row+1)*6 + col + 1),
                plt.subplot(8,6,(row+1)*6 + col + 2),
                data,
                **kwargs
            )

            ax.set_title(label)

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

'''

USE the following within "slice" and "grand_tour" logic?

def aaa_imshow(
        ha,
        figtmp,
        norms=None,
        cmap='Reds',
        vmin=-2.0,
        vmax=2.0,
        label='',
        thr=0.0,
        structures=False,
        grand_tour=False,
        orthographic=False,
    ):
    print('iterating over scales')

    compute_norms = norms is None
    if compute_norms:
        norms = []

    ha.idecompose() # reset the decomposition

    ind = 0
    while ha.active[0] > 1:
        scales = '-'.join('%03d'%_ for _ in ha.scales)

        print('processing: '+scales)

        title = label # + '\nscale : ' + scales

        n0, n1, n2 = ha.active

        array_dict = dict()

        array_dict['aaa'] = ha.array[:n0, :n1, :n2]

        # get normalization
        if compute_norms:
            norm = np.std(array_dict['aaa'])
            norms.append(norm)
        else:
            norm = norms[ind]

        # approximate the mid-plane
        plane = 0.5*(array_dict['aaa'][:,:,n2//2] + array_dict['aaa'][:,:,n2//2-1]) / norm

        # plot the mid-plane
        fig = plt.plt.figure(figsize=(5,5))
        ax = fig.add_axes([0.05, 0.01, 0.90, 0.90])

        ax.imshow(
            plane,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            origin='lower',
            extent=(0, 1, 0, 1),
        )

        # decorate
        fig.suptitle(title, fontsize=10)

        ax.set_xticks([]) # remove any ticks
        ax.set_yticks([])

        # save
        plt.save(fig, (figtmp%scales)+'.%s', ['png'], dpi=dpi, verbose=True)
        plt.close(fig)

        #---

        # plot a histogram of values within plane

        fig = plt.plt.figure()
        ax = fig.gca()

        _, bins, _ = ax.hist(np.abs(plane.flatten()), bins=max(100, int(0.5*(n0*n1)**0.5)), histtype='step', density=True)
        ax.hist(np.abs(plane.flatten()), bins=bins, histtype='step', density=True, cumulative=True)

        ylim = ax.get_ylim()
        ax.plot([thr]*2, ylim, color='k', alpha=0.5)
        ax.set_ylim(ylim)

        ax.set_xlabel('abs(pixel coefficient)')

        # save
        plt.save(fig, (figtmp%scales+'_hist')+'.%s', ['png'], dpi=dpi, verbose=True)
        plt.close(fig)

        #---

        if grand_tour: # plot every plane separately
            print('plotting grand tour')
            for axis, (n, x) in enumerate([(n0, 'x'), (n1, 'y'), (n2, 'z')]):
                for xnd in range(n):
                    plane = np.take(array_dict['aaa'], xnd, axis=axis) / norm

                    fig = plt.plt.figure(figsize=(5,5))
                    ax = fig.add_axes([0.05, 0.01, 0.90, 0.90])

                    ax.imshow(
                        plane,
                        cmap=cmap,
                        vmax=vmax,
                        vmin=vmin,
                        origin='lower',
                        extent=(0, 1, 0, 1),
                    )

                    # decorate
                    fig.suptitle(title + '\n$%s=%d$'%(x,xnd), fontsize=10)

                    ax.set_xticks([]) # remove any ticks
                    ax.set_yticks([])

                    # save
                    plt.save(fig, (figtmp%scales)+('-%s-%03d'%(x,xnd))+'.%s', ['png'], dpi=dpi, verbose=True)
                    plt.close(fig)

        #---

        del array_dict

        ha.dwt() # decompose
        ind += 1

    return norms
'''
