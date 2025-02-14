"""utils for plotting the Haar decomposition of planar (2D) data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from .utils import *

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

HIST_TICK_PARAMS = dict(
    left=True,
    right=True,
    top=True,
    bottom=True,
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

APPROX_CMAP = 'RdGy'
DETAIL_CMAP = 'PuOr'

#-------------------------------------------------

def imshow(aa, ad, da, dd):
    """plot 2D data from a Haar decomposed 2D array (assumed to be square)
    """
    fig = plt.figure(figsize=FIGSIZE)

    #---

    for ind, (data, cmap) in enumerate([
            (aa, APPROX_CMAP),
            (ad, DETAIL_CMAP),
            (da, DETAIL_CMAP),
            (dd, DETAIL_CMAP),
        ]):

        if np.prod(data.shape) == 0: # no data
            continue

        ax = plt.subplot(2,2,ind+1)

        if ind > 0: # make scales symmetric for details
            vlim = np.max(np.abs(data))
            vmin = -vlim
            vmax = +vlim
        else:
            vmin = vmax = None

        ax.imshow(
            data,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            origin='lower',
            extent=(0, 1, 0, 1),
        )

        ax.set_xlim(xmin=0, xmax=1)
        ax.set_ylim(ymin=0, ymax=1)

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

        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

        ax.tick_params(**IMSHOW_TICK_PARAMS)

    #---

    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig

#------------------------

def hist(aa, ad, da, dd, grid=False, **kwargs):
    """plot histograms of coefficients from a Haar decomposed 2D array (assumed to be square)
    """
    fig = plt.figure(figsize=FIGSIZE)

    #---

    for ind, data in enumerate([aa, ad, da, dd]):

        if np.prod(data.shape) == 0: # no data
            continue

        ax = plt.subplot(2,2,ind+1)

        ax.hist(
            np.ravel(data),
            bins=min(1000, max(10, int(np.prod(data.shape)**0.5))),
            **kwargs,
        )

        if ind == 0:
            ax.xaxis.tick_top()

        else:
            xlim = np.max(np.abs(ax.get_xlim()))
            ax.set_xlim(xmin=-xlim, xmax=+xlim)

            if ind == 1:
                ax.xaxis.tick_top()
                ax.yaxis.tick_right()

            elif ind == 3:
                ax.yaxis.tick_right()

        ax.tick_params(**HIST_TICK_PARAMS)
        ax.grid(grid, which='both')

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ax.text(xmin + 0.01*(xmax-xmin), ymax / (ymax/ymin)**0.01, '%d samples' % np.prod(data.shape), ha='left', va='top')

    #---

    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig
