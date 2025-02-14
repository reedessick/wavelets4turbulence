"""utils for plotting the Haar decomposition of planar (2D) data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from .utils import *

#-------------------------------------------------

FIGSIZE = (5.0, 3.0)

#---

TICK_PARAMS = dict(
    left=True,
    right=True,
    top=True,
    bottom=True,
    direction='in',
    which='both',
)

#---

SUBPLOTS_ADJUST = dict(
    left=0.12,
    right=0.88,
    bottom=0.15,
    top=0.875,
    hspace=0.03,
    wspace=0.03,
)

#---

SCALOGRAM_CMAP = 'RdBu'

#-------------------------------------------------

def plot(a, d, **kwargs):
    """plot 1D data from a Haar decomposed 2D array (assumed to be square)
    """
    fig = plt.figure(figsize=FIGSIZE)

    #---

    for ind, data in enumerate([a, d]):

        if np.prod(data.shape) == 0: # no data
            continue

        ax = plt.subplot(1,2,ind+1)

        dx = 0.5/len(data)
        ax.plot(dx + np.arange(len(data))/len(data), data, **kwargs)
        ax.set_xlim(xmin=0, xmax=1)
        ax.set_xticks(ax.get_xticks()[1:-1])

        if ind == 0:
            ax.set_ylabel('approximant')

        elif ind == 1:
            ax.set_ylabel('detail')
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

            ylim = np.max(np.abs(ax.get_ylim()))
            ax.set_ylim(ymin=-ylim, ymax=+ylim)

        ax.tick_params(**TICK_PARAMS)

    #---

    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig

#------------------------

def hist(a, d, grid=False, **kwargs):
    """plot histograms of coefficients from a Haar decomposed 2D array (assumed to be square)
    """
    fig = plt.figure(figsize=FIGSIZE)

    #---

    for ind, data in enumerate([a, d]):

        if np.prod(data.shape) == 0: # no data
            continue

        ax = plt.subplot(1,2,ind+1)

        ax.hist(
            np.ravel(data),
            bins=min(1000, max(10, int(np.prod(data.shape)**0.5))),
            **kwargs,
        )

        if ind == 0:
            ax.set_xlabel('approximant')

        else:
            ax.set_xlabel('detail')
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

            xlim = np.max(np.abs(ax.get_xlim()))
            ax.set_xlim(xmin=-xlim, xmax=+xlim)

        ax.tick_params(**TICK_PARAMS)
        ax.grid(grid, which='both')

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ax.text(xmin + 0.01*(xmax-xmin), ymax / (ymax/ymin)**0.01, '%d samples' % np.prod(data.shape), ha='left', va='top')

    #---

    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig

#-------------------------------------------------

def scalogram(ha):
    """plot a scalogram of 1D Haar decomposition
    """
    fig = plt.figure()

    ax1 = fig.add_axes([0.10, 0.10, 0.80, 0.70]) # scalogram
    ax3 = fig.add_axes([0.10, 0.81, 0.80, 0.14]) # raw data
    ax2 = fig.add_axes([0.91, 0.10, 0.01, 0.70]) # colorbar

    #---

    ha.decompose()

    if ha.active[0] == 1: # ignore the lowest order
        ha.ihaar()

    X = []
    Y = []
    Z = []
    scales = []

    while ha.scales[0] > 1:
        scales.append(ha.scales[0])

        xs = np.arange(ha.active[0], dtype=float) / ha.active[0]
        xs += 0.5*(xs[1]-xs[0])

        # add to arrays for scatter points
        X.append(xs)
        Y.append(ha.scales[0]*np.ones(ha.active[0]))

        detail = np.array(ha.detail[:]) # make a copy to avoid the fact that ha will edit this in-place
        s = np.std(detail)
        if s > 0:
            detail /= s # only scale this if there is some variation
        Z.append( detail )

        # iterate
        ha.ihaar()

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
    ax1.set_ylim(ymin=np.min(scales)/2**0.5, ymax=np.max(scales)*2**0.5)

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

    ax3.plot(np.arange(len(ha.array))/len(ha.array), ha.array, )

    ax3.set_xlim(ax1.get_xlim())
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax3.tick_params(**TICK_PARAMS)

    #---

    return fig
