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
    ha.idecompose()

    raise NotImplementedError('''
def scalogram(fig, ax, AX, xs, ha1d, cmap=DEFAULT_CMAP):
    """make a scalogram and rough power spectrum
    """
    if ha1d.active[0] == 1: # ignore the lowest order
        ha1d.ihaar()

    X = []
    Y = []
    Z = []
    scales = []

    while ha1d.scales[0] > 1:
        scales.append(ha1d.scales[0])

        xs = np.arange(ha1d.active[0], dtype=float) / ha1d.active[0]
        xs += 0.5*(xs[1]-xs[0])

        v = np.var(ha1d.detail)

        # add to arrays for scatter points
        X.append(xs)
        Y.append(ha1d.scales[0]*np.ones(ha1d.active[0]))
        Z.append( np.array(ha1d.detail[:]) / v**0.5 ) # make a copy to avoid the fact that ha1d will edit this in-place
                                                      # also scale this by the std dev at that scale for visualization purposes

        # add to power spectrum
        # FIXME? make this a violin plot or something to show the full distribution
        AX.plot(v, ha1d.scales[0], marker='o', markeredgecolor='k', markerfacecolor='none')

        # work back up the decomposition levels
        ha1d.ihaar()

    # plot the scalogram as a scatter
    X = np.concatenate(tuple(X))
    Y = np.concatenate(tuple(Y))
    Z = np.concatenate(tuple(Z))
    vlim = np.max(np.abs(Z))

    # FIXME? change this to tiles

    cb = fig.colorbar(
        ax.scatter(
            X.flatten(),
            Y.flatten(),
            c=Z.flatten(),
#            alpha=0.25,
            vmin=-vlim,
            vmax=+vlim,
            s=Y.flatten(), # increase dot size to match scale
            marker='.',
            cmap=cmap,
        ),
        cmap=cmap,
        ax=ax,
        location='left',
        shrink=1.0,
    )

    cb.set_label('scaled detail coeff')

    ax.set_yscale('log')
    ax.set_yticks(scales)
    plt.setp(ax.get_yticklabels(), visible=False)

    ax.set_xlim(xmin=0, xmax=1)
    ax.set_xlabel('x')

    ax.tick_params(**scat_tick_params)

    #---

    AX.set_xscale('log')
    AX.set_xlabel('var(detail coeffs)')

    AX.set_yscale('log')
    AX.set_yticks(ax.get_yticks())
    AX.set_yticklabels(['%d'%_ for _ in ax.get_yticks()])
    AX.set_ylim(ax.get_ylim())

    AX.yaxis.tick_right()
    AX.yaxis.set_label_position('right')

    AX.set_ylabel('scale')

    AX.tick_params(**hist_tick_params)

    #---

    plt.subplots_adjust(**scalogram_subplots_adjust)
    ''')
