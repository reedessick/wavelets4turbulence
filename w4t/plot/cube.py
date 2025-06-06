"""utils for plotting the Haar decomposition of 3D data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from .plot import *

#-------------------------------------------------

FIGSIZE = (5.0, 8.0)

#---

HIST_TICK_PARAMS = dict(
    left=True,
    right=True,
    top=True,
    bottom=True,
    direction='in',
    which='both',
)

SCATTER_TICK_PARAMS = dict(
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

GROUPS = [
    [(2, 'aaa')],
    [(4, 'daa'), (5, 'ada'), (6, 'aad')],
    [(7, 'add'), (8, 'dad'), (9, 'dda')],
    [(11, 'ddd')],
]

#---

CMAP = 'RdGy'

LOG_POS_CMAP = 'YlOrRd'
LOG_NEG_CMAP = 'YlGnBu'

#---

DEFAULT_THR = 0.25 # used to select what actually gets plotted in a scatter plot

#-------------------------------------------------

def show(array, title=None, thr=DEFAULT_THR):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')

    xs = np.arange(len(array)) / len(array) # NOTE this could be fragile
    xs += (xs[1]-xs[0])/2
    xs, ys, zs = np.meshgrid(xs, xs, xs, indexing='ij')

    cs = array / np.max(np.abs(array[array==array]))

    sel = np.abs(cs).flatten() > thr

    ax.scatter(
        xs.flatten()[sel],
        ys.flatten()[sel],
        zs.flatten()[sel],
        c=cs.flatten()[sel],
        alpha=(1+cs.flatten()[sel])/2,
        vmin=-1,
        vmax=+1,
        s=1.0, # small dots
        marker='.',
        cmap=CMAP,
    )

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    plt.show()
    plt.close(fig)

#------------------------

def scatter(array_dict, log=False, thr=DEFAULT_THR):
    """plot 3D data from a Haar decomposed (assumed to be square)
    """
    fig = plt.figure(figsize=FIGSIZE)

    #---

    for group in GROUPS:

        for ind, nickname in group:
            array = array_dict[nickname]

            num = np.prod(array.shape)
            if num == 0: # no data
                continue

            ax = fig.add_subplot(4,3,ind, projection='3d')

            xs = np.arange(len(array)) / len(array) # NOTE this could be fragile
            xs += (xs[1]-xs[0])/2
            xs, ys, zs = np.meshgrid(xs, xs, xs, indexing='ij')

            if log:
                data = np.log10(np.abs(array))

                # plot pos data
                cs = data / np.max(data[data==data])
                sel = (array.flatten() > 0) * (data.flatten() == data.flatten()) * (cs.flatten() > thr) # avoid nans

                ax.scatter(
                    xs.flatten()[sel],
                    ys.flatten()[sel],
                    zs.flatten()[sel],
                    c=cs.flatten()[sel],
                    alpha=0.25,
#                    vmax=+1,
                    s=1.0, # small dots
                    marker='.',
                    cmap=LOG_POS_CMAP,
                )

                # plot neg data
                cs = data / np.max(-data[data==data])
                sel = (array.flatten() < 0) * (data.flatten() == data.flatten()) * (cs.flatten() > thr) # avoid nans

                ax.scatter(
                    xs.flatten()[sel],
                    ys.flatten()[sel],
                    zs.flatten()[sel],
                    c=cs.flatten()[sel],
                    alpha=0.25,
#                    vmax=+1,
                    s=1.0, # small dots
                    marker='.',
                    cmap=LOG_NEG_CMAP,
                )

            else:
                cs = array / np.max(np.abs(array[array==array]))

                sel = np.abs(cs).flatten() > thr

                ax.scatter(
                    xs.flatten()[sel],
                    ys.flatten()[sel],
                    zs.flatten()[sel],
                    c=cs.flatten()[sel],
                    alpha=(1+cs.flatten()[sel])/2,
                    vmin=-1,
                    vmax=+1,
                    s=1.0, # small dots
                    marker='.',
                    cmap=CMAP,
                )
      
            ax.set_xlim(xmin=0, xmax=1)
            ax.set_ylim(ymin=0, ymax=1)
            ax.set_zlim(zmin=0, zmax=1)

            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.get_zticklabels(), visible=False)

            ax.tick_params(**SCATTER_TICK_PARAMS)

    #---

    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    return fig

#------------------------

def hist(array_dict, grid=False, **kwargs):
    """plot histograms of coefficients from a Haar decomposed 2D array (assumed to be square)
    """
    fig = plt.figure(figsize=FIGSIZE)

    #---

    for group in GROUPS:

        _ymin = +np.inf
        _ymax = -np.inf
        text = []

        for ind, nickname in group:
            data = array_dict[nickname]

            num = np.prod(data.shape)
            if num == 0: # no data
                continue

            ax = plt.subplot(4,3,ind)

            data = np.ravel(data)

            if nickname == 'aaa':
                xmin = np.min(data)
                xmax = np.max(data)
            else:
                xlim = np.max(np.abs(data))
                xmin = -xlim
                xmax = +xlim

            bins = np.linspace(xmin, xmax, min(1000, max(10, int(num**0.5))))

            ax.hist(data, bins=bins, **kwargs)

            ax.set_xlim(xmin=xmin, xmax=xmax)

            ax.tick_params(**HIST_TICK_PARAMS)
            ax.grid(grid, which='both')

            ymin, ymax = ax.get_ylim()

            _ymin = min(_ymin, ymin)
            _ymax = max(_ymax, ymax)

            text.append((ax, xmin + 0.01*(xmax-xmin), '%d samples' % num, 'left', 'top'))
            text.append((ax, xmax - 0.01*(xmax-xmin), nickname, 'right', 'top'))

        y = ymax / (ymax/ymin)**0.01
        for ax, x, text, ha, va in text:
            ax.set_ylim(_ymin, _ymax)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.text(x, y, text, ha=ha, va=va)

    #---

    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig
