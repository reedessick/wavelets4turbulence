"""utils for plotting the Haar decomposition of 3D data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from .utils import *

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

#-------------------------------------------------

def scatter(array_dict, log=False, xmin=None, xmax=None, ymin=None, ymax=None):
    """plot 3D data from a Haar decomposed (assumed to be square)
    """
    fig = plt.figure(figsize=FIGSIZE)

    #---

    for group in GROUPS:

        for ind, nickname in group:
            data = array_dict[nickname]

            num = np.prod(data.shape)
            if num == 0: # no data
                continue

            ax = fig.add_subplot(4,3,ind, projection='3d')

            array = array_dict[nickname]

            xs = np.arange(len(array)) / len(array) # NOTE this could be fragile
            xs += (xs[1]-xs[0])/2
            xs, ys, zs = np.meshgrid(xs, xs, xs, indexing='ij')

            if log:
                array = np.log10(array)

                # plot pos data
                cs = data / np.max(array)
                sel = data >= 0

                ax.scatter(
                    xs.flatten()[sel],
                    ys.flatten()[sel],
                    zs.flatten()[sel],
                    c=cs.flatten()[sel],
                    alpha=cs.flatten()[sel],
                    vmin=-1,
                    vmax=+1,
                    s=1.0, # small dots
                    marker='.',
                    cmap=LOG_POS_CMAP,
                )

                # plot neg data
                cs = data / np.max(-array)
                sel = data < 0

                ax.scatter(
                    xs.flatten()[sel],
                    ys.flatten()[sel],
                    zs.flatten()[sel],
                    c=cs.flatten()[sel],
                    alpha=cs.flatten()[sel],
                    vmin=-1,
                    vmax=+1,
                    s=1.0, # small dots
                    marker='.',
                    cmap=LOG_NEG_CMAP,
                )

            else:
                cs = array / np.max(np.abs(array))

                ax.scatter(
                    xs.flatten(),
                    ys.flatten(),
                    zs.flatten(),
                    c=cs.flatten(),
                    alpha=(1+cs.flatten())/2,
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
