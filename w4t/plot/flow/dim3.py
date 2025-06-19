"utils for plotting 3D data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from w4t.plot..plot import *

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

'''

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

'''

#-------------------------------------------------

'''

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

        if orthographic and (ha.ndim==3):
            print('plotting orthographic views')
            levels = copy.copy(ha.levels) # make a copy so we can reset

            fig = plt.plt.figure(figsize=(10,10))

            for ax, axis, xlabel, ylabel in [
                    (plt.plt.subplot(2,2,1), 1, None, 'z'),
                    (plt.plt.subplot(2,2,3), 2, 'y', 'x'),
                    (plt.plt.subplot(2,2,4), 0, 'z', None),
                ]:
                ha.decompose(axis=axis) # collapse along one axis
                a = np.take(ha.approx, 0, axis=axis)
                vlim = np.max(np.abs(a))

                ax.imshow(
                    a,
                    cmap=cmap,
                    vmax=+vlim,
                    vmin=-vlim,
                    origin='lower',
                    extent=(0, 1, 0, 1),
                )

                ax.set_xticks([]) # remove any ticks
                ax.set_yticks([])

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

                ha.set_levels(levels)

            # decorate
            fig.suptitle(title, fontsize=10)

            plt.plt.subplots_adjust(
                left=0.05,
                right=0.95,
                bottom=0.05,
                top=0.95,
                hspace=0.01,
                wspace=0.01,
            )

            # save
            plt.save(fig, (figtmp%scales)+'-ortho'+'.%s', ['png'], dpi=dpi, verbose=True)
            plt.close(fig)

        #---

        if structures: # identify structures and make separate plots for each
#            print('identifying clusters in 2D midplane')
#            sel = np.abs(plane) > thr # which pixels we consider in 2D

            print('identifying clusters in full 3D box')
            sel = np.abs(array_dict['aaa']/norm) > thr # which pixels we consider in 3D

            clusters = ha.structures(sel, num_proc=6)
            clusters.sort(key=lambda x: -len(x)) # biggest first

            num_clusters = len(clusters)
            print('    found %d clusters' % num_clusters)

            ect = 0.5*(sel[:,:,n2//2].astype(int) + sel[:,:,n2//2-1].astype(int)) # just the midplane

            for cnd, cluster in enumerate(clusters):
                num_pix = len(cluster)
                sys.stdout.write('\r    cluster : %d / %d (len = %d / %d)' % (cnd, num_clusters, num_pix, n0*n1*n2))
                sys.stdout.flush()

#                a = np.zeros((n0, n1), dtype=float) # 2D representation
                a = np.zeros((n0, n1, n2), dtype=float) # 3D representation
                a[:] = np.nan

                cluster = tuple(np.transpose(cluster)) # make this so an array can understand it
#                a[cluster] = plane[cluster] # 2D
                a[cluster] = array_dict['aaa'][cluster] / norm # 3D

                plane = np.where( # deal with nans in neighboring planes
                    np.isnan(a[:,:,n2//2]),
                    np.where(
                        np.isnan(a[:,:,n2//2-1]),
                        np.nan,
                        a[:,:,n2//2-1],
                    ),
                    np.where(
                        np.isnan(a[:,:,n2//2-1]),
                        a[:,:,n2//2],
                        0.5*(a[:,:,n2//2] + a[:,:,n2//2-1]),
                    ),
                )

                if np.all(np.isnan(plane)):
                    sys.stdout.write('\r    nothing to show!')

                else:
                    sys.stdout.write('\r    plotting!\n')

                    fig = plt.plt.figure(figsize=(5,5))
                    ax = fig.add_axes([0.05, 0.01, 0.90, 0.90])

                    # plot the overall set of pixels selected
                    ax.imshow(
                        ect,
                        cmap='Greys',
                        vmax=1,
                        vmin=0,
                        origin='lower',
                        extent=(0, 1, 0, 1),
                    )

                    # plot only this cluster
                    ax.imshow(
                        plane,
                        cmap=cmap,
                        vmax=vmax,
                        vmin=vmin,
                        origin='lower',
                        extent=(0, 1, 0, 1),
                    )

                    # decorate
                    fig.suptitle('cluster %d (%d pixels)' % (cnd, num_pix), fontsize=10)

                    ax.set_xticks([]) # remove any ticks
                    ax.set_yticks([])

                    # save
                    plt.save(fig, (figtmp%scales + '_cluster-%03d'%cnd)+'.%s', ['png'], dpi=dpi, verbose=True)
                    plt.close(fig)

                #---

                if orthographic and (ha.ndim==3) and (num_pix > 100):
                    print('plotting orthographic views')

                    _ha = pywt.PyWaveletArray(np.where(np.isnan(a), 0.0, a), wavelet)
                    levels = copy.copy(_ha.levels) # make a copy so we can reset

                    fig = plt.plt.figure(figsize=(10,10))

                    for ax_ind, axis, xlabel, ylabel, trans in [
                            (1, 1, 'x', 'z', True),
                            (3, 2, 'x', 'y', True),
                            (4, 0, 'z', 'y', False),
                        ]:
                        ax = plt.plt.subplot(2,2,ax_ind)
                        _ha.decompose(axis=axis) # collapse along one axis

                        b = np.take(_ha.approx, 0, axis=axis)
                        if trans:
                            b = np.transpose(b)
                        vlim = np.max(np.abs(b))

                        ax.imshow(
                            b,
                            cmap=cmap,
                            vmax=+vlim,
                            vmin=-vlim,
                            origin='lower',
                            extent=(0, 1, 0, 1),
                        )

                        ax.set_xticks([]) # remove any ticks
                        ax.set_yticks([])

                        ax.set_xlabel(xlabel)
                        if ax_ind <= 2:
                            ax.xaxis.set_label_position('top')

                        ax.set_ylabel(ylabel)
                        if (ax_ind % 2) == 0:
                            ax.yaxis.set_label_position('right')

                        _ha.set_levels(levels)

                    # decorate
                    fig.suptitle(title, fontsize=10)

                    plt.plt.subplots_adjust(
                        left=0.05,
                        right=0.95,
                        bottom=0.05,
                        top=0.95,
                        hspace=0.01,
                        wspace=0.01,
                    )

                    # save
                    plt.save(fig, (figtmp%scales)+('-ortho_cluster-%03d'%cnd)+'.%s', ['png'], dpi=dpi, verbose=True)
                    plt.close(fig)

                #---

                if grand_tour and (num_pix > 100): # FIXME? arbitrary cut-off for how big the cluster has to be to get a grand tour...
                    sys.stdout.write('\r    plotting grand tour!\n')

                    for axis, (n, x) in enumerate([(n0, 'x'), (n1, 'y'), (n2, 'z')]):
                        for xnd in range(n):
                            fig = plt.plt.figure(figsize=(5,5))
                            ax = fig.add_axes([0.05, 0.01, 0.90, 0.90])

                            # plot the overall set of pixels selected
                            ax.imshow(
                                np.take(sel, xnd, axis=axis),
                                cmap='Greys',
                                vmax=1,
                                vmin=0,
                                origin='lower',
                                extent=(0, 1, 0, 1),
                            )

                            # plot only this cluster
                            ax.imshow(
                                np.take(a, xnd, axis=axis),
                                cmap=cmap,
                                vmax=vmax,
                                vmin=vmin,
                                origin='lower',
                                extent=(0, 1, 0, 1),
                            )

                            # decorate
                            fig.suptitle('cluster %d (%d pixels)\n$%s=%d$' % (cnd, num_pix, x, xnd), fontsize=10)

                            ax.set_xticks([]) # remove any ticks
                            ax.set_yticks([])

                            # save
                            plt.save(
                                fig,
                                (figtmp%scales + '_cluster-%03d-%s-%03d'%(cnd,x,xnd))+'.%s',
                                ['png'],
                                dpi=dpi,
                                verbose=True,
                            )
                            plt.close(fig)

            sys.stdout.write('\n')
            sys.stdout.flush()

        #---

        del array_dict

        ha.dwt() # decompose
        ind += 1

    return norms
'''
