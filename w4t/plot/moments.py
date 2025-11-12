"""a module for plotting logic for moments
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from .plot import *

from w4t.utils.infer import (structure_function_ansatz, logarithmic_derivative_ansatz)

#-------------------------------------------------

TICK_PARAMS = dict(
    left=True,
    right=True,
    top=True,
    bottom=True,
    direction='in',
    which='both',
)

SUBPLOTS_ADJUST = dict(
    left=0.12,
    right=0.95,
    bottom=0.10,
    top=0.92,
    hspace=0.03,
)

#------------------------

DEFAULT_NUM_STD = 1.0

DEFAULT_LINESTYLE = 'none'
DEFAULT_MARKER = 'o'

#-------------------------------------------------

def moments(
        scales,
        indexes,
        mom,
        cov,
        label='',
        linestyle=DEFAULT_LINESTYLE,
        marker=DEFAULT_MARKER,
#        poly=None,
        num_std=DEFAULT_NUM_STD,
        rescale=False,
        normalize=None,
        verbose=False,
        ncols=None,
        legend=False,
        grid=True,
        alpha=0.75,
        fig=None,
    ):
    """plot moments, including polynomial fits if supplied.
    if rescale: plot mom**(1./index) instead of just mom
    """
    if fig is None:
        fig = plt.figure()

    ax = fig.gca()

    #---

#    if poly is not None:
#        poly, polybins = poly

    for ind, index in enumerate(indexes):

        exp = 1./index if rescale else 1

        color = 'C%d' % ind

        if normalize is not None:
            norm = np.interp(normalize, scales, mom[:,ind])
        else:
            norm = 1.0

        ax.plot(
            scales,
            (mom[:,ind]/norm)**exp,
            marker=marker,
            markerfacecolor='none',
            linestyle=linestyle,
            color=color,
            alpha=alpha,
            label='%s $p=%d$'%(label, index),
        )

        for snd, scale in enumerate(scales):
            m = mom[snd,ind]/norm
            s = cov[snd,ind,ind]
            if s > 0: # only plot sensible error estimates
                s = s**0.5 * num_std / norm
                ax.plot([scale]*2, np.array([m-s, m+s])**exp, color=color, alpha=alpha)
            elif verbose:
                print('        WARNING! skipping error estimate for index=%d at scale=%d with var=%.3e' % (index, scale, s))

#        if poly is not None: ### add the fit
#            for bnd, (m, M) in enumerate(polybins[ind]):
#                x = np.linspace(np.log(m), np.log(M), 101)
#                y = 0.
#                for order, p in enumerate(poly[ind][bnd][::-1]):
#                    y += p * x**order
#
#                ax.plot(
#                    np.exp(x),
#                    np.exp(y * exp) / scale,
#                    color=color,
#                    alpha=alpha,
#                )

    #---

    ax.set_xlabel('scale')
    ax.set_xscale('log')
    ax.set_xlim(xmin=scales[-1]*1.1, xmax=scales[0]/1.1)
    ax.set_xticks([], minor=True)
    ax.set_xticks(scales)
    ax.set_xticklabels(['%d'%_ for _ in ax.get_xticks()])

    if np.any(mom > 0): # only plot log-scale if there are non-zero values
        ax.set_yscale('log')

    if rescale:
        ax.set_ylabel('$\left<\left|d_{x,s}\\right|^p\\right>^{1/p}_x$')
    else:
        ax.set_ylabel('$\left<\left|d_{x,s}\\right|^p\\right>_x$')

    #---

    if legend:
        ax.legend(loc='best', ncol=ncols)

    if grid:
        ax.grid(True, which='both')

    ax.tick_params(**TICK_PARAMS)
    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig

#-------------------------------------------------

def scaled_moments(
        scales,
        indexes,
        mom,
        cov,
        label='',
        linestyle=DEFAULT_LINESTYLE,
        marker=DEFAULT_MARKER,
        num_std=DEFAULT_NUM_STD,
        rescale=False,
        verbose=False,
        ncols=None,
        legend=False,
        grid=True,
        alpha=0.75,
        fig=None,
    ):
    """plot moments scaled by the standard deviation
    """
    assert 2 in list(indexes), 'cannot scale moments if 2nd moment is not present!'

    ind2 = list(indexes).index(2)
    std = mom[:,ind2]**0.5

    #---

    if fig is None:
        fig = plt.figure()

    ax = fig.gca()

    #---

    for ind, index in enumerate(indexes):

        exp = 1./index if rescale else 1

        color = 'C%d' % ind

        ax.plot(
            scales,
            (mom[:,ind] / std**index)**exp,
            marker=marker,
            markerfacecolor='none',
            linestyle=linestyle,
            color=color,
            alpha=alpha,
            label='%s $p=%d$'%(label, index),
        )

        for snd, scale in enumerate(scales):
            m = mom[snd,ind] / std[snd]**index
            s = (-(index/2)*mom[snd,ind]/std[snd]**(index+2))**2 * cov[snd,ind2,ind2] \
                + (1./std[snd]**index)**2 * cov[snd,ind,ind] \
                + (1./std[snd]**index)*(-(index/2)*mom[snd,ind]/std[snd]**(index+2)) * cov[snd,ind2,ind]
            if s > 0: # only plot sensible error estimates
                s = s**0.5 * num_std
                ax.plot([scale]*2, np.array([m-s, m+s])**exp, color=color, alpha=alpha)
            elif verbose:
                print('        WARNING! skipping error estimate for index=%d at scale=%d with var=%.3e' % (index, scale, s))

    #---

    ax.set_xlabel('scale')
    ax.set_xscale('log')
    ax.set_xlim(xmin=scales[-1]*1.1, xmax=scales[0]/1.1)
    ax.set_xticks([], minor=True)
    ax.set_xticks(scales)
    ax.set_xticklabels(['%d'%_ for _ in ax.get_xticks()])

    if np.any(mom > 0):
        ax.set_yscale('log')

    if rescale:
        ax.set_ylabel('$\left<\left|d_{x,s}\\right|^p\\right>^{1/p}_x \left/ \left<d_{x,s}^2\\right>_x^{1/2} \\right.$')
    else:
        ax.set_ylabel('$\left<\left|d_{x,s}\\right|^p\\right>_x \left/ \left<d_{x,s}^2\\right>_x^{p/2} \\right.$')

    #---

    if legend:
        ax.legend(loc='best', ncol=ncols)

    if grid:
        ax.grid(True, which='both')

    ax.tick_params(**TICK_PARAMS)
    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig

#-------------------------------------------------

def extended_intermittency(
        scales,
        indexes,
        mom,
        cov,
        label='',
        linestyle=DEFAULT_LINESTYLE,
        marker=DEFAULT_MARKER,
        num_std=DEFAULT_NUM_STD,
        rescale=False,
        verbose=False,
        ncols=None,
        legend=False,
        grid=True,
        alpha=0.75,
        fig=None,
    ):
    """plot extended intermittency diagram
    """
    assert 2 in list(indexes), 'cannot scale moments if 2nd moment is not present!'

    ind2 = list(indexes).index(2)
    std = mom[:,ind2]**0.5

    #---

    if fig is None:
        fig = plt.figure()

    ax = fig.gca()

    #---

    exp2 = 0.5 if rescale else 1

    for ind, index in enumerate(indexes):

        exp = 1./index if rescale else 1

        color = 'C%d' % ind
        ax.plot(
            mom[:,ind2]**exp2,
            mom[:,ind]**exp,
            marker=marker,
            markerfacecolor='none',
            linestyle=linestyle,
            color=color,
            alpha=alpha,
            label='%s $p=%d$'%(label, index),
        )

        for snd, scale in enumerate(scales):
            # plot errors for x-axis
            m = mom[snd,ind2]
            s = cov[snd,ind2,ind2]
            if s > 0: # only plot sensible error estimates
                s = s**0.5 * num_std
                ax.plot(np.array([m-s, m+s])**exp2, np.array([mom[snd,ind]]*2)**exp, color=color, alpha=alpha)
            elif verbose:
                print('        WARNING! skipping error estimate for index=2 at scale=%d with var=%.3e' % (scale, s))

            # plot errors for y-axis
            m = mom[snd,ind]
            s = cov[snd,ind,ind]
            if s > 0: # only plot sensible error estimates
                s = s**0.5 * num_std
                ax.plot(np.array([mom[snd,ind2]]*2)**exp2, np.array([m-s, m+s])**exp, color=color, alpha=alpha)
            elif verbose:
                print('        WARNING! skipping error estimate for index=2 at scale=%d with var=%.3e' % (scale, s))

    #---

    if np.any(mom > 0):
        ax.set_xscale('log')
        ax.set_yscale('log')

    if rescale:
        ax.set_xlabel('$\left<d_{x,s}^2\\right>^{1/2}_x$')
        ax.set_ylabel('$\left<\left|d_{x,s}\\right|^p\\right>^{1/p}_x$')
    else:
        ax.set_xlabel('$\left<d_{x,s}^2\\right>_x$')
        ax.set_ylabel('$\left<\left|d_{x,s}\\right|^p\\right>_x$')

    #---

    if legend:
        ax.legend(loc='best', ncol=ncols)

    if grid:
        ax.grid(True, which='both')

    ax.tick_params(**TICK_PARAMS)
    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig

#-------------------------------------------------

def structure_function_ansatz_samples(
        scales,
        indexes,
        mom,
        cov,
        samples,
        alpha=0.75,
        legend=False,
        grid=True,
        title=None,
        verbose=False,
    ):
    """make a simple plot of structure function ansatz
    """
    fig = plt.figure()

    ax = fig.gca()

    #---

    # plot the original data
    if verbose:
        print('plotting the original data')

    for ind, index in enumerate(indexes):
        color = 'C%d' % ind

        if verbose:
            print('    index=%d' % index)

        ax.plot(
            scales,
            mom[:,ind],
            color=color,
            alpha=alpha,
            marker='o',
            linestyle='none',
            markerfacecolor='none',
            label='$p=%d$'%ind,
        )

        std = cov[:,ind,ind]**0.5
        for snd, scale in enumerate(scales):
            ax.plot([scale]*2, mom[snd,ind]+std[snd]*np.array([+1,-1]), color=color, alpha=alpha)

    if np.any(mom > 0):
        ax.set_yscale('log')

    ylim = ax.get_ylim()

    #---

    # plot the inferred ansatz

    if verbose:
        print('plotting inferred ansatz')

    dense_scales = np.logspace(np.log10(np.min(scales)), np.log10(np.max(scales)), 1001)

    for ind, index in enumerate(indexes):
        color = 'C%d' % ind

        if index not in samples: # no fit for this index
            if verbose:
                print('    index=%d not in samples; skipping...' % index)
            continue

        elif verbose:
            print('    index=%d' % index)

        samp = samples[index]
        _alpha = max(0.01, 1./len(samp['amp']))

        for amp, xi, sl, bl, nl, sh, bh, nh in zip(*[samp[key] for key in ['amp', 'xi', 'sl', 'bl', 'nl', 'sh', 'bh', 'nh']]):
            ax.plot(dense_scales, structure_function_ansatz(dense_scales, amp, xi, sl, bl, nl, sh, bh, nh), color=color, alpha=_alpha)

    #---

    ax.set_xlabel('scale')
    ax.set_xscale('log')
    ax.set_xlim(xmin=scales[-1]*1.1, xmax=scales[0]/1.1)
    ax.set_xticks([], minor=True)
    ax.set_xticks(scales)
    ax.set_xticklabels(['%d'%_ for _ in ax.get_xticks()])

    ax.set_ylabel('$\left<\left|d_{x,s}\\right|^p\\right>_x$')

    ax.set_ylim(ylim)

    if legend:
        ax.legend(loc='best')

    if grid:
        ax.grid(True, which='both')

    if title:
        ax.set_title(title)

    ax.tick_params(**TICK_PARAMS)
    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig

#-------------------------------------------------

def structure_function_ansatz_violin(
        posterior,
        scale,
        color=None,
        title=None,
        hatch=None,
        alpha=0.5,
        fill=True,
        num_grid=101,
        verbose=False,
        grid=True,
        legend=True,
        fig=None,
    ):
    """violin plots of ansatz parameters as a function of structure function order
    """
    if fig is None:
        fig = plt.figure()
        ax = fig.gca()
        ymin = +np.inf
        ymax = -np.inf

    else:
        ax = fig.gca()
        ymin, ymax = ax.get_ylim()

    #-------

    if verbose:
        print('plotting violins for logarithmic derivative at scale=%.3f' % scale)

    xmin = +np.inf
    xmax = -np.inf

    for ind, index in enumerate(sorted(posterior.keys())):
        if verbose:
            print('    index=%d' % index)

        c = 'C%d' % ind if color is None else color

        # plot a quick KDE of the logarithmic derivative at a reference scale
        samp = logarithmic_derivative_ansatz(
            scale,
            posterior[index]['amp'],
            posterior[index]['xi'],
            posterior[index]['sl'],
            posterior[index]['bl'],
            posterior[index]['nl'],
            posterior[index]['sh'],
            posterior[index]['bh'],
            posterior[index]['nh'],
        )

        smin = np.min(samp)
        smax = np.max(samp)

        stdv = 1.06 * np.std(samp) / len(samp)**0.2 # rule of thumb for optimal KDE bandwidth w/r/t IMSE

        y = np.linspace(smin-3*stdv, smax+3*stdv, num_grid)
        x = np.zeros(num_grid, dtype=float)

        for s in samp:
            x += np.exp(-0.5*(y-s)**2/stdv**2)

        x *= 0.25/np.max(x)

        if fill:
            ax.fill_betweenx(y, index-x, index+x, color=c, hatch=hatch, alpha=alpha, label='$p=%d$'%index)
        else:
            ax.plot(index-x, y, color=c, alpha=alpha, label='$p=%d$'%index)
            ax.plot(index+x, y, color=c, alpha=alpha)

        ymin = min(np.min(y), ymin)
        ymax = max(np.max(y), ymax)

        xmin = min(index, xmin)
        xmax = max(index, xmax)

    xmin = min(xmin-0.5, 0)
    xmax = xmax+0.5

    ymin = min(ymin, 0)

    #---

    # add reference lines

    x = np.linspace(xmin, xmax, 11)
    ax.plot(x, x/3, color='k', linestyle='dashed', label='$p/3$')
    ax.plot(x, x/4, color='k', linestyle='dotted', label='$p/4$')

    #---

    ax.set_xlabel('$p$')
    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_xticks(range(int(xmin), int(xmax)+1))

    ax.set_ylabel('$d\log S^P_\\tau/d\log \\tau \ @ \ \\tau=%.1f$' % scale)
    ax.set_ylim(ymin=ymin, ymax=ymax)

    if legend:
        ax.legend(loc='best')

    if grid:
        ax.grid(True, which='both')

    if title:
        ax.set_title(title)

    ax.tick_params(**TICK_PARAMS)
    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #-------

    return fig
