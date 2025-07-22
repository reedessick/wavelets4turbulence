"""a module for plotting logic for moments
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from .plot import *

from w4t.utils.infer import structure_function_ansatz

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

    ax.set_yscale('log')
    if rescale:
        ax.set_ylabel('$\left<\left|d_{x,s}\\right|^p\\right>^{1/p}_x$')
    else:
        ax.set_ylabel('$\left<\left|d_{x,s}\\right|^p\\right>_x$')

    #---

    if legend:
        ax.legend(loc='best', ncols=ncols)

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

    ax.set_yscale('log')
    if rescale:
        ax.set_ylabel('$\left<\left|d_{x,s}\\right|^p\\right>^{1/p}_x \left/ \left<d_{x,s}^2\\right>_x^{1/2} \\right.$')
    else:
        ax.set_ylabel('$\left<\left|d_{x,s}\\right|^p\\right>_x \left/ \left<d_{x,s}^2\\right>_x^{p/2} \\right.$')

    #---

    if legend:
        ax.legend(loc='best', ncols=ncols)

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
        ax.legend(loc='best', ncols=ncols)

    if grid:
        ax.grid(True, which='both')

    ax.tick_params(**TICK_PARAMS)
    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig

#-------------------------------------------------

def structure_function_ansatz_samples(scales, indexes, mom, cov, samples, alpha=0.75, legend=False, grid=True, verbose=False):
    """make a simple plot of structure function ansatz
    """
    fig = plt.figure()

    ax = fig.gca()

    #---

    # plot the original data

    for ind, index in enumerate(indexes):
        color = 'C%d' % ind

        ax.plot(scales, mom[:,ind], color=color, alpha=alpha, marker='o', linestyle='none', markerfacecolor='none', label='$p=%d$'%ind)

        std = cov[:,ind,ind]**0.5
        for snd, scale in enumerate(scales):
            ax.plot([scale]*2, mom[snd,ind]+std[snd]*np.array([+1,-1]), color=color, alpha=alpha)

    ax.set_yscale('log')
    ylim = ax.get_ylim()

    #---

    # plot the inferred ansatz

    for ind, index in enumerate(indexes):
        color = 'C%d' % ind

        samp = samples[index]
        _alpha = max(0.01, 1./len(samp['amp']))

        for amp, xi, sl, bl, nl, sh, bh, nh in zip(*[samp[key] for key in ['amp', 'xi', 'sl', 'bl', 'nl', 'sh', 'bh', 'nh']]):
            ax.plot(scales, structure_function_ansatz(scales, amp, xi, sl, bl, nl, sh, bh, nh), color=color, alpha=_alpha)

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

    ax.tick_params(**TICK_PARAMS)
    plt.subplots_adjust(**SUBPLOTS_ADJUST)

    #---

    return fig
