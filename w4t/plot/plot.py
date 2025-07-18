"""a module for plotting logic
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

import matplotlib
try:
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
except:
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True

#-------------------------------------------------

def close(fig, **kwargs):
    plt.close(fig, **kwargs)

def save(fig, figtmp, figtypes, verbose=False, indent='    ', **kwargs):
    for figtype in figtypes:
        figname = figtmp % figtype
        if verbose:
            print('%ssaving: %s' % (indent, figname))
        fig.savefig(figname, **kwargs)
