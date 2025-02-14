"""a module for plotting logic
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True

#-------------------------------------------------

def close(fig, **kwargs):
    plt.close(fig, **kwargs)

def save(fig, figtmp, figtypes, verbose=False, **kwargs):
    for figtype in figtypes:
        figname = figtmp % figtype
        if verbose:
            print('saving: '+figname)
        fig.savefig(figname, **kwargs)
