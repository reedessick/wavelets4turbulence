"""a module for general wavelet decompositions that relies on PyWavelets (https://pywavelets.readthedocs.io/en/latest/index.html)
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import pywt

from .w4t import WaveletArray

#-------------------------------------------------

DEFAULT_WAVELET = 'haar'

#-------------------------------------------------

class PyWaveletArray(WaveletArray):
    """an object that manages storage and wavelet decompositions of ND arrays
    """
    _mode = 'periodization' # this is important to the memory structure within this array and should not be changed!

    def __init__(self, array, wavelet):
        self._wavelet = wavelet
        WaveletArray.__init__(self, array)

    #--------------------

    @property
    def wavelet(self):
        return self._wavelet

    @property
    def mode(self):
        return self._mode

    #--------------------

    def _dwtn(self, array, axis):

        # perform decomposition
        ans = pywt.dwtn(array, self.wavelet, mode=self.mode, axes=[axis])
        return ans['a'], ans['d']

    def _idwtn(self, a, d, axis):

        # perform inverse decomposition
        return pywt.idwtn(dict(a=a, d=d), self.wavelet, mode=self.mode, axes=[axis])
