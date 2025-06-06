# wavelets4turbulence

A simple repository for investigations and supporting software for fast wavelet analyses of turbulent boxes.

The beating heart of this repository is a memory efficient N-dimensional wavelet decomposition.
There is a

  * custom implementation of the Haar wavelet and a 
  * general purpose implementation of many wavelets (provided through [PyWavelets](https://pywavelets.readthedocs.io/en/latest/)).

This object will allow us to quickly decompose large N-dim arrays into wavelet coefficients, invert that decomposition if needed, and acess the coefficients (as well as scales and more general voxel bounding boxes) at multiple levels of the decomposition.

---

You will need to install the code and update your `PYTHONPATH` via

```
./install
. ./env.sh # assumes that you're running python3.10.
```
