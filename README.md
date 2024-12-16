# wavelets4turbulence

A simple repository for investigations and supporting software for fast wavelet analyses of turbulent boxes.

The beating heart of this repository is a (hopefully) memory efficient N-dimensional Haar wavelet decomposition object: `w4t.haar.HaarArray`.
This object will allow us to quickly decompose large N-dim arrays into wavelet coefficients, invert that decomposition if needed, and acess the coefficients (as well as scales and more general voxel bounding boxes) at multiple levels of the decomposition.
