As a simple test, I construct a scalar field within the unit-cube as follows

  * independent draws from a standard normal distribution at each grid point
  * an additive "vertical tube" with a smaller amplitude than the Gaussian noise: `0.5*np.exp(-0.5*((x-0.5)**2 + (y-0.5)**2)/0.1**2)`

I consider a unit-cube with `256` grid points per side.
I then take Haar decompositions of the field and look for the appearence of structure.

## 2-dimensional tests (looking at the mid-plane)

First, I examine the behavior of the field at different scales along the mid-plane (`z=0.5`).
Structure identified in this slice therefore only uses spatial information from the x- and y-directions.

The original field

<img src="test-2d-scatter-001-001.png">

The coefficients after a single Haar decomposition (256/2=128 points per side).

<img src="test-2d-scatter-002-002.png">

The coefficients after 2 Haar decompositions (256/4=64 points per side).
We begin to see structure visually.

<img src="test-2d-scatter-004-004.png">

The coefficients after 3 Haar decompositions (256/8=32 poitns per side).
In addition to the clearer visual structure, the histogram of approx coefficients begins to show a clear tail.
Note, however, that the distribution of detail coefficients remains quite Gaussian.

<img src="test-2d-scatter-008-008.png">

## 2-dimensional tests (collapsed along the z-axis)

We then consider 2-dimensional structure when we first completely decompose the field along the z-axis (i.e., take the average of all values along the z-axis at each x- and y-position separately).
In this case, the structure is much more visible immediately.
This is likely because the coordinate system is aligned with the direction of the "vertical tube"

<img src="test-2da-scatter-001-001.png">

<img src="test-2da-scatter-002-002.png">

<img src="test-2da-scatter-004-004.png">

<img src="test-2da-scatter-008-008.png">

## 3-dimensional tests

Finally, we also examine 3-dimensional decompositions to look for the appearence of structure.
While the vertical tube is aligned with the z-axis, this is likely not a necessary assumption for structure discovery.
It is surely helpful, but the 3D structure is clearly visible even with minimal spatial averaging through the Haar decomposition.

<img src="test-3d-scatter-001-001-001.png">

<img src="test-3d-scatter-002-002-002.png">

<img src="test-3d-scatter-004-004-004.png">

<img src="test-3d-scatter-008-008-008.png">
