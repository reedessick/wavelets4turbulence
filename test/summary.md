As a simple test, I construct a scalar field within the unit-cube as follows

  * independent draws from a standard normal distribution at each grid point
  * an additive "vertical tube" with a smaller amplitude than the Gaussian noise: `0.5*np.exp(-0.5*((x-0.5)**2 + (y-0.5)**2)/0.1**2)`

I consider a unit-cube with `256` grid points per side.
I then take Haar decompositions of the field and look for the appearence of structure.

## 2-dimensional tests (looking at the mid-plane)

First, I examine the behavior of the field at different scales along the mid-plane (`z=0.5`).
Structure identified in this slice therefore only uses spatial information from the x- and y-directions.

<img src="test-2d-scatter-001-001.png">

test-2d-scatter-002-002.png
test-2d-scatter-004-004.png
test-2d-scatter-008-008.png
test-2d-scatter-016-016.png
test-2d-scatter-032-032.png
test-2d-scatter-064-064.png
test-2d-scatter-128-128.png

## 2-dimensional tests (collapsed along the z-axis)

test-2d-scatter-001-001.png
test-2d-scatter-002-002.png
test-2d-scatter-004-004.png
test-2d-scatter-008-008.png
test-2d-scatter-016-016.png
test-2d-scatter-032-032.png
test-2d-scatter-064-064.png
test-2d-scatter-128-128.png

## 3-dimensional tests

test-3d-scatter-001-001-001.png
test-3d-scatter-002-002-002.png
test-3d-scatter-004-004-004.png
test-3d-scatter-008-008-008.png
test-3d-scatter-016-016-016.png
test-3d-scatter-032-032-032.png
test-3d-scatter-064-064-064.png
test-3d-scatter-128-128-128.png
