# Analysis of `M05MA10 Turb_hdf5_plt_cnt_0050`

## Basic seach for intermittency at high wavenumbers (small scales)

A common feature of turblent flows analyzed with multi-resolution decompositions in the literature is the appearence of "intermittency" at high wave numbers.
As best as I can tell, this is defined as excess kurtosis compared to a Gaussian in the distribution of wavelet coefficients at a particular wavenumber.
More concisely, the tails of the distribution are fat a high wavenumber.

Below, I show histograms of the wavelet coefficients at different scales for the density, magnitude of the magnetic field, and magnitude of the fluid velocity.
For our immediately purposes, one should look at the detail coefficients (right side of each figure).
Each figure shows both a differential and cumulative histogram of the wavelet coefficients in blue.
The grey shading corresponds to a Gaussian distribution with the same mean and variance as the true distribution.
Importantly, the tails of the real distribution are broader than the Gaussian at high wavenumbers, but they become comparable to the Gaussian fit at lower wavenumbers.

**CAVEAT**, it's unclear whether the apparent agreement with a Gaussian at low wavenumbers could just be a "sample size" effect in that we have fewer points at low wavenumbers and are therefore less able to probe deep into the tails of the distributions.

Each figure also contains a scatter plot of points that are identified as being relatively deep in the tails of the distributions.
I don't personally get much out of the scatter plots at the moment besides the fact that we can maybe identify large(-ish) structures by inspection from the approximate coefficients at different resolutions (i.e., the structures become clearer and clearer, like what is seen in [dummy density data](summary-dummy.md)).

|wavenumber|scale|density|magnitude of magnetic field|magnitude of velocity|
|----------|-----|-------|---------------------------|---------------------|
|    1/512 |   1 |<img src="M05MA10/test-3d-scatter-001-001-001-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-001-001-001-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-001-001-001-vel_Turb_hdf5_plt_cnt_0050.png">|
|    1/256 |   2 |<img src="M05MA10/test-3d-scatter-002-002-002-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-002-002-002-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-002-002-002-vel_Turb_hdf5_plt_cnt_0050.png">|
|    1/128 |   4 |<img src="M05MA10/test-3d-scatter-004-004-004-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-004-004-004-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-004-004-004-vel_Turb_hdf5_plt_cnt_0050.png">|
|    1/ 64 |   8 |<img src="M05MA10/test-3d-scatter-008-008-008-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-008-008-008-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-008-008-008-vel_Turb_hdf5_plt_cnt_0050.png">|
|    1/ 32 |  16 |<img src="M05MA10/test-3d-scatter-016-016-016-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-016-016-016-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-016-016-016-vel_Turb_hdf5_plt_cnt_0050.png">|
|    1/ 16 |  32 |<img src="M05MA10/test-3d-scatter-032-032-032-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-032-032-032-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-032-032-032-vel_Turb_hdf5_plt_cnt_0050.png">|
|    1/  8 |  64 |<img src="M05MA10/test-3d-scatter-064-064-064-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-064-064-064-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-064-064-064-vel_Turb_hdf5_plt_cnt_0050.png">|
|    1/  4 | 128 |<img src="M05MA10/test-3d-scatter-128-128-128-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-128-128-128-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-128-128-128-vel_Turb_hdf5_plt_cnt_0050.png">|

---

**TO DO**

The following seem like good ideas

  * inspect the behavior of dens, mag, vel by eye for anything interesting
    - for vectors, look both at
      * magnitude
      * cartesian components
    - also compute the following for vectors (and treat them as additional vectors)
      * curl? <-- look for evidence of vorticity?
      * div?
  * look at whether the same points are identified (separately with either large approx or detail coefficients) across multiple scales
  * look at "normal" spectrograms (k vs x) for 1D lines through the grid
    - consider ever possible line? (expensive...)
    - just pick one (or a few) as representative?
  * look at spectrograms at a fixed position
    - at each scale, plot the wavelet coefficients for all pixels within the original pixel?
    - expect them to be scattered, but perhaps there is something we can say about the characteristics of their distribution? a predictable scaling of their variance with the scale?
  * look at whether points identified in one field correspond to points identified in other fields
    - show that density gradients source vorticity?
    - does the matter live in the same place as the magnetic field?
    - is the densest matter moving the fastest?

These might be useful updates to the software

  * update xlim selection to better highlight distributions that are not centered on zero
  * look at other wavelet transforms (maybe Haar is obscuring something?)
    - everything is starting to look reasonable to me, so maybe we don't have to worry about this?
