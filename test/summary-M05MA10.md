# Analysis of `M05MA10 Turb_hdf5_plt_cnt_0050`

**WRITE DESCRIPTION OF WHAT FOLLOWS**

|scale|density|magnitude of magnetic field|magnitude of velocity|
|-----|-------|---------------------------|---------------------|
|   1 |<img src="M05MA10/test-3d-scatter-001-001-001-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-001-001-001-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-001-001-001-vel_Turb_hdf5_plt_cnt_0050.png">|
|   2 |<img src="M05MA10/test-3d-scatter-002-002-002-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-002-002-002-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-002-002-002-vel_Turb_hdf5_plt_cnt_0050.png">|
|   4 |<img src="M05MA10/test-3d-scatter-004-004-004-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-004-004-004-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-004-004-004-vel_Turb_hdf5_plt_cnt_0050.png">|
|   8 |<img src="M05MA10/test-3d-scatter-008-008-008-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-008-008-008-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-008-008-008-vel_Turb_hdf5_plt_cnt_0050.png">|
|  16 |<img src="M05MA10/test-3d-scatter-016-016-016-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-016-016-016-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-016-016-016-vel_Turb_hdf5_plt_cnt_0050.png">|
|  32 |<img src="M05MA10/test-3d-scatter-032-032-032-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-032-032-032-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-032-032-032-vel_Turb_hdf5_plt_cnt_0050.png">|
|  64 |<img src="M05MA10/test-3d-scatter-064-064-064-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-064-064-064-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-064-064-064-vel_Turb_hdf5_plt_cnt_0050.png">|
| 128 |<img src="M05MA10/test-3d-scatter-128-128-128-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-128-128-128-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-128-128-128-vel_Turb_hdf5_plt_cnt_0050.png">|
| 256 |<img src="M05MA10/test-3d-scatter-256-256-256-dens_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-256-256-256-mag_Turb_hdf5_plt_cnt_0050.png">|<img src="M05MA10/test-3d-scatter-256-256-256-vel_Turb_hdf5_plt_cnt_0050.png">|

---

**TO DO**

  * inspect the behavior of dens, mag, vel by eye for anything interesting
    - for vectors, look both at
      * magnitude
      * cartesian components
    - also compute the following for vectors (and treat them as additional vectors)
      * curl? <-- look for evidence of vorticity?
      * div?
  * update xlim selection to better highlight distributions that are not centered on zero
  * look at "normal" spectrograms (k vs x) for 1D lines through the grid
    - consider ever possible line? (expensive...)
    - just pick one as representative?
  * look at other wavelet transforms (maybe Haar is obscuring something?)
