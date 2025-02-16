# This is a table of PyPty parameters for creating custom presets. 
For an easy preset configuration, please see the initialize module. It allows to easily create all arrays. But if your experiment is pretty complex, use this guide or provided examples to create your own.

## Backend Settings

| Parameter      | Default Value | Description | 
|---------------|--------------|-------------|
| `backend`     | `cp`         | Currently not used, but useful for future. Right now whenever cupy is availible, it is used as GPU backend. In no Cuda is detected, numpy is used as a CPU replacement |
| `default_dtype` | `"double"` | Default data type for computations. Other option is "single" |

## Dataset

| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `data_path`                      | `""`                         | Path to the dataset.  It can be an .h5 file with a dataset "data" containing 3d measurement array (N_measurements, y,x). Other option is a 4d .npy array or 3d .npy array|
| `masks`                          | `None`                       | Masks (virtual detectors) used for data compression. For uncompressed data leave it None. |
| `data_multiplier`                | `1`                          | Multiplier for data values. Use it if you want to rescale your patterns on the fly without modifying the stored dataset. All patterns will be multiplied by this number. |
| `data_pad`                       | `0`                          | Padding applied to data.  Use it if you want to pad your patterns on the fly without modifying the stored dataset. All patterns will be padded with zeros. We recomend to set it to 1/4 of the width of your patterns for optimal sampling conditions (in far-field mode).|
| `data_bin`                       | `1`                          | Binning factor for data. Use it if you want to bin your patterns on the fly without modifying the stored dataset. All patterns will be binned by this number. |
| `data_is_numpy_and_flip_ky`      | `False`                      | Flag indicating if data is NumPy format and whether to flip ky.  Use it if your patterns are flipped and you don't want to modify the stored dataset.'|
| `data_shift_vector`              | `[0,0]`                      | Shift vector applied to data.  Use it if you want to shift your patterns on the fly without modifying the stored dataset. All patterns will be shifted by provided number of pixels. |
| `upsample_pattern`               | `1`                          | Upsampling factor. If your beam footprint ends up being larger than the extent (in far-field mode), use it to artificially upsample the beam in reciprocal space. This is experimental feature and you do want to apply windowing constraints later! |
| `sequence`                       | `None`                       | Sequence used in data processing. This is a list indiciating the measuremnts that will be used for iterative refinement. If None, all measurements will contribute. This parameter is usefull for reconstructions on subscans if you don't want to create additional data files. |
| `use_full_FOV`                   | `True`                       | Boolean flag. It is only usefull if you provided a sequence. True will result in an object that can accomodate all measurements, False will create an object that accomodates only selected measurements.|

## Saving and Printing

| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `output_folder`                  | `""`                         | Path to a folder to save the output files. |
| `save_loss_log`                  | `True`                       | Boolean flag. If true, the loss log will be saved as loss.csv |
| `epoch_prev`                     | `0`                          | Previous epoch count. Usefull for restaring a reconstruction.|
| `save_checkpoints_every_epoch`   | `False`                      | Save checkpoints every epoch.  |
| `save_inter_checkpoints`         | `True`                       | Save intermediate checkpoints.  It will create .npy arrays co.npy for the object, cp.npy for probe, cg.npy for the scan grid, ct.npy for the tilts, cs.npy for static background and cb.npy for the beam current. |
| `print_flag`                     | `3`                          | Print verbosity level. 0 for no printing and 1 for just one overwritable line, 2 and 3 for most detailed outputs.|

## Experimental Parameters

| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `acc_voltage`                    | `60`                         | Acceleration voltage in kV. |
| `aperture_mask`                  | `None`                       | Mask for the aperture. |
| `recon_type`                     | `"far_field"`                | Type of reconstruction (e.g., "far-field").  Other option is "near_field"|
| `alpha_near_field`               | `0`                          | Alpha parameter for near-field reconstruction & flux preservation. |
| `defocus_array`                  | `np.array([0.0])`            | Array of defocus values for near-field measurement. Irrelevant for far-field. It can contain either just one defocus common for all measurements or indicate individual values for all measurements. Units - Angstroms!|
| `Cs`                             | `0`                          | Spherical aberration coefficient. Units - Angstroms!|

## Refinable arrays

| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `num_slices`                     | `1`                          | Number of slices in the object. |
| `obj`                            | `np.ones((1, 1, num_slices, 1))` | Initial guess for transmission function to be retrieved. Shape is (y,x,z,modes). If the y and x dimensions are not sufficent for a scan grid, object will be padded with ones.  |
| `probe`                          | `None`                       | Real-space probe. Shape should be (y,x,modes). For a very advanced experiment probe can be four dimensional where the last dimension accounts for different beams for different measurements. In this case you should specify a probe_marker. If Note, the PyPty will automatically initialize the beam from the dataset. For more see section beam initialization.|
| `positions`                      | `np.array([[0.0, 0.0]])`     | Scan positions. Units should be pixels of your reconstruction. The shape of postions array shold be `[N_measurements, 2]`. Or to be more presice is should look like [[y0,x0],[y1,x1],....[yn,xn]]. For single-shot type of experiments you can define one common scan point, e.g. [[0,0]. |
| `tilts`                          | `np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])` | Tilt angles in real and reciprocal spaces. There are 3 types of tilts in PyPty framework: before, inside and after. First one is a beam tilt before the specimen, i.e. a shift in aperture plane. Second type is a tilt inside of a specimen, i.e. after each slice the beam is shifted in real space. Third type is a post-specimen tilt i.e. a shift in a detector plane.  All three types of shifts are contained in this tilt array. The shape should be  [N_measurements, 6].  Or to be more presice is should look like [[y0before,x0before, y0inside, x0inside, y0after, x0after ],.... [ynbefore,x0before, yninside, xninside, ynafter, xnafter ]]. For single-shot type of experiments you can define one common tilt.|
| `tilt_mode`                      | `0`                          | Mode for applying tilts. tilt inside 0, 3,4; tilt before 2,4; tilt after 1,3,4;|
| `static_background`              | `0`                          | Static background intensity. It should be numpy array with the same shape as the initial patterns but padded by data_pad//upsample_pattern. Leave it 0 for no static offset on the diffractions patterns |
| `beam_current`                   | `None`                       | Numpy array accounting for different current (or exposure times) during different measuremtns. If not None (no variation), it should be 1D array with length corresponding to the total number of measurements. |

## Spatial Calibration
| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `slice_distances`                | `np.array([10])`            | Distances between slices in the object. |
| `pixel_size_x_A`                 | `1`                          | Pixel size in x-direction (Angstroms). |
| `pixel_size_y_A`                 | `1`                          | Pixel size in y-direction (Angstroms). |
| `scan_size`                      | `None`                       | Scan size of the probe. |

## Propagation, Windowing, and Resizing
| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `propmethod`                     | `"multislice"`              | Method used for wave propagation. Default is "multislice". Other options are additive splitting- "better_multislice" and "yoshida". The last two options have higher precision than multislice, but requiere more time. |
| `allow_subPixel_shift`           | `True`                       | Allow subpixel shifts. If False, positions will be rounded to integer values until you start to refine them.|
| `dynamically_resize_yx_object`   | `False`                      | Dynamically resize the object. |
| `extra_space_on_side_px`         | `0`                          | Extra space added around the object (pixels). |

## Constraints
| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `mixed_variance_weight`          | `0`                          | Mixed variance weight. |
| `mixed_variance_sigma`           | `0.5`                        | Mixed variance sigma. |
| `probe_constraint_mask`          | `None`                       | Mask for probe constraint. |
| `probe_reg_constraint_weight`    | `0`                          | Probe regularization weight. |
| `window_weight`                  | `0`                          | Window function weight. |
| `window`                         | `None`                       | Window function. |
| `abs_norm_weight`                | `0`                          | Absolute normalization weight. |
| `phase_norm_weight`              | `0`                          | Phase normalization weight. |
| `atv_weight`                     | `0`                          | Adaptive Total Variation (ATV) weight. |
| `atv_q`                          | `1`                          | ATV q parameter. |
| `atv_p`                          | `2`                          | ATV p parameter. |
| `fast_axis_reg_weight_positions` | `0`                          | Regularization weight for fast-axis positions. |
| `fast_axis_reg_weight_tilts`     | `0`                          | Regularization weight for fast-axis tilts. |
| `slow_axis_reg_weight_positions` | `0`                          | Regularization weight for slow-axis positions. |
| `slow_axis_reg_coeff_positions`  | `0`                          | Regularization coefficient for slow-axis positions. |
| `slow_axis_reg_weight_tilts`     | `0`                          | Regularization weight for slow-axis tilts. |
| `slow_axis_reg_coeff_tilts`      | `0`                          | Regularization coefficient for slow-axis tilts. |

## Beam Initialization
| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `n_hermite_probe_modes`          | `None`                       | Number of Hermite probe modes. |
| `defocus_spread_modes`           | `None`                       | Modes for defocus spread. |
| `aberrations`                    | `None`                       | Aberration coefficients. |
| `extra_probe_defocus`            | `0`                          | Extra probe defocus. |
| `estimate_aperture_based_on_binary` | `False`                   | Estimate aperture based on binary mask. |
| `beam_ctf`                       | `None`                       | Beam contrast transfer function. |
| `mean_pattern`                   | `None`                       | Mean diffraction pattern. |


