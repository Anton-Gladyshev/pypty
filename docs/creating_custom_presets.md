# This is a table of PyPty parameters for creating custom presets. 
For an easy preset configuration, please reffer to the initialize module. It allows to easily create all arrays. But if your experiment is pretty complex, use this guide and provided examples to create your own.

## Backend Settings

| Parameter      | Default Value | Description | 
|---------------|--------------|-------------|
| `backend`     | `cp`         | Currently not used, but will be a feature in future. Right now whenever cupy is availible, it is used as GPU backend. In no Cuda is detected, numpy is used as a CPU replacement. We do plan to add suport for Apple Silion, but wait for an optimal library to apper.|
| `default_dtype` | `"double"` | Default data type for computations. Other option is "single" |

## Dataset

| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `data_path`                      | `""`                         | Path to the dataset.  It can be an .h5 file with a dataset "data" containing 3d measurement array (N_measurements, y,x). Other option is a 4d .npy array or 3d .npy array|
| `masks`                          | `None`                       | Masks (virtual detectors) used for data compression. For uncompressed data leave it None. |
| `data_multiplier`                | `1`                          | Multiplier for data values. Use it if you want to rescale your patterns on the fly without modifying the stored dataset. All patterns will be multiplied by this number. |
| `data_pad`                       | `0`                          | Padding applied to data.  Use it if you want to pad your patterns on the fly without modifying the stored dataset. All patterns will be padded with zeros. We recomend to set it to 1/4 of the width of your patterns for optimal sampling conditions (in far-field mode).|
| `data_bin`                       | `1`                          | Binning factor for data. Use it if you want to bin your patterns on the fly without modifying the stored dataset. All patterns will be binned by this number. |
| `data_is_numpy_and_flip_ky`      | `False`                      | Flag indicating that data is in NumPy format and whether to flip ky.  Use it if your patterns are flipped and you don't want to modify the stored dataset. Other option is to create a pypty-style h5 dataset.'|
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
| `print_flag`                     | `3`                          | Print verbosity level. 0 for no printing, 1 for just one overwritable line, 2 and 3 for most detailed outputs.|

## Experimental Parameters

| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `acc_voltage`                    | `60`                         | Acceleration voltage in kV. |
| `aperture_mask`                  | `None`                       | Mask for the aperture. |
| `recon_type`                     | `"far_field"`                | Type of reconstruction (e.g., "far-field").  Other option is "near_field"|
| `alpha_near_field`               | `0`                          | Alpha parameter for near-field reconstruction & flux preservation. |
| `defocus_array`                  | `np.array([0.0])`            | Array of defocus values for near-field measurement. Irrelevant for far-field. It can contain either just one defocus common for all measurements or indicate individual values for all measurements. Units - Angstroms!|
| `Cs`                             | `0`                          | Spherical aberration coefficient. Units - Angstroms!|

## Spatial Calibration
| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `slice_distances`                | `np.array([10])`            | Distances between object slices. Units are Angstroms, you can specify just one value common for all slices, or provide individual ones. |
| `pixel_size_x_A`                 | `1`                          | Pixel size in x-direction (Angstroms). |
| `pixel_size_y_A`                 | `1`                          | Pixel size in y-direction (Angstroms). |
| `scan_size`                      | `None`                       | Tuple destibing number of scan points in y- and x- directions, only required for constraining postions and tilts. |
| `num_slices`                     | `1`                          | Number of slices in the object. |



## Refinable arrays

| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `obj`                            | `np.ones((1, 1, num_slices, 1))` | Initial guess for transmission function to be retrieved. Shape is (y,x,z,modes). If the y and x dimensions are not sufficent for a scan grid, object will be padded with ones.  |
| `probe`                          | `None`                       | Real-space probe. Shape should be (y,x,modes). For a very advanced experiment probe can be four dimensional where the last dimension accounts for different beams for different measurements. In this case you should specify a probe_marker. If Note, the PyPty will automatically initialize the beam from the dataset. For more see section beam initialization.|
| `positions`                      | `np.array([[0.0, 0.0]])`     | Scan positions. Units should be pixels of your reconstruction. The shape of postions array shold be `[N_measurements, 2]`. Or to be more presice is should look like [[y0,x0],[y1,x1],....[yn,xn]]. For single-shot type of experiments you can define one common scan point, e.g. [[0,0]. |
| `tilts`                          | `np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])` | Tilt angles in real and reciprocal spaces. There are 3 types of tilts in PyPty framework: before, inside and after. First one is a beam tilt before the specimen, i.e. a shift in aperture plane. Second type is a tilt inside of a specimen, i.e. after each slice the beam is shifted in real space. Third type is a post-specimen tilt i.e. a shift in a detector plane.  All three types of shifts are contained in this tilt array. The shape should be  [N_measurements, 6].  Or to be more presice is should look like [[y0before,x0before, y0inside, x0inside, y0after, x0after ],.... [ynbefore,x0before, yninside, xninside, ynafter, xnafter ]]. For single-shot type of experiments you can define one common tilt.|
| `tilt_mode`                      | `0`                          | Mode for applying tilts. tilt inside 0, 3,4; tilt before 2,4; tilt after 1,3,4;|
| `static_background`              | `0`                          | Static background intensity. It should be numpy array with the same shape as the initial patterns but padded by data_pad//upsample_pattern. Leave it 0 for no static offset on the diffractions patterns |
| `beam_current`                   | `None`                       | Numpy array accounting for different current (or exposure times) during different measuremtns. If not None (no variation), it should be 1D array with length corresponding to the total number of measurements. |



## Propagation, Shifting and Resizing
| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `propmethod`                     | `"multislice"`              | Method used for wave propagation. Default is "multislice". Other options are additive splitting- "better_multislice" and "yoshida". The last two options have higher precision than multislice, but requiere more time. |
| `allow_subPixel_shift`           | `True`                       | Allow subpixel shifts. If False, positions will be rounded to integer values until you start to refine them.|
| `dynamically_resize_yx_object`   | `False`                      | If position update becomes too large, object will be padded with ones in order to properly accomodate the updated grid. When specified as a positive integer, resizing will happen every time positon correction exceeds this integer value. |
| `extra_space_on_side_px`         | `0`                          | Extra space added around the object (pixels). |


## Bandwidth Limitation
| Parameter                    | Default Value   | Description |
|------------------------------|----------------|-------------|
| `damping_cutoff_multislice`  | `2/3`          | This is a frequency cutoff for multislice beam propgation. Anything larger than 2/3 will result in aliasing artifacts, we recommend to keep it smaller than this value|
| `smooth_rolloff`             | `0`            | Rolloff parameter for smooth frequency cutoffs. |
| `update_extra_cut`           | `0.005`        | Extra frequency cutoff for a full object. During multislice only cropped ROIs of object are bandwidthlimited to avoid artifacts in a full transmission function, we suggest to limit it as well. Full object will have a limit of damping_cutoff_multislice-update_extra_cut of the largest spatial frequency.|
| `lazy_clean`                 | `False`        | Whether to perform lazy cleaning. If True, full transmission function will not be bandwidth limited (only the cropped ROIs will be). We recommend to keep it False. |

## Optimization Settings
| Parameter                  | Default Value  | Description |
|----------------------------|---------------|-------------|
| `algorithm`                | `"lsq_sqrt"`  | Error metric used for comparing outcome of a reconstruction with measurement. Optipns are "lsq_sqrt"- Gaussian statistic, "ml" for Poisson, "lsq" for classic summed squared error and "lsq_sqrt_2" for modified Gaussian statistic|
| `epoch_max`                | `200`         | Maximum number of epochs (iterations). |
| `wolfe_c1_constant`        | `0.5`         | Wolfe condition parameter (C1). PyPty is supplied with weak Wolfe conditions. This parameter prevents update step being too large. C1 shold be postive and smaller than C2 constant. Larger C1 values are less forgiving and make step smaller. |
| `wolfe_c2_constant`        | `0.999999`    | Wolfe condition parameter (C2). This parameter prevents update step being too small. C2 should be sticktly larger than C1 but smaller than 1. Larger C2 values are more forgiving and allow larger steps. |
| `loss_weight`              | `1`           | Weight applied to the loss function. |
| `max_count`                | `None`        | Maximum number of forward-backward propagations for one linesearch iteration. When exceeded, algorithm terminates the linesearch, the update is rejected and the history is restarted. Set to None or np.inf if you don't want to use this option.|
| `reduce_factor`            | `0.5`         | Factor for reducing step size. Applied when 1st Wolfe condition is not satistied.|
| `optimism`                 | `2`           | Optimism factor in optimization. Applied when 2nd Wolfe condition is not satistied. |
| `min_step`                 | `1e-20`       | Minimum step size allowed. When the actual step is smaller than this value, algorithm terminates the linesearch, the update is rejected and the history is restarted. Set to zero if you don't want to use this option.|
| `hist_length`              | `10`          | PyPty optimization is essentialy a two-loop BFGS algorithm. History length controlls the behaviour of update constuction. Gradients and updates of previous N=hist_length iterations will be stored and used for l-BFGS update construction. Set to 0 for classical gradient descent, 1 for Hestenes–Stiefel conjugate gradient, any integer larger than 1 for limited-memory BFGS algorithm and np.inf for true BFGS algorithm.  |
| `update_step_bfgs`         | `1`           | Common step applied to all refinable quantities. By default, after the first iteration, a Barzilai-Borwein method is used to inistialize the inverse Hessian, so most of the time, an update step of 1 should be accepted. Only during the very first iteration the linesearch might take some time to find an appropriate step. |
| `phase_only_obj`                 | `False`       | Whether to consider the object as phase-only. |
| `tune_only_probe_phase`          | `False`       | Optimize only the reciprocal-space phase (CTF) of the probe. |
| `tune_only_probe_abs`            | `False`       | Optimize only the reciprocal-space amplitude (aperture) of the probe. |
| `reset_history_flag`       | `None`        | Flag to reset optimization history. See section "lambda-types" in the end of this document. If provided, history will be manually resetted. |

## Updating refinable arrays
| Parameter                  | Default Value | Description |
|----------------------------|--------------|-------------|
| `update_probe`             | `1`          | Whether to update the probe. |
| `update_obj`               | `1`          | Whether to update the object. |
| `update_probe_pos`         | `0`          | Whether to update probe position. |
| `update_tilts`             | `0`          | Whether to update tilt angles. |
| `update_beam_current`      | `0`          | Whether to update beam current. |
| `update_aberrations_array` | `0`          | Whether to update aberrations array. |
| `update_static_background` | `0`          | Whether to update static background. |

## Multiple Illumination Functions
| Parameter                  | Default Value     | Description |
|----------------------------|------------------|-------------|
| `aberrations_array`        | `np.array([[0.0]])` | Array of aberration values. Note that this array is accouting for a varying beam and is usefull for large fields of view where the beam is litteraly changing. If applied, the iterative reconstruction will split your field of view into multiple subscans and have the same beam in each of them, but apply a different CTF in each of these regions. This array should have a shape of [N_subscans, N_aberrations]. If you are shure that your illumination function is not changing, leave it aside.|
| `phase_plate_in_h5`        | `None`           | Phase plate stored in HDF5 format. For each of the measurements, you can apply a different phase plate. You have to create an h5 file with dataset ["configs"]. This array should have a shape [N_measrurements, Y_probe, X_probe]. Note that probe dimenstions assumed to be padded and upsampled. Phase plate is applied to all modes of your beam.|
| `aberration_marker`        | `None`           | Marker for multiple CTFs. If you provided an aberrations_array, you should also specify this marker. This marker should be a 1D array of length N_measurements and each enty should have an index of a particular CTF stored in aberrations_array. |
| `probe_marker`             | `None`           | If your beam is varying too strongly and specifying aberrations_array is not sufficint, you can create completely independent beams for each subscan. In this case you initialize the probe with shape [y,x,N_modes, N_subscans] and provide this marker as 1D array with length [N_measurements] where each entry contains an index of a beam to use for a particular measurement.  |

## Memory Usage
| Parameter                | Default Value       | Description |
|--------------------------|--------------------|-------------|
| `load_one_by_one`        | `True`             | Load data one by one to save memory. If True, the full dataset will NOT be loaded into GPU memory and only the patterns that are being processed will be converted on-the-fly. If False, the alogorithm runs faster, but takes more GPU memory.|
| `smart_memory`           | `True`             | Enable smart memory management. If True, pool and pinned pool will be occasionally cleared after heavy computations to avoid memory fragmentation and the FFT cache will be cleared. False results in a slightly faster runtime, but can cause memory leaks.|
| `remove_fft_cache`       | `False`            | Whether to remove FFT cache. Experimental feature that should enable smart memory without clearing the FFT cache. Currently not fully implemented.|
| `compute_batch`          | `1`                | Batch size for multislice computation. PyPty computes the gradients for all patterns at once via a for-loop iterating through all the measurements. You can partially vectorize this loop via this parameter to make the computsation faster. Larger values result in faster computations, but you should find out which value is best for your GPU. For Tesla V100 compute_batch of 8 saturates the GPU with fairly larger object and patterns, for H100, compute batches of up to 100 can be used. |
| `force_dataset_dtype`    | `default_float_cpu`| Force dataset data type. Numpy dtype. If you want to optimize memory usage via a loss of precision, you can specify a different datatype for your dataset. |
| `preload_to_cpu`        | `False`            | Preload data to CPU. If your data is stored as .h5 file, you can accelerate the memory transfers by avoiding lazy-loading. |
| `force_pad`             | `False`            | Whether to force padding. If you pad your data with zeros, you can do it either on-the-fly to avoid storing the zeros in your GPU memory (force_pad=False), or accelerate the reconstruction by padding during the very first epoch (force_pad=True). True uses MUCH more GPU memory.  |



## Constraints contributing to the loss

| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `mixed_variance_weight`          | `0`                          | Custom mixed object constraint preventing variation of the states at lower spatial frequencies. This parameter controlls the strength of a regularization.|
| `mixed_variance_sigma`           | `0.5`                        | Custom mixed object constraint preventing variation of the states at lower spatial frequencies. This parameter controlls the range of spatial frequencies. |
| `probe_constraint_mask`          | `None`                       | Mask for probe constraint in reciprocal space. Masked pixel will be damped via l2-regularization. |
| `probe_reg_constraint_weight`    | `0`                          | Reciprocal space probe l2-regularization weight. |
| `window_weight`                  | `0`                          | Real space probe l2-regularization weight.|
| `window`                         | `None`                       | Window function used for probe constraint in real space. Masked pixel will be damped via l2-regularization. |
| `abs_norm_weight`                | `0`                          | Weight applied to the l1 norm of an absorptive potential (negative log of transmission functions absolute value) |
| `phase_norm_weight`              | `0`                          | Weight applied to the l1 norm of an electrostatic potential (phase of complex transmission functions) |
| `atv_weight`                     | `0`                          | Adaptive Total Variation (ATV) weight applied to transmission function|
| `atv_q`                          | `1`                          | ATV q parameter. We suggest to keep it 1|
| `atv_p`                          | `2`                          | ATV p parameter. p constrolls the behaviour of ATV. 1 yiels l1-style regularization and 2 - l2. |
| `fast_axis_reg_weight_positions` | `0`                          | Regularization weight for fast-axis positions. |
| `fast_axis_reg_weight_tilts`     | `0`                          | Regularization weight for fast-axis tilts. |
| `slow_axis_reg_weight_positions` | `0`                          | Regularization weight for slow-axis positions. |
| `slow_axis_reg_coeff_positions`  | `0`                          | Regularization coefficient for slow-axis positions. |
| `slow_axis_reg_weight_tilts`     | `0`                          | Regularization weight for slow-axis tilts. |
| `slow_axis_reg_coeff_tilts`      | `0`                          | Regularization coefficient for slow-axis tilts. |


## Constraints that Modify the Object and Probe 'By Hand'. 
### Warning: when applied, this constraints always reset the BFGS history 

| Parameter                         | Default Value  | Description |
|-----------------------------------|---------------|-------------|
| `apply_gaussian_filter`          | `False`       | Apply a Gaussian filter to the phase of object.  |
| `apply_gaussian_filter_amplitude`| `False`       | Apply a Gaussian filter to the amplitude of object. |
| `beta_wedge`                     | `0`           | Missing wedge beta parameter. 3D Objects FFT will be cleaned to avoid high kz frequencies for low kx and ky. |
| `keep_probe_states_orthogonal`   | `False`       | Keep probe modes orthogonal to each other. |
| `do_charge_flip`                 | `False`       | Perform charge flipping on the object. |
| `cf_delta_phase`                 | `0.1`         | Delta phase for charge flipping. |
| `cf_delta_abs`                   | `0.01`        | Delta amplitude for charge flipping. |
| `cf_beta_phase`                  | `-0.95`       | Beta phase parameter for charge flipping. |
| `cf_beta_abs`                    | `-0.95`       | Beta amplitude parameter for charge flipping. |
| `fancy_sigma`                    | `None`        | Custom sigma parameter to engource atomicity. |
| `restart_from_vacuum`            | `None`          | Restart reconstruction from a vacuum state. Object will be set to 1 while everything else will remain the same. |



## Beam Initialization
| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `n_hermite_probe_modes`          | `None`                       | Number of Hermite probe modes. Tuple of two values: [nx, ny]. When provided, probe modes will be created based on Hermite polynomials. |
| `defocus_spread_modes`           | `None`                       | Modes for defocus spread. 1D numpy array with different defocus values. When provided, probe modes will be intialized by defocussing the beam.|
| `aberrations`                    | `None`                       | Aberration coefficients. This should be list or 1d numpy array containing beam aberrations (in Å). Aberrations are stored in Krivanek notation, e.g. C10, C12a, C12b, C21a, C21b, C23a, C23b, C30 etc. If None, no CTF will be applied.|
| `extra_probe_defocus`            | `0`                          | Extra probe defocus in Angstrom. Usefull for propagating the intial beam in multislice reconstrctions. |
| `estimate_aperture_based_on_binary` | `0`                   | Estimate aperture based on binary treshold. If postive, the aperture will be estimated based on data entries that are larger than the mean value times `estimate_aperture_based_on_binary  |
| `beam_ctf`                       | `None`                       | Beam CTF. If not None, it must be a 2D Numpy array with the same shape as upsampled (and padded or unpadded) probe. |
| `mean_pattern`                   | `None`                       | If the beam is None, and this array is provided, probe will be created from this array via an inverse Fourier transform.  |


