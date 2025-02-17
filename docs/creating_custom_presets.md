# PyPty Parameters for Creating Custom Presets

The main function of PyPty package is iterative ptychographic reconsturction launched via `run_ptychography()` functon. It takes a single argument- `pypty_params` dictionary describing your preset.

For an easy preset configuration, please refer to the `pypty.initialize` module. It allows easy creation of all arrays. However, if your experiment is complex, use this guide and the provided examples to create your own dictionary and fill in requiered entries.



### Before starting this guide, one important usage case must be discussed.
PyPty is an iterative algorithm and, as you will see, it requires a number of input parameters. Some of these parameters can be specified in an iteration-dependent fashion using a lambda function. This function should take a single input argument and return the desired value for a given epoch.

For example, if you want to apply smart_memory parameter every 10 epochs, you can set `smart_memory` in `pypty_params` dictionary as:

```python
smart_memory: lambda x: x % 10 == 0;
```

The parameters that can be written in this way are marked as `pypty_lambda` **type** in the  **Default Data Type** column. They can also be specified as a sting containing the code, e.g. 

```python
smart_memory: "lambda x: x % 10 == 0";
```

We do not recommend applying constraints every n epochs, as PyPty’s BFGS algorithm attempts to construct a Hessian matrix, and such modifications can disrupt this process.
As a general rule of thumb, we suggest configuring lambda functions so that once an optimization parameter is activated, it maintains a consistent value throughout execution.




---

## Backend Settings

| Parameter      | Default Value | Default Data Type | Description |
|---------------|--------------|--------------------|-------------|
| `backend`     | `cp`         | `NumPy-like python module`             | Currently not used, but will be a feature in the future. Right now, whenever CuPy is available, it is used as the GPU backend. If no CUDA is detected, NumPy is used as a CPU replacement. We plan to add support for Apple Silicon, but we are waiting for an optimal library to appear. |
| `default_dtype` | `"double"`  | `str`             | Default data type for computations. Another option is `"single"`. |

---

## Dataset

| Parameter                        | Default Value               | Default Data Type | Description |
|-----------------------------------|-----------------------------|--------------------|-------------|
| `data_path`                      | `""`                         | `str`              | Path to the dataset. It can be an `.h5` file with a dataset named `"data"` containing a **3D measurement array** `(N_measurements, y, x)`. Another option is a **4D `.npy` array** or a **3D `.npy` array**. |
| `masks`                          | `None`                       | `numpy.ndarray` or `None` | Masks (virtual detectors) used for data compression. For uncompressed data, leave it as `None`. |
| `data_multiplier`                | `1`                          | `float`            | Multiplier for data values. Used to rescale patterns on the fly without modifying the stored dataset. All patterns will be multiplied by this number. |
| `data_pad`                       | `0`                          | `int`              | Padding applied to data. Use it to pad patterns on the fly without modifying the stored dataset. We recommend setting it to **1/4 of the pattern width** for optimal sampling conditions in far-field mode. |
| `data_bin`                       | `1`                          | `int`              | Binning factor for data. Used to bin patterns on the fly without modifying the stored dataset. All patterns will be binned by this number. |
| `data_is_numpy_and_flip_ky`      | `False`                      | `bool`             | Flag indicating whether the data is in NumPy format and whether to flip `ky`. Useful if patterns are flipped and you don’t want to modify the stored dataset. Another option is to create a PyPty-style `.h5` dataset. |
| `data_shift_vector`              | `[0,0]`                      | `list`             | Shift vector (list with two-values) applied to measurements. Used to shift patterns on the fly without modifying the stored dataset. All patterns will be shifted by the specified number of pixels. |
| `upsample_pattern`               | `1`                          | `int`              | Upsampling factor. If the beam footprint is larger than the extent (in far-field mode), this allows to artificially upsample the beam in reciprocal space. **Experimental feature!** Windowing constraints may be required. |
| `sequence`                       | `None`                       | `list` or `None` or `pypty_lambda`     | Sequence used in data processing. This is a list indicating the measurements that will be used for iterative refinement. If `None`, all measurements contribute. This parameter is useful for reconstructions on subscans without creating additional data files. |
| `use_full_FOV`                   | `True`                       | `bool`             | Boolean flag. Only useful if a sequence is provided. If `True`, the object can accommodate all measurements. If `False`, the object accommodates only selected measurements. |

---

## Saving and Printing

| Parameter                        | Default Value               | Default Data Type | Description |
|-----------------------------------|-----------------------------|--------------------|-------------|
| `output_folder`                  | `""`                         | `str`              | Path to the folder where output files will be saved. |
| `save_loss_log`                  | `True`                       | `bool`             | Boolean flag. If `True`, the loss log will be saved as `loss.csv`. |
| `epoch_prev`                     | `0`                          | `int`              | Previous epoch count. Useful for restarting a reconstruction. |
| `save_checkpoints_every_epoch`   | `False`                      | `bool`   or `int`          | Save checkpoints every epoch. If `True`, checkpoints will be always saved, if it is provided as an integer, checkpoints will be saved every n'th epoch.  |
| `save_inter_checkpoints`         | `True`                       | `bool`  or `int`           | Save intermediate **overwritable** checkpoints. This will create `.npy` arrays: `co.npy` for the object, `cp.npy` for the probe, `cg.npy` for the scan grid, `ct.npy` for the tilts, `cs.npy` for the static background, and `cb.npy` for the beam current. If `True`, checkpoints will be always saved, if it is provided as an integer, checkpoints will be saved every n'th epoch. |
| `print_flag`                     | `3`                          | `int`              | Print verbosity level: `0` for no printing, `1` for one overwritable line, `2` for moderate output, and `3` for detailed outputs. |

---

## Experimental Parameters

| Parameter                        | Default Value               | Default Data Type | Description |
|-----------------------------------|-----------------------------|--------------------|-------------|
| `acc_voltage`                    | `60`                         | `float`            | Acceleration voltage in kV. |
| `aperture_mask`                  | `None`                       | `numpy.ndarray` or `None` | Mask for the aperture. Can be used for reciprocal probe constaint later (see section constraints).|
| `recon_type`                     | `"far_field"`                | `str`              | Type of reconstruction. Options: `"far_field"` or `"near_field"`. |
| `alpha_near_field`               | `0.0`                          | `float`            | Alpha parameter for near-field reconstruction & flux preservation. |
| `defocus_array`                  | `np.array([0.0])`            | `numpy.ndarray`    | Array of defocus values for near-field measurement. Irrelevant for far-field. It can contain either a single common defocus value for all measurements or individual values for each measurement. **Units: Angstroms.** |
| `Cs`                             | `0.0`                          | `float`            | Spherical aberration coefficient. **Units: Angstroms.** |

---

## Spatial Calibration

| Parameter                        | Default Value               | Default Data Type | Description |
|-----------------------------------|-----------------------------|--------------------|-------------|
| `slice_distances`                | `np.array([10.0])`            | `numpy.ndarray`    | Distances between object slices. **Units: Angstroms.** You can specify a single value common for all slices or provide individual values. |
| `pixel_size_x_A`                 | `1.0`                          | `float`            | Pixel size in the **x-direction** (**Angstroms**). |
| `pixel_size_y_A`                 | `1.0`                          | `float`            | Pixel size in the **y-direction** (**Angstroms**). |
| `scan_size`                      | `None`                       | `tuple` or `None`  | Tuple describing the number of scan points in **y- and x- directions**. Required for constraining positions and tilts. |
| `num_slices`                     | `1`                          | `int`              | Number of slices in the object. |

---


## Refinable Arrays

| Parameter                        | Default Value               | Default Data Type | Description |
|-----------------------------------|-----------------------------|--------------------|-------------|
| `obj`                            | `np.ones((1, 1, num_slices, 1))` | `numpy.ndarray`    | Initial guess for the transmission function to be retrieved. Shape: `(y, x, z, modes)`. If the `y` and `x` dimensions are insufficient for the scan grid, the object will be padded with ones. |
| `probe`                          | `None`                       | `numpy.ndarray` or `None` | Real-space probe. Shape: `(y, x, modes)`. For advanced experiments, the probe can be 4D `(y, x, modes, subscans)`. If `None`, PyPty will automatically initialize the beam from the dataset. |
| `positions`                      | `np.array([[0.0, 0.0]])`     | `numpy.ndarray`    | Scan positions in **pixels** of the reconstruction. Shape: `[N_measurements, 2]`, formatted as `[[y0, x0], [y1, x1], ..., [yn, xn]]`. Single-shot experiments can define one common scan point, e.g., `[[0,0]]`. |
| `tilts`                          | `np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])` | `numpy.ndarray` | Tilt angles in **real and reciprocal spaces**. There are 3 types of tilts in PyPty framework: before, inside and after. First one is a beam tilt before the specimen, i.e. a shift in aperture plane. Second type is a tilt inside of a specimen, i.e. after each slice the beam is shifted in real space. Third type is a post-specimen tilt i.e. a shift in a detector plane.  All three types of shifts are contained in this tilt array. Shape:  `[N_measurements, 6]`. Format: `[[y0_before, x0_before, y0_inside, x0_inside, y0_after, x0_after], ..., [yN, xN]]`. Single-shot experiments can define one common tilt (with shape `[1, 6]`). |
| `tilt_mode`                      | `0`                          | `int`              | Mode for applying tilts: `0, 3, 4` for **inside**, `2, 4` for **before**, and `1, 3, 4` for **after** the specimen. |
| `static_background`              | `0`                          | `numpy.ndarray` or `float` | Static background intensity. Shape should match initial patterns but padded by `data_pad//upsample_pattern`. Use `0` for no static offset. If provided as postive float, the algorithm will initialize the backgrund with a proper shape on its own.|
| `beam_current`                   | `None`                       | `numpy.ndarray` or `None` | Accounts for different currents (or exposure times) during measurements. If provided, must be a **1D array** with length matching `N_measurements`. |

---

## Propagation, Shifting, and Resizing

| Parameter                        | Default Value               | Default Data Type | Description |
|-----------------------------------|-----------------------------|--------------------|-------------|
| `propmethod`                     | `"multislice"`              | `str`              | Wave propagation method. Options: `"multislice"`, `"better_multislice"`, and `"yoshida"`. The last two are **higher precision but slower**. |
| `allow_subPixel_shift`           | `True`                       | `bool`             | Allow **subpixel shifts**. If `False`, positions will be rounded to integers until refined. |
| `dynamically_resize_yx_object`   | `False`                      | `bool` or `int`  or `pypty_lambda`    | If position updates become too large, the object will be padded to accommodate the new scan grid. If set to a **positive integer**, resizing occurs when position updates exceed this value. |
| `extra_space_on_side_px`         | `0`                          | `int`              | Extra space added around the object in **pixels**. |

---

## Bandwidth Limitation

| Parameter                    | Default Value   | Default Data Type | Description |
|------------------------------|----------------|--------------------|-------------|
| `damping_cutoff_multislice`  | `2/3`          | `float`            | Frequency cutoff for **multislice beam propagation**. Values larger than `2/3` can cause aliasing artifacts. Recommended **≤ 2/3**. |
| `smooth_rolloff`             | `0`            | `float`            | Rolloff parameter for smooth frequency cutoffs. |
| `update_extra_cut`           | `0.005`        | `float`            | Extra frequency cutoff for the **full object**. Ensures bandwidth limitation beyond the **cropped ROIs** of the multislice object. |
| `lazy_clean`                 | `False`        | `bool`             | If `True`, the full transmission function **will not** be bandwidth-limited (only cropped ROIs will be). **Recommended: `False`**. |

---

## Optimization Settings

| Parameter                  | Default Value  | Default Data Type | Description |
|----------------------------|---------------|--------------------|-------------|
| `algorithm`                | `"lsq_sqrt"`  | `str`              | Error metric for reconstruction comparison. Options: `"lsq_sqrt"` (Gaussian), `"ml"` (Poisson), `"lsq"` (classic summed squared error), and `"lsq_sqrt_2"` (modified Gaussian). If data is compressed via virtual detectors, the only option is `"lsq_compressed"` (summed squared error  between signals). |
| `epoch_max`                | `200`         | `int`              | Maximum number of **epochs (iterations)**. |
| `wolfe_c1_constant`        | `0.5`         | `float`  or `pypty_lambda`   | **Wolfe condition parameter (C1)**. Prevents update steps from being too large. Must be **> 0** and **< C2**. Larger values enforce shorter step size. |
| `wolfe_c2_constant`        | `0.999999`    | `float`   or `pypty_lambda`          | **Wolfe condition parameter (C2)**. Prevents update steps from being too small. Must be **> C1** but **< 1**. Larger values allow larger steps. |
| `loss_weight`              | `1`           | `float`    or `pypty_lambda`         | Weight applied to the **loss function**. |
| `max_count`                | `None`        | `int` or `None`    | Maximum number of forward-backward propagations per **line search iteration**. If exceeded, the update is **rejected** and history is reset. Use `None` or `np.inf` to disable. |
| `reduce_factor`            | `0.5`         | `float`            | Factor for **reducing step size** when the first Wolfe condition **is not met**. |
| `optimism`                 | `2`           | `float`            | Factor for **increasing step size** when the second Wolfe condition **is not met**. |
| `min_step`                 | `1e-20`       | `float`            | **Minimum step size**. If the step falls below this value, the algorithm **resets history**. Use `0` to disable. |
| `hist_length`              | `10`          | `int` or `np.inf` or `pypty_lambda`  | **BFGS optimization history length**. Values: `0` (Gradient Descent), `1` (Conjugate Gradient), `N>1` (Limited-memory BFGS), `np.inf` (Full BFGS). |
| `update_step_bfgs`         | `1`           | `float` or `pypty_lambda`  |Common step applied to all refinable quantities. By default, after the first iteration, a Barzilai-Borwein method is used to inistialize the inverse Hessian, so most of the time, an update step of 1 should be accepted. Only during the very first iteration the linesearch might take some time to find an appropriate step. |
| `phase_only_obj`                 | `False`       | `bool` or `pypty_lambda`   |Whether to consider the object as phase-only. |
| `tune_only_probe_phase`          | `False`       | `bool` or `pypty_lambda`  |Optimize only the reciprocal-space phase (CTF) of the probe. |
| `tune_only_probe_abs`            | `False`       | `bool` or `pypty_lambda`  |Optimize only the reciprocal-space amplitude (aperture) of the probe. |
| `reset_history_flag`       | `None`        | `None` or `pypty_lambda`  |Flag to reset optimization history. See section "lambda-types" in the end of this document. If provided, history will be manually resetted. |


---

## Updating Refinable Arrays

| Parameter                  | Default Value | Default Data Type | Description |
|----------------------------|--------------|--------------------|-------------|
| `update_probe`             | `1`          | `bool`   or `pypty_lambda`            | Whether to update the **probe** (`1` for yes, `0` for no). |
| `update_obj`               | `1`          | `bool`   or `pypty_lambda`            | Whether to update the **object** (`1` for yes, `0` for no). |
| `update_probe_pos`         | `0`          | `bool`   or `pypty_lambda`            | Whether to update **probe positions** (`1` for yes, `0` for no). |
| `update_tilts`             | `0`          | `bool`   or `pypty_lambda`            | Whether to update **tilt angles** (`1` for yes, `0` for no). |
| `update_beam_current`      | `0`          | `bool`    or `pypty_lambda`           | Whether to update **beam current** (`1` for yes, `0` for no). |
| `update_aberrations_array` | `0`          | `bool`   or `pypty_lambda`            | Whether to update **aberration array** (`1` for yes, `0` for no). |
| `update_static_background` | `0`          | `bool`   or `pypty_lambda`            | Whether to update **static background** (`1` for yes, `0` for no). |


---

## Multiple Illumination Functions

| Parameter                  | Default Value     | Default Data Type | Description |
|----------------------------|------------------|--------------------|-------------|
| `aberrations_array`        | `np.array([[0.0]])` | `numpy.ndarray`    | Array of aberration values for multiple beams. Useful for large fields of view where the beam changes. Shape: `[N_subscans, N_aberrations]`. |
| `phase_plate_in_h5`        | `None`           | `str` or `None`    | Path to an **HDF5 file** containing phase plates for different measurements. Dataset name should be `"configs"`. Shape: `[N_measurements, Y_probe, X_probe]`. |
| `aberration_marker`        | `None`           | `numpy.ndarray` or `None` | **Marker for multiple CTFs**. Should be a **1D array** of length `N_measurements`, where each entry corresponds to a CTF index in `aberrations_array`. |
| `probe_marker`             | `None`           | `numpy.ndarray` or `None` | **Marker for probe variations**. If provided, the probe should have shape `[y, x, modes, N_subscans]`, and this array should contain indices specifying which probe to use for each measurement. |

---


## Memory Usage

| Parameter                | Default Value       | Default Data Type | Description |
|--------------------------|--------------------|--------------------|-------------|
| `load_one_by_one`        | `True`             | `bool`             | If `True`, data is **loaded dynamically** to save GPU memory. If `False`, all data is loaded at once (faster but memory-intensive). |
| `smart_memory`           | `True`             | `bool`   or `pypty_lambda`            | If `True`, **memory is managed intelligently**, clearing cache when necessary to prevent memory fragmentation. |
| `remove_fft_cache`       | `False`            | `bool`             | If `True`, **FFT cache is removed** periodically to save memory. **(Experimental feature)** |
| `compute_batch`          | `1`                | `int`              | **Batch size for multislice computation.** Increasing this can speed up reconstruction but requires more GPU memory. |
| `force_dataset_dtype`    | `default_float_cpu`| `numpy.dtype`      | Forces the dataset to be stored in a specified **data type**. Can help reduce memory usage at the cost of precision. |
| `preload_to_cpu`         | `False`            | `bool`             | If `True`, **preloads data to CPU** before transferring it to GPU, improving transfer speeds for `.h5` datasets. |
| `force_pad`              | `False`            | `bool`             | If `True`, pads data **at the start** of reconstruction (uses more memory but speeds up computation). If `False`, padding is applied **on the fly** to save memory. |

---

## Constraints Contributing to the Loss

| Parameter                        | Default Value               | Default Data Type | Description |
|-----------------------------------|-----------------------------|--------------------|-------------|
| `mixed_variance_weight`          | `0`                          | `float`   or `pypty_lambda`          | Regularization weight that prevents **low-frequency variations** between object states. |
| `mixed_variance_sigma`           | `0.5`                        | `float`   or `pypty_lambda`          | Controls the **spatial frequency range** affected by `mixed_variance_weight`. |
| `probe_constraint_mask`          | `None`                       | `numpy.ndarray` or `None` | Mask for **probe constraint in reciprocal space**. Masked pixels are **regularized using L2 norm**. |
| `probe_reg_constraint_weight`    | `0`                          | `float`            | L2 regularization weight for the **probe in reciprocal space**. |
| `window_weight`                  | `0`                          | `float`   or `pypty_lambda`           | L2 regularization weight for the **probe in real space**. |
| `window`                         | `None`                       | `numpy.ndarray` or `None` or or `pypty_lambda` | **Window function** used to constrain the probe in real space. Masked pixels are damped using L2 regularization. It can be either a 2d- real valued array with the same shape as upsampled and padded beam **or** a list containing two values: inner radius (fraction) and outer radius (fraction). Fractions will be multiplied with half of the probe width, everything inside of window will be kept intact, everything outside will be zeroed and intermediate values will be slighly damped.|
| `abs_norm_weight`                | `0`                          | `float`            | L1 regularization weight applied to the **absorptive potential** (negative log of the transmission function’s absolute value). |
| `phase_norm_weight`              | `0`                          | `float`   or `pypty_lambda`          | L1 regularization weight applied to the **phase of the object**. |
| `atv_weight`                     | `0`                          | `float`   or `pypty_lambda`          | Weight for **Adaptive Total Variation (ATV)** regularization on the transmission function. |
| `atv_q`                          | `1`                          | `float`   or `pypty_lambda`          | ATV **q parameter** (controls the strength of smoothing). Recommended: `1`. |
| `atv_p`                          | `2`                          | `float`  or `pypty_lambda`           | ATV **p parameter** (`1` = L1-like regularization, `2` = L2-like smoothing). Recommended: `2`. |
| `fast_axis_reg_weight_positions` | `0`                          | `float` or `pypty_lambda`            | Regularization weight for **fast-axis scan positions**. |
| `fast_axis_reg_weight_tilts`     | `0`                          | `float`  or `pypty_lambda`           | Regularization weight for **fast-axis tilts**. |
| `slow_axis_reg_weight_positions` | `0`                          | `float`  or `pypty_lambda`           | Regularization weight for **slow-axis scan positions**. |
| `slow_axis_reg_coeff_positions`  | `0`                          | `float`  or `pypty_lambda`           | Regularization coefficient for **slow-axis scan positions**. |
| `slow_axis_reg_weight_tilts`     | `0`                          | `float`  or `pypty_lambda`           | Regularization weight for **slow-axis tilts**. |
| `slow_axis_reg_coeff_tilts`      | `0`                          | `float`  or `pypty_lambda`           | Regularization coefficient for **slow-axis tilts**. |

---

## Constraints That Modify the Object and Probe 'By Hand'

**Warning:** These constraints **reset the BFGS history** when applied.

| Parameter                         | Default Value  | Default Data Type | Description |
|-----------------------------------|---------------|--------------------|-------------|
| `apply_gaussian_filter`          | `False`       | `bool`   or `pypty_lambda`           | Applies a **Gaussian filter** to the **phase** of the object. |
| `apply_gaussian_filter_amplitude`| `False`       | `bool`    or `pypty_lambda`          | Applies a **Gaussian filter** to the **amplitude** of the object. |
| `beta_wedge`                     | `0`           | `float`  or `pypty_lambda`           | **Removes high kz frequencies** for low kx and ky in **3D object FFTs**. |
| `keep_probe_states_orthogonal`   | `False`       | `bool`  or `pypty_lambda`            | Enforces **orthogonality of probe modes**. |
| `do_charge_flip`                 | `False`       | `bool`             | Performs **charge flipping** on the object. |
| `cf_delta_phase`                 | `0.1`         | `float`            | **Delta phase** for charge flipping. |
| `cf_delta_abs`                   | `0.01`        | `float`            | **Delta amplitude** for charge flipping. |
| `do_charge_flip`                 | `False`       | `bool`      or `pypty_lambda`        |Perform charge flipping on the object. |
| `cf_delta_phase`                 | `0.1`         |`float` or `pypty_lambda`         | Delta phase for charge flipping. |
| `cf_delta_abs`                   | `0.01`        | `float` or `pypty_lambda`        |Delta amplitude for charge flipping. |
| `cf_beta_phase`                  | `-0.95`       | `float` or `pypty_lambda`          |Beta phase parameter for charge flipping. |
| `cf_beta_abs`                    | `-0.95`       | `float` or `pypty_lambda`          |Beta amplitude parameter for charge flipping. |
| `fancy_sigma`                    | `None`        | `None` or `float` oor `pypty_lambda`             |Custom sigma parameter to enforce atomicity. |
| `restart_from_vacuum`            | `None`        | `bool` or `pypty_lambda`    | **Resets the object to 1** while keeping other parameters unchanged. |

---


## Beam Initialization

| Parameter                        | Default Value               | Default Data Type | Description |
|-----------------------------------|-----------------------------|--------------------|-------------|
| `n_hermite_probe_modes`          | `None`                       | `tuple` or `None`  | Number of **Hermite probe modes**. A tuple `[nx, ny]` specifying mode orders in `x` and `y` directions. If `None`, no Hermite modes are applied. |
| `defocus_spread_modes`           | `None`                       | `numpy.ndarray` or `None` | 1D array with **different defocus values** for initializing probe modes. Useful for simulating **defocus spread** in the beam. |
| `aberrations`                    | `None`                       | `numpy.ndarray` or `None` | 1D array of **aberration coefficients** in **Krivanek notation**, e.g., `C10, C12a, C12b, C21a, C21b, C23a, C23b, C30`. Units: **Angstroms**. |
| `extra_probe_defocus`            | `0`                          | `float`            | Extra **probe defocus** applied in **Angstroms**. Useful for adjusting initial beam focus in **multislice reconstructions**. |
| `estimate_aperture_based_on_binary` | `0`                   | `float`            | If **> 0**, the aperture is estimated **based on a binary threshold**. Pixels in the data larger than `mean(data) * estimate_aperture_based_on_binary` are considered part of the **aperture**. |
| `beam_ctf`                       | `None`                       | `numpy.ndarray` or `None` | **Beam Contrast Transfer Function (CTF).** If provided, must be a **2D NumPy array** with dimensions matching the upsampled probe size. |
| `mean_pattern`                   | `None`                       | `numpy.ndarray` or `None` | **Mean diffraction pattern** used for probe initialization. If provided, the probe is created using an **inverse Fourier transform** of this pattern. |

---
