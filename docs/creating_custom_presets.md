# Parameter Table

## Backend Settings
| Parameter      | Default Value | Description |
|---------------|--------------|-------------|
| `backend`     | `cp`         | Currently not used, but useful for future. |
| `default_dtype` | `"double"` | Default data type for computations. |

## Dataset
| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `data_path`                      | `""`                         | Path to the dataset. |
| `masks`                          | `None`                       | Mask data. |
| `data_multiplier`                | `1`                          | Multiplier for data values. |
| `data_pad`                       | `0`                          | Padding applied to data. |
| `data_bin`                       | `1`                          | Binning factor for data. |
| `data_is_numpy_and_flip_ky`      | `False`                      | Flag indicating if data is NumPy format and whether to flip ky. |
| `data_shift_vector`              | `[0,0]`                      | Shift vector applied to data. |
| `upsample_pattern`               | `1`                          | Pattern used for upsampling. |
| `sequence`                       | `None`                       | Sequence used in processing. |
| `use_full_FOV`                   | `True`                       | Flag to use full field of view. |

## Saving and Printing
| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `output_folder`                  | `""`                         | Folder to save output. |
| `save_loss_log`                  | `True`                       | Flag to save loss log. |
| `epoch_prev`                     | `0`                          | Previous epoch count. |
| `save_checkpoints_every_epoch`   | `False`                      | Save checkpoints every epoch. |
| `save_inter_checkpoints`         | `True`                       | Save intermediate checkpoints. |
| `print_flag`                     | `3`                          | Print verbosity level. |

## Experimental Parameters
| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `acc_voltage`                    | `60`                         | Acceleration voltage in kV. |
| `aperture_mask`                  | `None`                       | Mask for the aperture. |
| `recon_type`                     | `"far_field"`                | Type of reconstruction (e.g., far-field). |
| `alpha_near_field`               | `0`                          | Alpha parameter for near-field reconstruction. |
| `defocus_array`                  | `np.array([0.0])`            | Array of defocus values. |
| `Cs`                             | `0`                          | Spherical aberration coefficient. |

## Ptycho Settings
| Parameter                        | Default Value               | Description |
|-----------------------------------|-----------------------------|-------------|
| `num_slices`                     | `1`                          | Number of slices in the object. |
| `obj`                            | `np.ones((1, 1, num_slices, 1))` | Object representation in reconstruction. |
| `probe`                          | `None`                       | Probe representation. |
| `positions`                      | `np.array([[0.0, 0.0]])`     | Probe positions during scanning. |
| `tilts`                          | `np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])` | Tilt angles in real and reciprocal space. |
| `tilt_mode`                      | `0`                          | Mode for applying tilts. |
| `static_background`              | `0`                          | Static background intensity. |
| `beam_current`                   | `None`                       | Electron beam current. |

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
| `propmethod`                     | `"multislice"`              | Method used for wave propagation. |
| `allow_subPixel_shift`           | `True`                       | Allow subpixel shifts. |
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


