# **_Getting started with the `initialize` Module_**

PyPty comes with an `initialize` module (see the [API Reference](reference/initialize.md)), which is designed specifically for **far-field ptychography**.

This module makes it easy to:

- Create a **calibrated dataset**
- Generate a **PyPty preset** from scratch

If your experimental setup involves additional complexity or non-standard conditions, please refer to the [Custom Preset Guide](custom_presets.md) for more flexible configuration.

However, if your setup fits the typical **far-field ptychography** case, you can summarize all key experimental parameters in a single Python dictionary. For clarity, we usually refer to this as `experimental_params`.

The dictionary should contain the following keys:


## **Essential Data Paths**

| Key                  | Type    | Description |
|----------------------|---------|-------------|
| `data_path`         | `str`   | Path to a **3D PyPty `.h5` dataset** (`[N_measurements, ky, kx]`) or a **4D-STEM `.npy` dataset**. |
| `masks`            | `ndarray` or `None` | If the data is compressed, provide the virtual detectors (`[N_masks, ky, kx]`). |
| `output_folder`    | `str`   | Directory where results will be stored. |
| `path_json`        | `str`   | Path to a Nion-style `.json` file with metadata (optional). |

## **Electron Beam Properties**

| Key                  | Type    | Description |
|----------------------|---------|-------------|
| `acc_voltage`       | `float` | Accelerating voltage (in **kV**). |
| `rez_pixel_size_A`  | `float` | Reciprocal pixel size (in **Å⁻¹**). |
| `rez_pixel_size_mrad` | `float` | Reciprocal pixel size (in **mrad**). |
| `conv_semiangle_mrad` | `float` | Beam convergence semi-angle (in **mrad**). |
| `aperture`         | `ndarray` (optional) | Binary **2D mask** representing the aperture. |
| `bright_threshold`  | `float` | Threshold for estimating an aperture. Everything above `threshold * max(PACBED)` is considered bright field. |
| `defocus`          | `float` | Extra probe defocus (besides aberrations). |
| `aberrations`      | `list` or `ndarray` | Beam aberrations (stored in Krivanek notation). |

## **Scan and Positioning**

| Key                  | Type    | Description |
|----------------------|---------|-------------|
| `scan_size`         | `tuple(int, int)` | Number of scan points along **slow (y)** and **fast (x)** axes. |
| `scan_step_A`       | `float` | Scan step (STEM pixel size) in **Å**. |
| `fov_nm`           | `float` | Field of view (**FOV**) along the **fast axis** in **nm**. |
| `special_positions_A` | `ndarray` (optional) | If data was acquired on a non-rectangular grid, specify positions as `[y_0, x_0], ..., [y_n, x_n]` (in **Å**). |
| `transform_axis_matrix` | `ndarray (2×2)` | Transformation matrix for position correction. |
| `PLRotation_deg`    | `float` or `"auto"` | Rotation angle between scan and detector axes. If `"auto"`, an iDPC measurement estimates this angle. |

## **Basic Reconstruction Settings**

| Key                  | Type    | Description |
|----------------------|---------|-------------|
| `data_pad`         | `int` or `None` | Reciprocal space padding. Default: `1/4` of pattern width. |
| `upsample_pattern`  | `int`   | Upsampling factor for diffraction patterns. |
| `flip_ky`          | `bool`  | Flip the y-axis of diffraction patterns. |



## **Object to be Reconstructed**

| Key                  | Type    | Description |
|----------------------|---------|-------------|
| `num_slices`        | `int`   | Number of slices used for multislice propagation (default: `1`). |
| `total_thickness`   | `float` | Total thickness of the sample (in **Å**). |
| `num_obj_modes`      | `int` | Number of object modes. Default is `1`. Use it to initialize incoherent object modes  |
| `obj_phase_sigma`      | `float` |  Default is `1`. Standard deviation of Gaussian noise for initialization of incoherent object modes. |

## **Output & Debugging**

| Key                  | Type    | Description |
|----------------------|---------|-------------|
| `plot`             | `bool`  | If `True`, generates plots of key experimental parameters. |
| `print_flag`       | `int`   | Controls verbosity (`0` = silent, `1` = summary, `2+` = detailed logs). |
| `save_preprocessing_files` | `bool` | If `True`, saves intermediate preprocessing files. |


## **Nexus Tags**

| Key                  | Type    | Description |
|----------------------|---------|-------------|
| `chemical_fomula`             | `string`  | You can optionally provide a formula for your sample. PyPty will store this information. |
| `sample_name`             | `string`  | You can optionally provide a name of your sample. PyPty will store this information. |



---

