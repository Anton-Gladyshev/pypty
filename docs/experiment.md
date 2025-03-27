# **__Expected keys for experimental parameters__**

PyPty is supplied with an inialize module (see API reference). It allows to create a callibrated dataset and a pypty-preset from scratch. 
To do it, you usually need to summarize experimental parameters in one dictionary. I tend to call call it `experimental_params` and it should contain the following keys:


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

## **Scan and Positioning**

| Key                  | Type    | Description |
|----------------------|---------|-------------|
| `scan_size`         | `tuple(int, int)` | Number of scan points along **slow (y)** and **fast (x)** axes. |
| `scan_step_A`       | `float` | Scan step (STEM pixel size) in **Å**. |
| `fov_nm`           | `float` | Field of view (**FOV**) along the **fast axis** in **nm**. |
| `special_positions_A` | `ndarray` (optional) | If data was acquired on a non-rectangular grid, specify positions as `[y_0, x_0], ..., [y_n, x_n]` (in **Å**). |
| `transform_axis_matrix` | `ndarray (2×2)` | Transformation matrix for position correction. |
| `PLRotation_deg`    | `float` or `"auto"` | Rotation angle between scan and detector axes. If `"auto"`, an iDPC measurement estimates this angle. |

## **Reconstruction Settings**

| Key                  | Type    | Description |
|----------------------|---------|-------------|
| `num_slices`        | `int`   | Number of slices used for multislice propagation (default: `1`). |
| `total_thickness`   | `float` | Total thickness of the sample (in **Å**). |
| `data_pad`         | `int` or `None` | Reciprocal space padding. Default: `1/4` of pattern width. |
| `upsample_pattern`  | `int`   | Upsampling factor for diffraction patterns. |
| `flip_ky`          | `bool`  | Flip the y-axis of diffraction patterns. |
| `defocus`          | `float` | Extra probe defocus (besides aberrations). |
| `aberrations`      | `list` or `ndarray` | Beam aberrations (stored in Krivanek notation). |

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

