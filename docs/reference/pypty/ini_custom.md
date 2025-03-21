# Initialization functions availible for configuring custom presets

PyPty is supplied with an inialize module (see API reference). It allows to create a callibrated dataset and a pypty-preset from scratch. Here two core functions are explained in details:

### **create_pypty_data()**
The `create_pypty_data()` function is used to generate a **PyPty-style `.h5` dataset** from an existing **4D-STEM dataset** stored as an `.h5` or `.npy` file, or from a raw NumPy array. This function has no returns.


| Parameter        | Type              | Default   | Description |
|-----------------------------------|-----------------------------|--------------------|-------------|
| **`data`**      | `str` or `ndarray` | -         | Path to an input dataset (`.h5` or `.npy`) or a NumPy array containing 4D-STEM data. |
| **`path_output`** | `str`            | -         | Path where the new **PyPty dataset** will be saved. |
| **`swap_axes`**  | `bool`            | `False`   | Swap the last two coordinate axes. |
| **`flip_ky`**   | `bool`            | `False`   | Flip the second last axis. |
| **`flip_kx`**   | `bool`            | `False`   | Flip the last axis. |
| **`flip_y`**    | `bool`            | `False`   | Flip the first axis. |
| **`flip_x`**    | `bool`            | `False`   | Flip the second axis. |
| **`comcalc_len`** | `int`           | `1000`    | Number of measurements used to estimate the center. |
| **`comx`**      | `int` or `None`   | `None`    | X-center of the measurements (computed if `None`). |
| **`comy`**      | `int` or `None`   | `None`    | Y-center of the measurements (computed if `None`). |
| **`bin`**       | `int`             | `1`       | Binning factor applied to the last two axes. |
| **`crop_left`** | `int`             | `None/0`  | Number of pixels cropped from the left. |
| **`crop_right`** | `int`            | `None/0`  | Number of pixels cropped from the right. |
| **`crop_top`**  | `int`             | `None/0`  | Number of pixels cropped from the top. |
| **`crop_bottom`** | `int`           | `None/0`  | Number of pixels cropped from the bottom. |
| **`normalize`** | `bool`            | `False`   | If `True`, each pattern is rescaled so that its sum is **1 on average**. |
| **`cutoff_ratio`** | `float`        | `None`    | If provided, values farther than `cutoff_ratio * (width/2)` will be zeroed. |
| **`pad_k`**     | `int`             | `0`       | Padding applied to the last two axes. |
| **`data_dtype`** | `dtype`          | `np.float32` | Data type of the output file. |
| **`rescale`**   | `float`           | `1`       | If not **1**, patterns are divided by this value. |
| **`exist_ok`**  | `bool`            | `True`    | If `True`, does not overwrite the output file if it already exists. |


**Usage Example**
```python
pypty.initialize.create_pypty_data("input_data.h5", "output_data.h5", flip_x=True, bin=2, normalize=True)
```
### **append_exp_params()**
The `append_exp_params()` function is used to **calibrate an existing PyPty preset to new experimental data** by incorporating experimental metadata and calibration parameters.

#### **Parameters**
| Parameter                 | Type               | Default | Description |
|---------------------------|--------------------|---------|-------------|
| **`experimental_params`** | `dict`             | -       | A dictionary containing experimental metadata. (See details below) |
| **`pypty_params`**        | `dict` or `str` or `None` | `None` | Existing PyPty parameters. If `str`, it should be a path to a preset file. If `None`, a new parameter set is created. |

---

#### **Expected Keys in `experimental_params`**

The `experimental_params` dictionary should contain the following keys:

**Essential Data Paths**

| Key                  | Type    | Description |
|----------------------|---------|-------------|
| `data_path`         | `str`   | Path to a **3D PyPty `.h5` dataset** (`[N_measurements, ky, kx]`) or a **4D-STEM `.npy` dataset**. |
| `masks`            | `ndarray` or `None` | If the data is compressed, provide the virtual detectors (`[N_masks, ky, kx]`). |
| `output_folder`    | `str`   | Directory where results will be stored. |
| `path_json`        | `str`   | Path to a Nion-style `.json` file with metadata (optional). |

**Electron Beam Properties**

| Key                  | Type    | Description |
|----------------------|---------|-------------|
| `acc_voltage`       | `float` | Accelerating voltage (in **kV**). |
| `rez_pixel_size_A`  | `float` | Reciprocal pixel size (in **Å⁻¹**). |
| `rez_pixel_size_mrad` | `float` | Reciprocal pixel size (in **mrad**). |
| `conv_semiangle_mrad` | `float` | Beam convergence semi-angle (in **mrad**). |
| `aperture`         | `ndarray` (optional) | Binary **2D mask** representing the aperture. |
| `bright_threshold`  | `float` | Threshold for estimating an aperture. Everything above `threshold * max(PACBED)` is considered bright field. |

**Scan and Positioning**

| Key                  | Type    | Description |
|----------------------|---------|-------------|
| `scan_size`         | `tuple(int, int)` | Number of scan points along **slow (y)** and **fast (x)** axes. |
| `scan_step_A`       | `float` | Scan step (STEM pixel size) in **Å**. |
| `fov_nm`           | `float` | Field of view (**FOV**) along the **fast axis** in **nm**. |
| `special_positions_A` | `ndarray` (optional) | If data was acquired on a non-rectangular grid, specify positions as `[y_0, x_0], ..., [y_n, x_n]` (in **Å**). |
| `transform_axis_matrix` | `ndarray (2×2)` | Transformation matrix for position correction. |
| `PLRotation_deg`    | `float` or `"auto"` | Rotation angle between scan and detector axes. If `"auto"`, an iDPC measurement estimates this angle. |

**Reconstruction Settings**

| Key                  | Type    | Description |
|----------------------|---------|-------------|
| `num_slices`        | `int`   | Number of slices used for multislice propagation (default: `1`). |
| `total_thickness`   | `float` | Total thickness of the sample (in **Å**). |
| `data_pad`         | `int` or `None` | Reciprocal space padding. Default: `1/4` of pattern width. |
| `upsample_pattern`  | `int`   | Upsampling factor for diffraction patterns. |
| `flip_ky`          | `bool`  | Flip the y-axis of diffraction patterns. |
| `defocus`          | `float` | Extra probe defocus (besides aberrations). |
| `aberrations`      | `list` or `ndarray` | Beam aberrations (stored in Krivanek notation). |

**Output & Debugging**

| Key                  | Type    | Description |
|----------------------|---------|-------------|
| `plot`             | `bool`  | If `True`, generates plots of key experimental parameters. |
| `print_flag`       | `int`   | Controls verbosity (`0` = silent, `1` = summary, `2+` = detailed logs). |
| `save_preprocessing_files` | `bool` | If `True`, saves intermediate preprocessing files. |


**Nexus Tags**

| Key                  | Type    | Description |
|----------------------|---------|-------------|
| `chemical_fomula`             | `string`  | You can optionally provide a formula for your sample. PyPty will store this information. |
| `sample_name`             | `string`  | You can optionally provide a name of your sample. PyPty will store this information. |


**Usage Example**

```python
experimental_params = {
    "data_path": "experiment_data.h5",
    "acc_voltage": 300,
    "rez_pixel_size_A": 0.01,
    "scan_size": (256, 256),
    "scan_step_A": 1.5,
    "PLRotation_deg": "auto",
    "output_folder": "results/"
}
pypty_params = None # or your custom preset
pypty_params = pypty.initialize.append_exp_params(experimental_params, pypty_params)
```


---

