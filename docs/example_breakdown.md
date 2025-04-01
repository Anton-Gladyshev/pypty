# **_Breakdown of Examples for Iterative Ptychography_**

This section provides a detailed breakdown of an iterative ptychography reconstruction using PyPty. To get started, create a Python script and import the necessary libraries:

```python
import numpy as np
import pypty
# NumPy is used for minor preprocessing, while all major operations are handled by PyPty.
print("PyPty version: ", pypty._version_)
```
We will need numpy to do some small preprocessing outside of pypty, but the major steps will be done with pypty functions.

## **_Normal 4D-STEM iterative ptychography_**.

In this section I will go through the steps required to do a ptychographic reconstruction from normal 4D-STEM data (far-field).

### **_Creation of Calibrated Data_**

In a first step we want to create and center a 4D-STEM dataset (in h5 format). PyPty can also accept raw numpy-arrays, but it's better to do this preprocessing step.
```python
path = path_to_your_data
name = name of your dataset
path_raw = path+name+".npy"
path_h5  = path+name+".h5"

pypty.initialize.create_pypty_data(path_raw, path_h5, swap_axes=False,
                    flip_ky=0,flip_kx=0, comcalc_len=200*200,
                    comx=0, comy=0, bin=1, crop_left=None,
                    crop_right=None, crop_top=None, crop_bottom=None, normalize=False, exist_ok=1)
```

### **_Experimental parameters_**

PyPty accepts .json metadata from Nion microscopes, but you can also specify all entries yourself. Here I create a dictionary `experimental_params` with relevant parameters. See the [experiment description](experiment.md) for further details.

```python
path_json="" ## I will leave this string like this and specify everything by hand.
output_folder=your_output_folder

experimental_params={
    'output_folder': output_folder,
    'path_json': path_json, 
    'data_path': path_h5,
    
    ### You can specify one of the following three parameters
    'rez_pixel_size_A': None,
    'rez_pixel_size_mrad': None,
    'conv_semiangle_mrad': 33.8,
    
    
    'scan_size': [256,256], ## this is number of scan points along slow and fast axes
    
    ###  You can specify one of the following two parameters
    'scan_step_A': 0.18, # scan step in Angstrom
    'fov_nm': None,
    
    'acc_voltage': 60, ## kV

    'aberrations': [200,0,0],## Angstrom  [C10, C12a, C12b], can be longer
    
    'PLRotation_deg': 4.65, # deg
    
    
    'bright_threshold': 0.2, ## this parameter will be used to estimate the aperture
    
    'total_thickness': 6, ## Angstrom, thickness of the sample
    'num_slices': 1,  ## number of slices in your reconstruction
    
    'plot': True, ## pypty will plot a few things
    'print_flag': 3, ## max verbosity
    'data_pad': None, ## None will add 1/4 of pixels to each side. This is done to prevent aliasing. You can do any other positive int.
    
    ## now there are 2 nexus tags you can specify. This will help to identify your reconstruction in the Future.
    'chemical_formula': YOUR_Formula, 
    'sample_name': NAME_of_YOUR_Sample,
    
    }
```

### **_Loading a reconstruction preset_**

 
Parameters for iterative ptychography are stored in a dictionary typically named `pypty_params`. 

1) Such presets are typically stored as .pkl data and you can load them via a function in `utils` module

```python
path_to_your_pkl_preset=
pypty_params=pypty.utils.load_preset(path_to_your_pkl_preset)
```

2) Alternatively, you can load them from a nexus data of another reconstruction

```python
path_to_your_nxs_preset=
pypty_params=pypty.utils.load_nexus_params(path_to_your_nxs_preset)
```
3) A third option is to construct your custom preset as a dictionary from scratch (Please see the [guide](custom_presets.md) )

```python
pypty_params={
    "your_entry": something,
    ...
    }
```
4) Finally, you can specify it as `None`, then you will have a standard preset

```python
pypty_params=None
```
### **_Joining a preset with experimental parameters_**

Once you have a preset, you can add experimental data to it.


```python
pypty_params=pypty.initialize.append_exp_params(experimental_params, pypty_params)
```
### **_Starting Ptychography_**

This is done with a single function and single argument, `pypty_params`:

```python
pypty.iterative.run(pypty_params)
```


## **_Saving your Results as one Nexus file_**

Once your reconstruction is done, you will find a bunch of files in the `output_folder`. You can stack them into one big nexus file via a single function. You can also select where the .nxs file will be saved via
`path_to_your_nexus_file`

```python 
pypty.utils.convert_to_nxs(output_folder, path_to_your_nexus_file)
```


## Reconstruction without an extra h5 file

If you do not want to create an extra .h5 file (**not recommended**), you can specify the 4D-STEM data directly in the experimental parameters (before you join them with `pypty_params`). 

```python
experimental_params["dataset"]=dataset
```
Note than in this case the entry `data_path` will be ignored and you can leave this field empty. The dataset itself can be either 4D or 3D (with 2 scan axes merged into one).


## Iterative Ptychography from Virtual Detectors

If you want to perform a reconstruction from data compressed by virtual detectors, you first have to ensure that the array of detectors has shape `[N_detectors, k_Y, k_X]` and your data has shape `[Scan_Y, Scan_X, N_detectors]`. Then you attach these two arrays to experimental parameters.


```python
experimental_params={
    "dataset": compressed_data,
    "masks": virtual_detector_array,
    
    
    ###  In this case you must specify one of the following two parameters
    'rez_pixel_size_A': None,
    'rez_pixel_size_mrad': 1,
    
   # ... Rest of your parameters.
}
```
Also, you may want to update your reconstruction settings. For Compressed data PyPty has following objective function types:

|  Objective_name     |   Description    | 
|---------------------|------------------|
|    `lsq_compressed`               |     Least-squared fit              |
|    `gauss_compressed`               |     Gaussian noise model            |
|    `poisson_compressed`               |     Poissonian noise model            |


Then you must specify `algorithm` as one of these three options in `pypty_params`:
```python
pypty_params={
        "algorithm":  gauss_compressed, 
        #... Rest of your parameters
        }
```
Then you can launch ptychography with the same [pypty.iterative.run](reference/iterative.md) function.
