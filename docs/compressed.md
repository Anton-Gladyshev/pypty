# **_Iterative Ptychography from Virtual Detectors_**

PyPty can deal not only with raw 4D-STEM like data, but also with lossy-compression. If you want to perform a reconstruction from data compressed by virtual detectors, you first have to ensure that the array of detectors has shape `[N_detectors, k_Y, k_X]` and your data has shape `[Scan_Y, Scan_X, N_detectors]`. Then you should attach these two arrays to experimental parameters.


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
        "algorithm": "gauss_compressed", 
        #... Rest of your parameters
        }
```
Then you can launch ptychography with the same [pypty.iterative.run](reference/iterative.md) function.

