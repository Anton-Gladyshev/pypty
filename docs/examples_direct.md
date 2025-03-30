# **__Examples for direct methods__**

PyPty also offers a set of direct reconstruction methods applicable to 4D-STEM data:

1) [Differential Phase Contrast](reference/dpc.md)

2) [Tilt-Corrected Bright Field](reference/tcbf.md)

3) [Direct Ptychography](reference/direct.md)

Here I will give some usage examples for these methods. First, you need to create a calibrated `pypty_params` dictionary (for more info see [description of experiment](example_breakdown.md), [examples for iterative ptychography](example_breakdown.md) and [API reference](reference/initialize.md)). 

I will assume that you have constructed `experimental_params`.

```python
import pypty
import numpy as np
import matplotlib.pyplot as plt

pypty_params=pypty.initialize.append_exp_params(experimental_params, None)
```

## **__DPC Reconstruction__**

PyPty is supplied with two functions for differential phase contrast. Both of them aim so solve a Poisson equation, i.e. to reconstruct a phase from its laplacian, gradient of the COM vector field. The first method, based on the Fast Fourier Transform (FFT), reconstructs the phase efficiently and can be executed as follows (including visualization using matplotlib)

```python
dpc_phase, pypty_params= pypty.dpc.fft_based_dpc(pypty_params, hpass=1e-1, lpass=0)

plt.imshow(dpc_phase, cmap="gray")
plt.title("WDD Phase")
plt.colorbar()
plt.show()
```
Note that fft-based DPC returns both phase and parameter dictionary as it stores the calculated COM vector field (and the angle between real and reciprocal spaces). If these two quantities are not in `pypty_params`, it attempts to automatically calculate them.

Second method is iterative and accounts for non-periodic boundary conditions:

```python
dpc_phase=pypty.dpc.iterative_dpc(pypty_params, num_iterations=100, beta=0.5, hpass=1e-1, lpass=0, step_size=0.1)
```

Please see the [module](reference/dpc.md) description for detailed description of the parameters. 


## **__tcBF reconstructions__**

In PyPty framework tcBF is done in two steps. First one aims to fit the aberrations of the beam and rotation angle between real and reciprocal spaces by creating an unupsampled tcBF image via `run_tcbf_alignment`:

```
pypty_params=pypty.tcbf.run_tcbf_alignment(
        pypty_params,
        binning_for_fit=np.tile([5], 30), 
        optimize_angle=True,

        cross_corr_type="abs",
        refine_box_dim=5,
        upsample=10,
        cancel_large_shifts=0.9,
        reference_type="bf",
        tol_ctf=1e-8
        )
```

Note that you have to play with binning values and cross-correlation parameters to get an optimal result. Also, fitted rotation angle typically has a +- pi ambiguity, so if your defocus value comes out with a sign different from the experiment, I suggest to rerun the reconstruction.

A second tcBF function assumes that you already did an alignment and performs an upsampled reconstruction:

```python
tcbf_image,tcbf_px_size = pypty.tcbf.upsampled_tcbf(pypty_params, upsample=3,
                                    pad=10, default_float=32,round_shifts=True)
                                    
plt.imshow(tcbf_image,  cmap="gray")
plt.colorbar()
plt.show()
```



## **__WDD Ptychography__**

The last direct reconstruction method is Wigner distribution deconvolution (WDD). It is a direct method that reconstructs the object's complex wavefunction by deconvolving the measured intensity distribution with the known probe. WDD requires a calibrated `pypty_params`dictionary containing rotation angle, scan steps and a complex beam (or a set of aberrations). A simple usage example is

```python
obj_wdd=pypty.direct.wdd(pypty_params,  eps_wiener=1e-3)

plt.imshow(np.angle(obj_wdd), cmap="gray")
plt.title("WDD Phase")
plt.colorbar()
plt.show()
```
The parameter `eps_wiener` serves as a regularization term (high-pass filter) to stabilize the deconvolution, preventing division by zero.


## **__Using direct methods for creation of initial guess__**

[pypty.initialize](reference/initialize.md) module is supplied with a function `get_ptycho_obj_from_scan` that can be used to create complex transmission function based on a measurement sampled at the scan points, this can be a DPC or WDD phase. Usage example is:

```python
pypty_params =  pypty.initialize.get_ptycho_obj_from_scan(pypty_params, array_phase=dpc_phase, array_abs=None)
```

If you have an array on a grid with finer sampling than the actual scan grid of your reconstruction, you may first create an upsampled grid in the image coordinates and use the array for initial guess like this:

```python
image = ## your upsampled image
image_pixel_size = ## pixel size of your image
left_zero_of_scan_grid = # how many pixels of the image are from left of the scan
top_zero_of_scan_grid  = # how many pixels of the image are from top of the scan
scan_array_A=pypty.initialize.get_grid_for_upsampled_image(pypty_params, image, image_pixel_size=image_pixel_size, left_zero_of_scan_grid=left_zero_of_scan_grid, top_zero_of_scan_grid=top_zero_of_scan_grid)

pypty_params =  pypty.initialize.get_ptycho_obj_from_scan(pypty_params, array_phase=image, array_abs=None, scan_array_A=scan_array_A)

```

You can also manually specify the initial guess by providing keys like `"obj"`, `"probe"` or `"positions"`, for more see [Guide for creating custom presets](custom_presets.md).
