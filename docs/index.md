# **_Overview_**

**PyPty** is a **phase retrieval** library for **near-field** and **far-field imaging** in **TEM/STEM**. It supports:

- **Iterative ptychography**
- **Direct ptychography** (Wigner Distribution Deconvolution)
- **Differential phase contrast (DPC)**
- **Tilt-corrected bright field (tcBF)**
- **Focal series reconstructions**
- **LARBED reconstructions**

Developed by Anton Gladyshev (AG SEM, Physics Department, Humboldt-Universität zu Berlin).

---


## **_Installation_**


To create a suitable Python environment for PyPty, you can use **conda**, **mamba**, or **micromamba**. Below are installation instructions using `conda`.

### GPU Installation

```bash
git clone git@github.com:Anton-Gladyshev/pypty.git
cd pypty
conda env create -f pypty_gpu.yml
conda activate pypty
pip install .[gpu]
```

### CPU Installation

```bash
git clone git@github.com:Anton-Gladyshev/pypty.git
cd pypty
conda env create -f pypty_cpu.yml
conda activate pypty
pip install .
```


## About this Code

### Main Purpose

The primary function of PyPty is **gradient-based phase reconstruction** from far-field intensities. This is handled via [`pypty.iterative.run()`](reference/iterative.md). Most other functions are designed to simplify usage of this core function.

A complete list of required arguments for `pypty.iterative.run()` is available in the [custom presets guide](custom_presets.md).

### Logic

PyPty is written in "functional programming" style, but most PyPty functions accept a single input dictionary called `pypty_params`, which typically includes both experimental and reconstruction settings. These experimental parameters are often defined separately in a dictionary called `experimental_params` (see [experiment description](experiment.md)).

Once you’ve created a valid `pypty_params` for one experiment, you can reuse it for others by simply ["attaching"](example_breakdown.md) new experimental parameters.

PyPty also supports various [direct reconstruction methods](examples_direct.md) and tools for [initialization](reference/initialize.md).

### Scaling within one and multiple GPU(-s)

PyPty supports various degrees of parallelization. I would like to highlight that most of operations in phase retrival are trivially parallelizable. Typically one deals with multiple measurements, outcomes of which do not depend on each other. Further, if a measurement can not be described as a perfectly coherent one, one has to repeat multiple similar operations to build a trace of an operator. These two facts are heavily exploited to accelerate the computation in PyPty. Incoherency-parallelization is done automatically. The multiple measurements can be proccessed simultaneously in two different ways. If more than one CUDA device is detected, PyPty will automatically split the dataset in even parts and execute computations concurrently on multiple GPUs. Moreover, one can process measurements in a vectorized fashion on one device. This requieres finding balance between memory usage and computation speed. Please see parameter `compute_batch` in section [custom presets guide](custom_presets.md) for more detailed information.

---

## Input Format

PyPty primarily operates on `.h5` files and provides tools to convert from NumPy. The `.h5` file should contain a dataset named `"data"` with a 3D array: one scan axis and two detector axes.

PyPty also accepts 4D Nion-style NumPy arrays (with accompanying JSON metadata).

---

## Output Format

By default, PyPty saves all output as `.npy` files inside an `output_folder`. The folder structure may look like this:

```text
output_folder
├── cg.npy                              **overwritable checkpoint for scan grid**
├── cp.npy                              **overwritable checkpoint for probe**
├── co.npy                              **overwritable checkpoint for object**
├── checkpoint_obj_epoch_N.npy          **checkpoint for object at epoch N**
├── checkpoint_probe_epoch_N.npy        **checkpoint for probe at epoch N**
├── checkpoint_positions_epoch_N.npy    **checkpoint for scan grid at epoch N**
├── params.pkl                          **parameter file of your reconstruction**
├── loss.csv                            **.csv log-file**
├── tcbf                                **folder  with tcBF results**
│   ├── tcbf_image_upsampling_5.npy     **upsampled tcBF image_**
│   ├── tcbf_image_N.png                **intermediate tcBF image at iteration N in .png format**
│   ├── estimated_shifts_N.npy          **Fitted shifts at iteration N in .npy format**
│   ├── aberrations_N.npy               **Fitted aberrations (in Angstrom) at iteration N in .npy format**
│   ├── PL_angle_deg.npy                **array constaining fitted rotation angles at all iterations in .npy format**
│   ├── aberrations_A.npy               **2d array constaining fitted aberrations at all iterations in .npy format**
└── dpc                                 **folder  with DPC results**
│   ├── idpc.npy                        **fft-based dpc phase**
│   ├── iterative_dpc.npy               **DPC phase (iterative reconstruction)**
└── wdd                                 **folder  with WDD results**
    ├── object.npy                      **complex object (WDD ptychography)**
```

You can also convert an output folder into single Nexus `.nxs` file via [pypty.utils.convert_to_nxs()](reference/utils.md) function.

## Examples

Example workflows are available in the `examples` folder on GitHub.

For a quick start, check out the following:


-  [Step-by-Step Breakdown](example_breakdown.md) — Explains key functions and logic
-  [Custom Presets Guide](custom_presets.md) — For advanced setups
-  [Direct Methods](examples_direct.md) — For Wigner distribution and other direct approaches
-  [Incohrent Ptychography](incoherent.md) - Reconsturcitons with incoherent scattering  
-  [Compressed Ptychography](compressed.md) - Reconsturcitons from compressed data



 
## **_Relevant Literature_**
If you have any questions after reading this guide, the following papers, books and links might explain the working principle of the code:

### Multi-slice formalism and the NN-style approach
1) Earl J. Kirkland. Advanced Computing  in Electron Microscopy 

2) W. Van den Broek and C. Koch. General framework for quantitative three-dimensional reconstruction from arbitrary detection geometries in TEM 

3) W. Van den Broek and C. Koch. Method for Retrieval of the Three-Dimensional Object Potential by Inversion of Dynamical Electron Scattering 

### Error metrics:
#### LSQ:    
4) M. Schloz et al. Overcoming information reduced data and experimentally uncertain parameters in ptychography with regularized optimization 

5) M. Du et al. Adorym: a multi-platform generic X-ray image reconstruction framework based on automatic differentiation 

#### Maximum Likelihood (ML):
6) P. Thibault and M. Guizar-Sicairos. Maximum-likelihood refinement for coherent diffractive imaging 

#### Compressed LSQ:
7) A. Gladyshev et al. Lossy Compression of Electron Diffraction Patterns for Ptychography via Change of Basis

#### lsq_sqrt and lsq_sqrt2:
8) P. Godard et al. (2012). Noise models for low counting rate coherent diffraction imaging

### Mixed state formalism:
9) P Thibault & A.Menzel Reconstructing state mixtures from diffraction measurements— Flux-preserving formalism (for near-field imaging)
### Near-field imaging
10) C. Koch A flux-preserving non-linear inline holography reconstruction algorithm for partially coherent electrons

### Tilted propagator:
11) Earl J. Kirkland. Advanced Computing  in Electron Microscopy 

12) H. She, J. Cui and R. Yu. Deep sub-angstrom resolution imaging by electron ptychography with misorientation correction

### Regularization constaints:
13) M. Schloz et al. Overcoming information reduced data and experimentally uncertain parameters in ptychography with regularized optimization 

14) A. Gladyshev et al. Lossy Compression of Electron Diffraction Patterns for Ptychography via Change of Basis

### Linesearch
15) L. Armijo (1966). Minimization of functions having Lipschitz continuous first partial derivatives
16) P. Wolfe (1969). Convergence Conditions for Ascent Methods


### BFGS algotithm
17) C. G. Broyden (1970). The convergence of a class of double-rank minimization algorithms

18) R. Fletcher   (1970). A New Approach to Variable Metric Algorithms

19) D. Goldfarb (1970). A Family of Variable Metric Updates Derived by Variational Means

20) D. F. Shanno (1970). Conditioning of quasi-Newton methods for function minimization

### Complex derivatives 
21) W. Wirtinger (1927). Zur formalen theorie der funktionen von mehr komplexen veränderlichen. 

