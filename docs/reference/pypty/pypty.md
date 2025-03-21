# **_Overview_**

PyPty is a **phase retrieval** code that can be applied to **near-field or far-field imaging** in **TEM/STEM**. It can be applied to do **iterative ptychography**, **direct ptychography** (Wigner distribution deconsvolution), **differential phase contrast**, **tilt-corrected bright field**, **focal series reconstructions** and **LARBED reconstructions**.

The code is written by Anton Gladyshev (AG SEM, Physics Department, Humboldt-Universität zu Berlin). 




# **_Installation_**

## Setting Up the Python Environment and Installing PyPty

To create a proper Python environment and install PyPty, you can use **conda**, **mamba**, or **micromamba**. With **conda**, use:

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


# **_Examples_**
---

The examples will be provided in the `examples` folder. To to configure a **completely custom preset**, please reffer to the next section.
 

# Relevant Literature
If you have any questions after reading this guide, the following papers, books and links might explain the working principle of the code:

## Multi-slice formalism and the NN-style approach
1) Earl J. Kirkland. Advanced Computing  in Electron Microscopy 

2) W. Van den Broek and C. Koch. General framework for quantitative three-dimensional reconstruction from arbitrary detection geometries in TEM 

3) W. Van den Broek and C. Koch. Method for Retrieval of the Three-Dimensional Object Potential by Inversion of Dynamical Electron Scattering 

## Error metrics:
### LSQ:    
4) M. Schloz et al. Overcoming information reduced data and experimentally uncertain parameters in ptychography with regularized optimization 

5) M. Du et al. Adorym: a multi-platform generic X-ray image reconstruction framework based on automatic differentiation 

### Maximum Likelihood (ML):
6) P. Thibault and M. Guizar-Sicairos. Maximum-likelihood refinement for coherent diffractive imaging 

### Compressed LSQ:
7) A. Gladyshev et al. Lossy Compression of Electron Diffraction Patterns for Ptychography via Change of Basis

### lsq_sqrt and lsq_sqrt2:
8) P. Godard et al. (2012). Noise models for low counting rate coherent diffraction imaging

## Mixed state formalism:
9) P Thibault & A.Menzel Reconstructing state mixtures from diffraction measurements— Flux-preserving formalism (for near-field imaging)
## Near-field imaging
10) C. Koch A flux-preserving non-linear inline holography reconstruction algorithm for partially coherent electrons

## Tilted propagator:
11) Earl J. Kirkland. Advanced Computing  in Electron Microscopy 

12) H. She, J. Cui and R. Yu. Deep sub-angstrom resolution imaging by electron ptychography with misorientation correction

## Regularization constaints:
13) M. Schloz et al. Overcoming information reduced data and experimentally uncertain parameters in ptychography with regularized optimization 

14) A. Gladyshev et al. Lossy Compression of Electron Diffraction Patterns for Ptychography via Change of Basis

## Linesearch
15) L. Armijo (1966). Minimization of functions having Lipschitz continuous first partial derivatives
16) P. Wolfe (1969). Convergence Conditions for Ascent Methods


## BFGS algotithm
17) C. G. Broyden (1970). The convergence of a class of double-rank minimization algorithms

18) R. Fletcher   (1970). A New Approach to Variable Metric Algorithms

19) D. Goldfarb (1970). A Family of Variable Metric Updates Derived by Variational Means

20) D. F. Shanno (1970). Conditioning of quasi-Newton methods for function minimization

## Complex derivatives 
21) W. Wirtinger (1927). Zur formalen theorie der funktionen von mehr komplexen veränderlichen. 
