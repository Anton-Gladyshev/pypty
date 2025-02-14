# PyPty v2

# This is a code for phase retrieval. 
It can be applied for near- or far-field imgaging in TEM/STEM, uses mixed-state formalism for the description of the probe, multi-slice formalism to describe multiple scattering events, can refine positions of the probe and uses a modified Fresnel propagator to describe tilted illumination. The code is written by Anton Gladyshev (AG SEM, Physics Department, Humboldt-Universität zu Berlin). 

# The following papers, books and links might help to understand the working principle of the code:

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

## Mixed state formalism:
8) P Thibault & A.Menzel Reconstructing state mixtures from diffraction measurements— Flux-preserving formalism (for near-field imaging)
9) C. Koch A flux-preserving non-linear inline holography reconstruction algorithm for partially coherent electrons

## Tilted propagator:
10) Earl J. Kirkland. Advanced Computing  in Electron Microscopy 
11) H. She, J. Cui and R. Yu. Deep sub-angstrom resolution imaging by electron ptychography with misorientation correction

## Regularization constaints:
12) M. Schloz et al. Overcoming information reduced data and experimentally uncertain parameters in ptychography with regularized optimization 
13) A. Gladyshev et al. Lossy Compression of Electron Diffraction Patterns for Ptychography via Change of Basis

## Linesearch
14) L. Armijo (1966). Minimization of functions having Lipschitz continuous first partial derivatives
