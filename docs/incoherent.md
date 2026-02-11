# Incoherent Ptychography

PyPty supports three types of incoherent scattering:

- **Probe modes**: multiple coherent probes
- **Object modes**: multiple transmission functions
- **Static background**: uniform parasitic scattering

Let \(O\) be a coherent object and \(P\) a coherent probe. Define the forward model

\[
FM(O, P)
\],

which maps object and probe to an intensity pattern. The total measured intensity is then

\[
S + \sum_{i,j} FM(O_i, P_j),
\]

where:

- \(S\) is a static offset common to all patterns,
- \(O_i\) are object modes,
- \(P_j\) are probe modes.

## Probe Modes

By default, PyPty generates a single coherent probe mode using a calibrated `pypty_params` dictionary. To use multiple modes, you have three options:

1. **User-defined modes**  
   Provide an array of shape `[Ny, Nx, Nmodes]` or `[Ny, Nx, Nmodes, Nroi]` via the `probe_modes` key.

2. **Defocus spread modes**  
   Set `defocus_spread_modes` in `pypty_params` to a 1D array of defocus values (in Å). PyPty applies different defocus values to generate multiple modes. This can be useful for monolayers or for samples with drift or wobble along the beam propgation direction.

3. **Hermite modes**  
   Specify `n_hermite_probe_modes: (order_x, order_y)` to generate Hermite polynomials up to the given orders in the x and y directions.

## Static Background

PyPty models a static incoherent offset in two ways:

1. **User-defined background**  
   Provide `static_background` as a 2D array matching your diffraction pattern size (including padding).

2. **Automatic initialization**  
   Set `static_background` to a float > 1E-7. PyPty will create a randomized background normalized to the total number counts specified by the provided float.

## Object Modes

Reconstructing incoherent object modes requires a well-calibrated probe and accurate positions. Initialization options include:

1. **User-defined modes**  
   Provide `obj` as a 4D array of shape `(y, x, z, Nmodes)` in `pypty_params` (**after** attached experimental parameters to the preset `pypty_params`).

2. **Automatic initialization**  
   In `experimental_params` (**before** appending them to `pypty_params`), set:
   - `num_obj_modes`: integer number of object modes
   - `obj_phase_sigma`: standard deviation of Gaussian noise for initial phase
   
   Ensure each mode starts from a different random seed to avoid identical updates.
