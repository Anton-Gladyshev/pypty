Module pypty.multislice
=======================

Functions
---------

`better_multislice(full_probe, this_obj_chopped, num_slices, n_obj_modes, n_probe_modes, this_distances, this_wavelength, q2, qx, qy, exclude_mask, is_single_dist, this_tan_x, this_tan_y, damping_cutoff_multislice, smooth_rolloff, master_propagator_phase_space, half_master_propagator_phase_space, mask_clean, waves_multislice, wave, default_float, default_complex)`
:   Simulate multislice wave propagation using an additive split-step method (5th order precision with respect to slice thickness).
    
    Parameters
    ----------
    full_probe : ndarray
        Probe wavefunction with shape [N_batch, y,x, modes]
    this_obj_chopped : ndarray
        Object slices with shape [N_batch, y,x, z, modes].
    num_slices : int
        Number of object slices.
    n_obj_modes : int
        Number of object modes.
    n_probe_modes : int
        Number of probe modes.
    this_distances : ndarray
        Slice thicknesses.
    this_wavelength : float
        Electron wavelength.
    q2, qx, qy : ndarray
        Spatial frequency grids.
    exclude_mask : ndarray
        Mask to exclude undesired frequencies.
    is_single_dist : bool
        If True, use the same distance for all slices.
    this_tan_x, this_tan_y : ndarray
        Beam tilts with shape N_batch
    damping_cutoff_multislice : float
        Damping frequency cutoff.
    smooth_rolloff : float
        Rolloff rate for the damping filter.
    master_propagator_phase_space : ndarray or None
        Full propagator in Fourier space (optional).
    half_master_propagator_phase_space : ndarray or None
        Half-step propagator (optional).
    mask_clean : ndarray
        Clean propagation mask.
    waves_multislice : ndarray
        This array contains interediate exit-waves
    wave : ndarray
        This array contains final exit-wave
    default_float, default_complex : dtype
        Numerical types.
    
    Returns
    -------
    waves_multislice : ndarray
        Multislice stack of propagated waves.
    wave : ndarray
        Final exit wave.

`better_multislice_grads(dLoss_dP_out, waves_multislice, this_obj_chopped, object_grad, tilts_grad, is_single_dist, this_distances, exclude_mask, this_wavelength, q2, qx, this_tan_x, qy, this_tan_y, num_slices, n_probe_modes, n_obj_modes, tiltind, this_step_tilts, master_propagator_phase_space, half_master_propagator_phase_space, damping_cutoff_multislice, smooth_rolloff, tilt_mode, compute_batch, mask_clean, masked_pixels_y, masked_pixels_x, default_float, default_complex)`
:   Compute gradients of object, probe, and tilts for the "better_multislice" wave propagation model.
    
    Parameters
    ----------
    dLoss_dP_out : ndarray
        Gradient of the loss with respect to the final exit wave.
    waves_multislice : ndarray
        Stack of intermediate wavefields saved during the forward pass.
    this_obj_chopped : ndarray
        The sliced object with shape [batch, y, x, z, modes].
    object_grad : ndarray
        Gradient accumulator for the object slices.
    tilts_grad : ndarray
        Gradient accumulator for the tilts.
    is_single_dist : bool
        Whether all slices have the same thickness.
    this_distances : ndarray
        Thickness per slice.
    exclude_mask : ndarray
        Frequency mask used in propagation.
    this_wavelength : float
        Wavelength of the probe in Ångströms.
    q2, qx, qy : ndarray
        Spatial frequency grids.
    this_tan_x, this_tan_y : float
        Beam tilt values per batch.
    num_slices : int
        Number of slices in the object.
    n_probe_modes : int
        Number of probe modes.
    n_obj_modes : int
        Number of object modes.
    tiltind : int
        Index for updating `tilts_grad`.
    this_step_tilts : int
        Whether to update tilts (0 = off).
    master_propagator_phase_space : ndarray
        Full propagator for the current slice.
    half_master_propagator_phase_space : ndarray
        Half-step propagator.
    damping_cutoff_multislice : float
        Cutoff for high frequencies in damping.
    smooth_rolloff : float
        Smoothing width for damping filter.
    tilt_mode : int
        Specifies which tilts to optimize.
    compute_batch : int
        Number of scan positions processed in batch.
    mask_clean : ndarray
        FFT mask used to remove unstable frequencies.
    masked_pixels_y, masked_pixels_x : ndarray
        Indices to scatter object gradients into global coordinates.
    default_float : dtype
        Floating point precision.
    default_complex : dtype
        Complex precision.
    
    Returns
    -------
    object_grad : ndarray
        Updated gradient of the object slices.
    interm_probe_grad : ndarray
        Gradient of the probe (summed over object modes).
    tilts_grad : ndarray
        Updated tilt gradients.

`multislice(full_probe, this_obj_chopped, num_slices, n_obj_modes, n_probe_modes, this_distances, this_wavelength, q2, qx, qy, exclude_mask, is_single_dist, this_tan_x, this_tan_y, damping_cutoff_multislice, smooth_rolloff, master_propagator_phase_space, half_master_propagator_phase_space, mask_clean, waves_multislice, wave, default_float, default_complex)`
:   Simulate multislice wave propagation using a classic split-step integrator (2nd order precision with respect to slice thickness if beam is optimized).
    
    Parameters
    ----------
    full_probe : ndarray
        Probe wavefunction with shape [N_batch, y,x, modes]
    this_obj_chopped : ndarray
        Object slices with shape [N_batch, y,x, z, modes].
    num_slices : int
        Number of object slices.
    n_obj_modes : int
        Number of object modes.
    n_probe_modes : int
        Number of probe modes.
    this_distances : ndarray
        Slice thicknesses.
    this_wavelength : float
        Electron wavelength.
    q2, qx, qy : ndarray
        Spatial frequency grids.
    exclude_mask : ndarray
        Mask to exclude undesired frequencies.
    is_single_dist : bool
        If True, use the same distance for all slices.
    this_tan_x, this_tan_y : ndarray
        Beam tilts with shape N_batch
    damping_cutoff_multislice : float
        Damping frequency cutoff.
    smooth_rolloff : float
        Rolloff rate for the damping filter.
    master_propagator_phase_space : ndarray or None
        Full propagator in Fourier space (optional).
    half_master_propagator_phase_space : ndarray or None
        Half-step propagator (optional).
    mask_clean : ndarray
        Clean propagation mask.
    waves_multislice : ndarray
        This array contains interediate exit-waves
    wave : ndarray
        This array contains final exit-wave
    default_float, default_complex : dtype
        Numerical types.
    
    Returns
    -------
    waves_multislice : ndarray
        Multislice stack of propagated waves.
    wave : ndarray
        Final exit wave.

`multislice_grads(dLoss_dP_out, waves_multislice, this_obj_chopped, object_grad, tilts_grad, is_single_dist, this_distances, exclude_mask, this_wavelength, q2, qx, this_tan_x, qy, this_tan_y, num_slices, n_obj_modes, tiltind, master_propagator_phase_space, this_step_tilts, damping_cutoff_multislice, smooth_rolloff, tilt_mode, compute_batch, mask_clean, this_step_probe, this_step_obj, this_step_pos_correction, masked_pixels_y, masked_pixels_x, default_float, default_complex, helper_flag_4)`
:   Compute gradients for classic multislice propagation model (object, probe, and tilts).
    
    
    Parameters
    ----------
    dLoss_dP_out : ndarray
        Gradient of the loss with respect to the final propagated wave.
    waves_multislice : ndarray
        Intermediate wave stack from the forward multislice pass.
    this_obj_chopped : ndarray
        4D sliced object [batch, y, x, z, modes].
    object_grad : ndarray
        Gradient accumulator for object slices.
    tilts_grad : ndarray
        Accumulator for tilt gradients.
    is_single_dist : bool
        If True, slice distances are constant.
    this_distances : ndarray
        Per-slice thicknesses.
    exclude_mask : ndarray
        Frequency mask for FFT operations.
    this_wavelength : float
        Probe wavelength (Å).
    q2, qx, qy : ndarray
        Spatial frequency grids.
    this_tan_x, this_tan_y : float
        Beam tilt values per batch.
    num_slices : int
        Number of slices.
    n_obj_modes : int
        Number of object modes.
    tiltind : int
        Index in tilt update array.
    master_propagator_phase_space : ndarray
        Full Fourier propagation kernel.
    this_step_tilts : int
        Whether tilt gradient is updated.
    damping_cutoff_multislice : float
        Damping cutoff for high-frequency noise.
    smooth_rolloff : float
        Width of damping transition.
    tilt_mode : int
        Mode selector for tilt optimization.
    compute_batch : int
        Current batch size.
    mask_clean : ndarray
        FFT domain mask.
    this_step_probe : int
        Whether to compute probe gradient.
    this_step_obj : int
        Whether to compute object gradient.
    this_step_pos_correction : int
        (Unused) Flag for positional corrections.
    masked_pixels_y, masked_pixels_x : ndarray
        Indices for applying gradients to global object.
    default_float : dtype
        Floating-point type.
    default_complex : dtype
        Complex type.
    helper_flag_4 : bool
        If True, return probe gradient; else return None.
    
    Returns
    -------
    object_grad : ndarray
        Gradient for object slices.
    interm_probe_grad : ndarray or None
        Gradient for input probe (if helper_flag_4 is True).
    tilts_grad : ndarray
        Updated tilt gradient.

`scatteradd(full, masky, maskx, chop)`
:   Adds batched object updates to their respective positions in the full object array.
    This wrapper is needed to support older CuPy versions.
    
    Parameters
    ----------
    full : ndarray
        Full object gradient array.
    masky : ndarray
        Index array for the y-axis.
    maskx : ndarray
        Index array for the x-axis.
    chop : ndarray
        Batched gradients to scatter-add.
    
    Returns
    -------
    None

`yoshida_multislice(full_probe, this_obj_chopped, num_slices, n_obj_modes, n_probe_modes, this_distances, this_wavelength, q2, qx, qy, exclude_mask, is_single_dist, this_tan_x, this_tan_y, damping_cutoff_multislice, smooth_rolloff, master_propagator_phase_space, half_master_propagator_phase_space, mask_clean, waves_multislice, wave, default_float, default_complex)`
:   Simulate multislice wave propagation using an yoshida integrator (5th order precision with respect to slice thickness).
    
    Parameters
    ----------
    full_probe : ndarray
        Probe wavefunction with shape [N_batch, y,x, modes]
    this_obj_chopped : ndarray
        Object slices with shape [N_batch, y,x, z, modes].
    num_slices : int
        Number of object slices.
    n_obj_modes : int
        Number of object modes.
    n_probe_modes : int
        Number of probe modes.
    this_distances : ndarray
        Slice thicknesses.
    this_wavelength : float
        Electron wavelength.
    q2, qx, qy : ndarray
        Spatial frequency grids.
    exclude_mask : ndarray
        Mask to exclude undesired frequencies.
    is_single_dist : bool
        If True, use the same distance for all slices.
    this_tan_x, this_tan_y : ndarray
        Beam tilts with shape N_batch
    damping_cutoff_multislice : float
        Damping frequency cutoff.
    smooth_rolloff : float
        Rolloff rate for the damping filter.
    master_propagator_phase_space : ndarray or None
        Full propagator in Fourier space (optional).
    half_master_propagator_phase_space : ndarray or None
        Half-step propagator (optional).
    mask_clean : ndarray
        Clean propagation mask.
    waves_multislice : ndarray
        This array contains interediate exit-waves
    wave : ndarray
        This array contains final exit-wave
    default_float, default_complex : dtype
        Numerical types.
    
    Returns
    -------
    waves_multislice : ndarray
        Multislice stack of propagated waves.
    wave : ndarray
        Final exit wave.

`yoshida_multislice_grads(dLoss_dP_out, waves_multislice, this_obj_chopped, object_grad, tilts_grad, is_single_dist, this_distances, exclude_mask, this_wavelength, q2, qx, this_tan_x, qy, this_tan_y, num_slices, n_probe_modes, n_obj_modes, tiltind, this_step_tilts, master_propagator_phase_space, half_master_propagator_phase_space, damping_cutoff_multislice, smooth_rolloff, tilt_mode, compute_batch, mask_clean, masked_pixels_y, masked_pixels_x, default_float, default_complex)`
:   Compute gradients for object, probe, and tilt parameters using Yoshida multislice propagation.
    
    
    Parameters
    ----------
    dLoss_dP_out : ndarray
        Gradient of the loss with respect to the output wave.
    waves_multislice : ndarray
        Stored intermediate wavefields from forward Yoshida multislice pass.
    this_obj_chopped : ndarray
        Object slices with shape [batch, y, x, z, modes].
    object_grad : ndarray
        Gradient buffer for object update.
    tilts_grad : ndarray
        Gradient buffer for tilt update.
    is_single_dist : bool
        Whether slice distances are constant.
    this_distances : ndarray
        Thickness of each slice.
    exclude_mask : ndarray
        FFT mask for excluding unstable frequencies.
    this_wavelength : float
        Probe wavelength in Ångströms.
    q2, qx, qy : ndarray
        FFT spatial frequency grids.
    this_tan_x, this_tan_y : float
        Beam tilts (tangent of the angle).
    num_slices : int
        Number of object slices.
    n_probe_modes : int
        Number of probe modes.
    n_obj_modes : int
        Number of object modes.
    tiltind : int
        Index of current tilt in `tilts_grad`.
    this_step_tilts : int
        Whether tilt updates are enabled.
    master_propagator_phase_space : ndarray
        Full propagation kernel in Fourier domain.
    half_master_propagator_phase_space : ndarray
        Half-step propagation kernel.
    damping_cutoff_multislice : float
        Frequency cutoff for damping high frequencies.
    smooth_rolloff : float
        Rolloff profile width for damping.
    tilt_mode : int
        Determines which tilt parameters to optimize.
    compute_batch : int
        Number of scan points in the current batch.
    mask_clean : ndarray
        Fourier domain mask to stabilize calculations.
    masked_pixels_y, masked_pixels_x : ndarray
        Indices for inserting gradients into global object.
    default_float : dtype
        Float precision.
    default_complex : dtype
        Complex precision.
    
    Returns
    -------
    object_grad : ndarray
        Updated object gradient.
    interm_probe_grad : ndarray
        Gradient of the probe (combined over object modes).
    tilts_grad : ndarray
        Updated tilt gradient.