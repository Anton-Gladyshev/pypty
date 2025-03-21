Module pypty.loss_and_direction
===============================

Functions
---------

`charge_flip(a, delta_phase=0.03, delta_abs=0.14, beta_phase=-0.95, beta_abs=-0.95, fancy_sigma=None)`
:   Perform charge-flipping style object regularization to enhance phase and absorption contrast.
    
    Parameters
    ----------
    a : ndarray
        Complex object to regularize.
    delta_phase : float
        Phase threshold ratio.
    delta_abs : float
        Absorption threshold ratio.
    beta_phase : float
        Inversion multiplier for low-phase regions.
    beta_abs : float
        Inversion multiplier for low-absorption regions.
    fancy_sigma : tuple or None
        Tuple of atomic-shape gaussian kernel sigmas (for phase, absorption).
    
    Returns
    -------
    ndarray
        Regularized complex object.

`clear_missing_wedge(obj, px_size_x_A, px_size_y_A, slice_distance, beta_wedge)`
:   Remove missing wedge artifacts by applying a cone filter in 3D FFT space.
    
    Parameters
    ----------
    obj : ndarray
        3D complex object.
    px_size_x_A : float
        Pixel size along x (Å).
    px_size_y_A : float
        Pixel size along y (Å).
    slice_distance : float
        Distance between slices (Å).
    beta_wedge : float
        Cone sharpness parameter.
    
    Returns
    -------
    ndarray
        Filtered object with reduced missing wedge effects.

`compute_atv_constraint(obj, atv_weight, atv_q, atv_p, pixel_size_x_A, pixel_size_y_A, atv_grad_mask, return_direction, smart_memory)`
:   Apply adaptive total variation (ATV) regularization to the object.
    
    Parameters
    ----------
    obj : ndarray
        Complex object.
    atv_weight : float
        Regularization weight.
    atv_q : float
        q-norm parameter.
    atv_p : float
        p-norm parameter.
    pixel_size_x_A : float
        Pixel size along x (Å).
    pixel_size_y_A : float
        Pixel size along y (Å).
    atv_grad_mask : ndarray or None
        Optional gradient mask.
    return_direction : bool
        If True, return the gradient.
    smart_memory : bool
        If True, use memory-efficient computation.
    
    Returns
    -------
    reg_term : float
        ATV regularization value.
    dR_dTerm : ndarray
        Gradient with respect to the object.

`compute_deformation_constraint_on_grid(something, scan_size, reg_weight)`
:   Penalize deviations from affine transformations in local scan patches.
    
    Parameters
    ----------
    something : ndarray
        Grid to regularize (positions or tilts).
    scan_size : tuple
        Dimensions of the scan.
    reg_weight : float
        Regularization weight.
    
    Returns
    -------
    reg_term : float
        Regularization loss.
    grad : ndarray
        Gradient of the loss with respect to the grid.

`compute_fast_axis_constraint_on_grid(something, scan_size, tv_reg_weight)`
:   Apply second-order TV regularization along the fast scan axis.
    
    Parameters
    ----------
    something : ndarray
        Positions or tilts to regularize.
    scan_size : tuple
        Size of the scan grid.
    tv_reg_weight : float
        Regularization weight.
    
    Returns
    -------
    reg_term : float
        Value of the regularization term.
    grad : ndarray
        Gradient of the regularization.

`compute_full_l1_constraint(object, abs_norm_weight, phase_norm_weight, grad_mask, return_direction, smart_memory)`
:   Apply L1 norm regularization to the object's phase and absorption.
    
    Parameters
    ----------
    object : ndarray
        Complex object array.
    abs_norm_weight : float
        Weight for absorption norm.
    phase_norm_weight : float
        Weight for phase norm.
    grad_mask : ndarray
        Mask to restrict gradient computation.
    return_direction : bool
        If True, return the gradient.
    smart_memory : bool
        Memory-efficient option.
    
    Returns
    -------
    reg_term : float
        Regularization loss.
    grad : ndarray or None
        Gradient if `return_direction` is True, else None.

`compute_missing_wedge_constraint(obj, px_size_x_A, px_size_y_A, slice_distance, beta_wedge, wegde_mu)`
:   Enforce missing wedge constraint in 3D reciprocal space.
    
    Parameters
    ----------
    obj : ndarray
        3D complex object.
    px_size_x_A : float
        Pixel size along x (Å).
    px_size_y_A : float
        Pixel size along y (Å).
    slice_distance : float
        Slice spacing (Å).
    beta_wedge : float
        Cone sharpness.
    wegde_mu : float
        Regularization weight.
    
    Returns
    -------
    loss_term : float
        Regularization loss.
    grad_obj : ndarray
        Gradient of the loss with respect to the object.

`compute_mixed_object_variance_constraint(this_obj, weight, sigma, return_direction, smart_memory)`
:   Regularize variance across object modes by penalizing their differences.
    
    Parameters
    ----------
    this_obj : ndarray
        Complex object array with multiple modes.
    weight : float
        Regularization strength.
    sigma : float
        Smoothing kernel width in frequency space.
    return_direction : bool
        If True, return the gradient.
    smart_memory : bool
        Use memory-efficient FFT loops.
    
    Returns
    -------
    reg_term : float
        Mixed variance loss.
    grad : ndarray or None
        Gradient with respect to the object if `return_direction` is True.

`compute_probe_constraint(to_reg_probe, aperture, weight, return_direction)`
:   Apply reciprocal space constraint to the probe using an aperture mask. Penalize probe values outside an aperture.
    
    Parameters
    ----------
    to_reg_probe : ndarray
        Complex probe array.
    aperture : ndarray or float
        Binary mask or scalar defining aperture radius.
    weight : float
        Regularization weight.
    return_direction : bool
        If True, return the gradient.
    
    Returns
    -------
    reg_term : float
        Loss from masked frequency components.
    probe_fft : ndarray or None
        Gradient of the constraint if requested.

`compute_slow_axis_constraint_on_grid(something, scan_size, tv_reg_weight)`
:   Apply second-order TV regularization along the slow scan axis.
    
    Parameters
    ----------
    something : ndarray
        Positions or tilts to regularize.
    scan_size : tuple
        Size of the scan grid.
    tv_reg_weight : float
        Regularization weight.
    
    Returns
    -------
    reg_term : float
        Regularization loss.
    grad : ndarray
        Gradient with respect to the input.

`compute_window_constraint(to_reg_probe, current_window, current_window_weight)`
:   Penalize probe values outside a predefined window region in real-space.
    
    Parameters
    ----------
    to_reg_probe : ndarray
        Complex probe array.
    current_window : ndarray
        Window mask.
    current_window_weight : float
        Weight of the constraint.
    
    Returns
    -------
    reg_term : float
        Window constraint loss.
    reg_grad : ndarray
        Gradient of the loss with respect to the probe.

`loss_and_direction(this_obj, full_probe, this_pos_array, this_pos_correction, this_tilt_array, this_tilts_correction, this_distances, measured_array, algorithm_type, this_wavelength, this_step_probe, this_step_obj, this_step_pos_correction, this_step_tilts, masks, pixel_size_x_A, pixel_size_y_A, recon_type, Cs, defocus_array, alpha_near_field, damping_cutoff_multislice, smooth_rolloff, propmethod, this_chopped_sequence, load_one_by_one, data_multiplier, data_pad, phase_plate_in_h5, this_loss_weight, data_bin, data_shift_vector, upsample_pattern, static_background, this_step_static_background, tilt_mode, aberration_marker, probe_marker, aberrations_array, compute_batch, phase_only_obj, beam_current, this_beam_current_step, this_step_aberrations_array, default_float, default_complex, xp, is_first_epoch, scan_size, fast_axis_reg_weight_positions, slow_axis_reg_weight_positions, slow_axis_reg_weight_tilts, current_deformation_reg_weight_positions, current_deformation_reg_weight_tilts, fast_axis_reg_weight_tilts, aperture_mask, probe_reg_weight, current_window_weight, current_window, phase_norm_weight, abs_norm_weight, atv_weight, atv_q, atv_p, mixed_variance_weight, mixed_variance_sigma, smart_memory, print_flag)`
:   Compute the total loss and gradients for ptychographic reconstruction.
    
    This is the core function of PyPty that performs forward and backward propagation,
    calculates the loss between measured and simulated patterns, and computes the gradients
    of all active reconstruction parameters (object, probe, positions, tilts, etc.).
    
    Parameters
    ----------
    Params : list
        Way to many to describe right now
        
    Returns
    -------
    loss : float
        Total loss value.
    sse : float
        Sum of squared errors.
    object_grad : ndarray
        Gradient of the loss with respect to the object.
    probe_grad : ndarray
        Gradient of the loss with respect to the probe.
    pos_grad : ndarray
        Gradient of the loss with respect to scan position corrections.
    tilts_grad : ndarray
        Gradient of the loss with respect to tilts.
    static_background_grad : ndarray
        Gradient of the loss with respect to static background.
    aberrations_array_grad : ndarray
        Gradient of the loss with respect to aberration coefficients.
    beam_current_grad : ndarray
        Gradient of the loss with respect to beam current.
    constraint_contributions : list
        Individual regularization loss terms added to the total loss.

`make_basis_orthogonal(vectors)`
:   Orthogonalize a set of 1D basis vectors using Gram-Schmidt.
    
    Parameters
    ----------
    vectors : ndarray
        2D array of vectors to orthogonalize.
    
    Returns
    -------
    ndarray
        Orthogonalized basis.

`make_states_orthogonal(probe_states)`
:   Apply Gram-Schmidt orthogonalization to probe modes.
    
    Parameters
    ----------
    probe_states : ndarray
        Probe array with multiple modes.
    
    Returns
    -------
    ndarray
        Orthogonalized probe states.

`scatteradd_abers(full, indic, batches)`
:   Adds batched aberration updates to their respective positions in the full aberration array. This wrapper is needed to support older CuPy version.
    
    Parameters
    ----------
    full : ndarray
        Full aberration gradient array.
    indic : array_like
        Indices specifying where to add each batch.
    batches : ndarray
        Batched gradients to scatter-add.
    Returns
    -------

`scatteradd_probe(full, indic, batches)`
:   Adds batched probe updates to their respective positions in the full probe array. This wrapper is needed to support older CuPy version.
    
    Parameters
    ----------
    full : ndarray
        Full probe gradient array.
    indic : array_like
        Indices specifying where to add each batch.
    batches : ndarray
        Batched gradients to scatter-add.
    Returns
    -------