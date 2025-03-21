Module pypty.iterative_ptychography
===================================

Functions
---------

`bfgs_update(algorithm_type, this_slice_distances, this_step_probe, this_step_obj, this_step_pos_correction, this_step_tilts, measured_array, this_wavelength, masks, pixel_size_x_A, pixel_size_y_A, phase_norm_weight, abs_norm_weight, stepsize_threshold_low, probe_reg_weight, aperture_mask, recon_type, defocus_array, Cs, alpha_near_field, damping_cutoff_multislice, smooth_rolloff, update_extra_cut, keep_probe_states_orthogonal, do_charge_flip, cf_delta_phase, cf_delta_abs, cf_beta_phase, cf_beta_abs, phase_only_obj, beta_wedge, wolfe_c1_constant, wolfe_c2_constant, atv_weight, atv_q, atv_p, tune_only_probe_phase, propmethod, this_chopped_sequence, load_one_by_one, data_multiplier, data_pad, phase_plate_in_h5, print_flag, this_loss_weight, max_count, reduce_factor, optimism, mixed_variance_weight, mixed_variance_sigma, data_bin, data_shift_vector, smart_memory, default_float, default_complex, default_int, upsample_pattern, this_step_static_background, tilt_mode, fancy_sigma, tune_only_probe_abs, aberration_marker, this_step_aberrations_array, probe_marker, compute_batch, current_window, current_window_weight, dynamically_resize_yx_object, lazy_clean, current_gaussian_filter, current_apply_gaussian_filter_amplitude, this_beam_current_step, xp, remove_fft_cache, is_first_epoch, hist_length, actual_step, fast_axis_reg_weight_positions, fast_axis_reg_weight_tilts, slow_axis_reg_weight_positions, slow_axis_reg_weight_tilts, scan_size, current_deformation_reg_weight_positions, current_deformation_reg_weight_tilts, warnings)`
:   This is one of the core functions of PyPty. It performs updates of all active reconstruction parameters (object, probe, positions, tilts, etc.) via l-BFGS algorithm.
    
    Parameters
    ----------
    Params : list
        Way to many to describe right now
    Returns
    -------
    total_loss : float
        Total loss value.
    this_sse : float
        Sum of squared errors.
    constraint_contributions: list
        List of reg. constraint values for this epoch
    actual_step : float
        Linesearch step that was found during this iteration
    count : int
        Number of linesearch iterations (calls of loss_and_direction) that was required during this epoch
    d_value : float
        Value of the direction derivative at this epoch
    new_d_value : float
        Value of the direction derivative at the newly estimated point
    warnings : string
        Warnings during this epoch

`reset_bfgs_history()`
:   Reset a global variable history_bfgs that contains information about previous steps.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None

`run_ptychography(pypty_params)`
:   Launch iterative ptychographic reconstuction.
    
    Parameters
    ----------
    pypty_params : dict
        Dictionary containing calibrated parameters, including paths and settings for data processing.
        
    Returns
    -------
    None
    
    Notes
    -----
    pypty_params dictionary can be constructed from a predefined preset and a given dataset via append_exp_params() function. Otherwise one can create the  pypty_params dictionary by hand. For more info about creating pypty_params from scratch please reffer to https://github.com/Anton-Gladyshev/pypty/tree/main (there is a .md file lisitng all possible entries). Otherwise contact Anton Gladyshev directly.