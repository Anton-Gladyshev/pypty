import numpy as np
import sys
import os
import h5py
import time
import random
import pickle
import types
import copy
import inspect
from pypty import fft as pyptyfft
from pypty import utils as pyptyutils
from pypty import objective as pyptyloss_and_direction
try:
    import cupyx.scipy.ndimage as ndi
    import cupy as cp
except:
    import scipy.ndimage as ndi
    import numpy as cp
import matplotlib.pyplot as plt




history_bfgs, obj,probe, positions, positions_correction, tilts, tilts_correction, static_background, aberrations_array, beam_current, pool, pinned_pool = None, None, None, None, None, None,None, None,None, None, None, None
def run(pypty_params):
    """
    Launch iterative reconstuction.
    
    Parameters
    ----------
    pypty_params : dict
        Dictionary containing calibrated parameters, including paths and settings for data processing.
    Notes
    -----
    pypty_params dictionary can be constructed from a predefined preset and a given dataset via append_exp_params() function. Full list of expected entries can be found in the documentation,
    """
    global obj, probe, pool, pinned_pool, positions, positions_correction, tilts, tilts_correction, beam_current,aberrations_array, history_bfgs
    obj, probe, positions, positions_correction, tilts, tilts_correction, static_background, aberrations_array, beam_current = None, None, None, None, None, None,None, None, None
    reset_bfgs_history()
    try:
        pool=cp.get_default_memory_pool()
        pinned_pool=cp.get_default_pinned_memory_pool()
    except:
        pass
    if type(pypty_params)==str:
        params=pyptyutils.load_params(pypty_params)
    else:
        params=pypty_params.copy()
    xp =  params.get('backend', cp) ## currently not used, but usefull for future:
    default_dtype=params.get('default_dtype', "double")
    if default_dtype=="double":
        default_float, default_complex, default_int, default_int_cpu, default_float_cpu, default_complex_cpu=xp.float64, xp.complex128, xp.int32, np.int32, np.float64, np.complex128
    if default_dtype=="single":
        default_float, default_complex, default_int, default_int_cpu, default_float_cpu, default_complex_cpu=xp.float32, xp.complex64, xp.int16, np.int16, np.float32, np.complex64
    if default_dtype=="half":
        default_float, default_complex, default_int, default_int_cpu, default_float_cpu, default_complex_cpu=xp.float16, xp.complex64, xp.int8, np.int8, np.float16, np.complex64
    ## Dataset
    data_path = params.get('data_path', "")
    dataset = params.get('dataset', None)
    masks = params.get('masks', None)
    data_multiplier = default_float_cpu(params.get('data_multiplier', 1))
    data_pad = int(params.get('data_pad', 0))
    data_bin=int(params.get('data_bin', 1))
    flip_ky=params.get('flip_ky', False)
    
    data_shift_vector=params.get('data_shift_vector', [0,0])
    upsample_pattern=params.get('upsample_pattern',1)
    sequence = params.get('sequence', None)
    use_full_FOV = params.get(' ', True)
    ## Saving and printing
    output_folder = params.get('output_folder', "")
    strip_dataset_from_params= params.get('strip_dataset_from_params', True)
    save_loss_log = params.get('save_loss_log', 1)
    epoch_prev = int(params.get('epoch_prev', 0))
    data_simulation_flag = params.get('data_simulation_flag', False)

    pyptyutils.prepare_saving_stuff(output_folder, save_loss_log, epoch_prev)
    if output_folder[-1]!="/": output_folder+="/";
    pyptyutils.save_params(output_folder+"params.pkl", params, strip_dataset_from_params) ### save the params
    pyptyutils.print_pypty_header(data_path, output_folder, save_loss_log)
    params=pyptyutils.string_params_to_usefull_params(params) ### here we want to convert some possible strings that may look like 'lambda x: x>1' into real functions
    save_checkpoints_every_epoch = params.get('save_checkpoints_every_epoch', False)
    save_inter_checkpoints = params.get('save_inter_checkpoints', True)
    print_flag = params.get('print_flag', 3)
    #-------Print where everything will be saved --------------
    ## Experimental parameters
    acc_voltage = default_float_cpu(params.get('acc_voltage', 60))
    aperture_mask = params.get('aperture_mask', None)
    recon_type = params.get('recon_type', "far_field")
    ## only for Near field:
    alpha_near_field = default_float_cpu(params.get('alpha_near_field', 0))
    defocus_array = params.get('defocus_array', np.array([0.0])).astype(default_float_cpu)
    Cs = default_float_cpu(params.get('Cs', 0))
    ## spatial callibration of the object
    slice_distances = params.get('slice_distances', np.array([10]))
    pixel_size_x_A = default_float_cpu(params.get('pixel_size_x_A', 1))
    pixel_size_y_A = default_float_cpu(params.get('pixel_size_y_A', 1))
    scan_size= params.get('scan_size', None)
    num_slices=params.get('num_slices', 1)
    ### ptycho stuff
    obj = params.get('obj', np.ones((1, 1, num_slices, 1))).astype(default_complex_cpu)
    probe = params.get('probe', None)
    positions = params.get('positions', np.array([[0.0, 0.0]])).astype(default_float_cpu)
    tilts = params.get('tilts', np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).astype(default_float_cpu) ### tilts are the tilt before specimen (slope in real space), tilts in the specimen (slope in reciprocal space) and tilts after the specimen (slope in real space)
    tilt_mode=params.get('tilt_mode', 0) ## tilt mode 0: tilt only inside of the specimen, 1 tilt only after the specimen; 2 for tilt before and after the specimen;  3 tilt inside and after the specimen; 4 is for tilting before, inside and after;
    static_background=params.get('static_background', 0)
    beam_current=params.get('beam_current', None)
  
    ## Propagation method, windowing and dinamic resizing
    propmethod = params.get('propmethod', "multislice")
    allow_subPixel_shift = params.get('allow_subPixel_shift', True)
    dynamically_resize_yx_object=params.get('dynamically_resize_yx_object', False)
    extra_space_on_side_px= int(params.get('extra_space_on_side_px', 0))
    skip_preprocessing = params.get('skip_preprocessing', False)
    
    ## Bandwidth limitation
    damping_cutoff_multislice = default_float_cpu(params.get('damping_cutoff_multislice', 2/3))
    smooth_rolloff=default_float_cpu(params.get('smooth_rolloff', 0))
    update_extra_cut=default_float_cpu(params.get('update_extra_cut', 0.005))
    lazy_clean=params.get('lazy_clean', False)
    
    ## optimization settings
    algorithm = params.get('algorithm', "lsq_sqrt")
    epoch_max = int(params.get('epoch_max', 200))
    wolfe_c1_constant = params.get('wolfe_c1_constant', 0.1)
    wolfe_c2_constant=params.get('wolfe_c2_constant', 0.9)
    loss_weight = params.get('loss_weight', 1)
    max_count = params.get('max_count', 20)
    reduce_factor = default_float_cpu(params.get('reduce_factor', 0.1))
    optimism = params.get('optimism', 3)
    min_step = params.get('min_step', 1e-20)
    
    hist_length=params.get('hist_length', 10)
    update_step_bfgs=params.get('update_step_bfgs', 1)
    phase_only_obj = params.get('phase_only_obj', False)
    tune_only_probe_phase = params.get('tune_only_probe_phase', False)
    tune_only_probe_abs=params.get('tune_only_probe_abs', False)
    
    reset_history_flag=params.get('reset_history_flag', False)

    update_probe = params.get('update_probe', 1)
    update_obj = params.get('update_obj', 1)
    update_probe_pos = params.get('update_probe_pos', 0)
    update_tilts = params.get('update_tilts', 0)
    update_beam_current = params.get('update_beam_current', 0)
    update_aberrations_array= params.get('update_aberrations_array', 0)
    update_static_background=params.get('update_static_background', 0)
    
    
    ## Multiple illumination functions for different measurments (Not mixed probe!!!)
    aberrations_array = params.get('aberrations_array',  np.array([[0.0]]))
    phase_plate_in_h5 = params.get('phase_plate_in_h5', None)
    aberration_marker = params.get('aberration_marker', None)
    probe_marker =  params.get('probe_marker', None)
    
    ## Memory usage
    memory_saturation= params.get('memory_saturation', 0.7)
    load_one_by_one= params.get('load_one_by_one', True)
    smart_memory = params.get('smart_memory', True)
    remove_fft_cache = params.get('remove_fft_cache', False)
    compute_batch=params.get('compute_batch', "auto")
    force_dataset_dtype=params.get('force_dataset_dtype', default_float_cpu)
    preload_to_cpu=params.get('preload_to_cpu', False)
    force_pad=params.get('force_pad', False)
    
    ## Constraints
    mixed_variance_weight = params.get('mixed_variance_weight', 0)
    mixed_variance_sigma = params.get('mixed_variance_sigma', 0.5)
    probe_constraint_mask=params.get('probe_constraint_mask', None)
    probe_reg_constraint_weight = params.get('probe_reg_constraint_weight', 0)
    window_weight= params.get('window_weight', 0)
    window = params.get('window', None) ## to do for bfgs!!!

    abs_norm_weight = params.get('abs_norm_weight', 0)
    phase_norm_weight = params.get('phase_norm_weight', 0)
    atv_weight = params.get('atv_weight', 0)
    atv_q = params.get('atv_q', 1)
    atv_p = params.get('atv_p', 2)
    wedge_mu = params.get('wedge_mu', 0)
    fast_axis_reg_weight_positions = params.get('fast_axis_reg_weight_positions', 0)
    slow_axis_reg_weight_positions = params.get('slow_axis_reg_weight_positions', 0)
    fast_axis_reg_weight_tilts = params.get('fast_axis_reg_weight_tilts', 0)
    slow_axis_reg_weight_tilts = params.get('slow_axis_reg_weight_positions', 0)
    deformation_reg_weight_positions= params.get('deformation_reg_weight_positions', 0)
    deformation_reg_weight_tilts= params.get('deformation_reg_weight_tilts', 0)
    beta_wedge = params.get('beta_wedge', 0) ## to do for bfgs!!!

    
    
    # Constraints that modify the object and probe 'by hand'
    apply_gaussian_filter=params.get('apply_gaussian_filter', False)
    apply_gaussian_filter_amplitude=params.get('apply_gaussian_filter_amplitude', False)
    
  
    keep_probe_states_orthogonal = params.get('keep_probe_states_orthogonal', False) ## to do for bfgs!!!
    
    do_charge_flip = params.get('do_charge_flip', False)
    cf_delta_phase = params.get('cf_delta_phase', 0.1)
    cf_delta_abs = params.get('cf_delta_abs', 0.01)
    cf_beta_phase = params.get('cf_beta_phase', -0.95)
    cf_beta_abs = params.get('cf_beta_abs', -0.95)
    fancy_sigma=params.get('fancy_sigma', None)
    restart_from_vacuum=params.get('restart_from_vacuum', False)
    reset_positions=params.get('reset_positions', False)
    ## added for puzzleing
    puzzle_positions=params.get('puzzle_positions', None) 
    ### beam initialisation
    n_hermite_probe_modes=params.get('n_hermite_probe_modes', None)
    defocus_spread_modes=params.get('defocus_spread_modes', None)
    aberrations = params.get('aberrations', None)
    if not(aberrations is None):
        aberrations=np.array(aberrations)
    extra_probe_defocus = default_float_cpu(params.get('extra_probe_defocus', 0))
    estimate_aperture_based_on_binary=params.get('estimate_aperture_based_on_binary', False)
    beam_ctf=params.get('beam_ctf', None)
    mean_pattern=params.get('mean_pattern',None)
    params['pypty_version']="2.1"
    ############################### done with params, starts other things ###################################
    ### get the data
    if not(masks is None):
        masks, *_=pyptyutils.preprocess_dataset(masks, False, "lsq_sqrt", recon_type, data_shift_vector, data_bin, data_pad, upsample_pattern, 1, np, True)
    if dataset is None:
        if data_path[-2:]=="h5":
            this_file=h5py.File(data_path, "r")
            dataset=this_file['data']
            if preload_to_cpu:
                dataset=np.array(dataset).astype(force_dataset_dtype)
        else:
            dataset=np.load(data_path).astype(force_dataset_dtype)
            if len(dataset.shape)==4:
                dataset=dataset.reshape(dataset.shape[0]*dataset.shape[1], dataset.shape[2], dataset.shape[3])
            if flip_ky:
                dataset=dataset[:,::-1,:]
            dataset, data_shift_vector, data_bin, data_pad, data_multiplier = pyptyutils.preprocess_dataset(dataset, False, algorithm, recon_type, data_shift_vector, data_bin, data_pad, upsample_pattern, data_multiplier, np, False)
    else:
        dataset=np.array(dataset).astype(force_dataset_dtype)
        
    measured_data_shape=dataset.shape
    probe=pyptyutils.create_probe_from_nothing(probe, data_pad, mean_pattern, aperture_mask, tilt_mode, tilts, dataset, estimate_aperture_based_on_binary, pixel_size_x_A, acc_voltage, data_multiplier, masks, data_shift_vector, data_bin, upsample_pattern, default_complex_cpu, print_flag, algorithm, measured_data_shape, obj.shape[-1], probe_marker, recon_type, defocus_array, Cs, skip_preprocessing) ### create probe from nothing
    static_background=pyptyutils.create_static_background_from_nothing(static_background, probe, damping_cutoff_multislice,data_pad,upsample_pattern,  default_float_cpu, recon_type) ## initializing static background
    obj, positions, t, sequence, wavelength, positions_correction, tilts_correction, aperture_mask = pyptyutils.prepare_main_loop_params(algorithm,probe, obj,positions,tilts, measured_data_shape, acc_voltage, allow_subPixel_shift, sequence, use_full_FOV, print_flag, default_float_cpu, default_complex_cpu, default_int_cpu, probe_constraint_mask, aperture_mask, extra_space_on_side_px, skip_preprocessing)  # now the we will initilize the object in this function (create from nothing if needed and pad an existing one if needed)
    try:
        obj, probe, positions,positions_correction, tilts, tilts_correction, masks, defocus_array, slice_distances, aperture_mask, dataset, static_background, aberrations_array, beam_current=pyptyutils.try_to_gpu(obj, probe, positions,positions_correction, tilts, tilts_correction, masks, defocus_array, slice_distances, aperture_mask, dataset, load_one_by_one, static_background, aberrations_array, beam_current, default_float, default_complex, default_int, xp) ##Convert numpy arrays to cupy arrays
    except:
        try:
            pool.free_all_blocks()
            pinned_pool.free_all_blocks()
        except:
            pass
        load_one_by_one=True
        smart_memory=True
        sys.stdout.write("\nWARNING: load one by one was forced to be True (not enough memory)!")
        sys.stdout.flush()
        obj, probe, positions,positions_correction, tilts, tilts_correction, masks, defocus_array, slice_distances, aperture_mask, dataset, static_background, aberrations_array, beam_current=pyptyutils.try_to_gpu(obj, probe, positions,positions_correction, tilts, tilts_correction, masks, defocus_array, slice_distances, aperture_mask, dataset, load_one_by_one, static_background, aberrations_array, beam_current, default_float, default_complex, default_int, xp) ##Convert numpy arrays to cupy arrays
    probe=pyptyutils.apply_probe_modulation(probe, extra_probe_defocus, acc_voltage, pixel_size_x_A, pixel_size_y_A, aberrations, print_flag, beam_ctf, n_hermite_probe_modes, defocus_spread_modes, probe_marker, default_complex, default_float, xp) #Here we will apply aberrations to an existing beam and create multiple modes
    probe=pyptyutils.fourier_clean(probe, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=xp) # clean the beam and object just to be on a safe side
    if compute_batch=="auto":
        try:
            history_size=hist_length(0)
        except:
            history_size=hist_length
        compute_batch, load_one_by_one, smart_memory = pyptyutils.get_compute_batch(compute_batch, load_one_by_one, history_size, measured_data_shape, memory_saturation, smart_memory, data_pad, obj.shape, probe.shape, default_dtype, propmethod, print_flag)
        
    try:
        first_smart_memory=smart_memory(0)
        first_smart_memory=True
    except:
        first_smart_memory=smart_memory
    if first_smart_memory:
        for i in range(obj.shape[2]):
            for j in range(obj.shape[3]):
                obj[:,:,i,j]=pyptyutils.fourier_clean(obj[:,:,i,j], cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=xp)
    else:
        obj = pyptyutils.fourier_clean(obj, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=xp)
    try:
        if remove_fft_cache:
            cp.fft.config.clear_plan_cache()
        pool.free_all_blocks()
        pinned_pool.free_all_blocks()
    except:
        pass
    pyptyutils.save_current_checkpoint_obj_probe(output_folder, obj, probe, tilts_correction, positions_correction,positions, tilts, static_background, 1,1,1,0,0, 0, aberrations_array,beam_current, 0, xp)
    dataset, data_shift_vector, data_bin, data_pad, data_multiplier = pyptyutils.preprocess_dataset(dataset, load_one_by_one, algorithm, recon_type, data_shift_vector, data_bin, data_pad, upsample_pattern, data_multiplier, xp, force_pad)
    #######----------------------------------------------------------------------------------------------------------------------------------
    #######------------------------------------------ HERE is the begin of the actual ptychography ------------------------------------------
    #######----------------------------------------------------------------------------------------------------------------------------------
    t0=time.time()
    is_first_epoch=True ## Just a flag that will enable recomputing of propagation- & shift- meshgrids
    constratins_prev, prev_steps_sum=0,0
    if not(reset_positions is None ): initial_postions, initial_postions_correction=1*positions, 1*positions_correction
    for epoch in range(epoch_prev,epoch_max,1):
        current_data_simulation_flag, current_reset_positions, current_restart_from_vacuum, this_reset_history_flag, this_smart_memory, current_wolfe_c1_constant,current_wolfe_c2_constant, current_window_weight, current_hist_length, current_deformation_reg_weight_tilts, current_deformation_reg_weight_positions, current_slow_axis_reg_weight_positions, current_slow_axis_reg_weight_tilts, current_fast_axis_reg_weight_positions,current_fast_axis_reg_weight_tilts, current_update_step_bfgs, current_apply_gaussian_filter_amplitude, current_apply_gaussian_filter, current_keep_probe_states_orthogonal, current_loss_weight, current_phase_norm_weight, current_abs_norm_weight, current_probe_reg_constraint_weight, current_do_charge_flip, current_atv_weight, current_wedge_mu, current_beta_wedge, current_tune_only_probe_phase, current_mixed_variance_weight,current_mixed_variance_sigma, current_phase_only_obj, current_tune_only_probe_abs, current_dynamically_resize_yx_object, current_beam_current_step, current_probe_step, current_obj_step, current_probe_pos_step, current_tilts_step, current_static_background_step, current_aberrations_array_step,current_puzzle_positions =               pyptyutils.get_value_for_epoch([data_simulation_flag, reset_positions, restart_from_vacuum, reset_history_flag, smart_memory, wolfe_c1_constant,wolfe_c2_constant, window_weight, hist_length, deformation_reg_weight_tilts, deformation_reg_weight_positions, slow_axis_reg_weight_positions, slow_axis_reg_weight_tilts, fast_axis_reg_weight_positions,fast_axis_reg_weight_tilts, update_step_bfgs, apply_gaussian_filter_amplitude, apply_gaussian_filter, keep_probe_states_orthogonal, loss_weight, phase_norm_weight, abs_norm_weight, probe_reg_constraint_weight, do_charge_flip, atv_weight, wedge_mu, beta_wedge, tune_only_probe_phase, mixed_variance_weight,mixed_variance_sigma, phase_only_obj, tune_only_probe_abs, dynamically_resize_yx_object, update_beam_current, update_probe, update_obj, update_probe_pos, update_tilts, update_static_background, update_aberrations_array,puzzle_positions], epoch, default_float_cpu) ## here we get the values of constraints for this epoch
        warnings=""
        if current_loss_weight=="mean": current_loss_weight=1/measured_data_shape[0];
        if current_loss_weight=="mean_full": current_loss_weight=1/np.prod(measured_data_shape);
        constratins_new, new_steps_sum=0,0
        for c in [current_slow_axis_reg_weight_positions, current_slow_axis_reg_weight_tilts, current_reset_positions, current_restart_from_vacuum, current_window_weight, current_deformation_reg_weight_tilts, current_deformation_reg_weight_positions, current_fast_axis_reg_weight_positions,current_fast_axis_reg_weight_tilts, current_apply_gaussian_filter_amplitude, current_apply_gaussian_filter, current_keep_probe_states_orthogonal, current_loss_weight, current_phase_norm_weight, current_abs_norm_weight, current_probe_reg_constraint_weight, current_do_charge_flip, current_atv_weight, current_wedge_mu, current_beta_wedge, current_tune_only_probe_phase, current_mixed_variance_weight,current_mixed_variance_sigma, current_phase_only_obj, current_tune_only_probe_abs]:
            constratins_new+= c!=0
        for st in [current_beam_current_step, current_probe_step, current_obj_step, current_probe_pos_step, current_tilts_step, current_static_background_step, current_aberrations_array_step]:
            new_steps_sum+=st!=0
        if epoch!=0 and (constratins_new!=constratins_prev or this_reset_history_flag or new_steps_sum<prev_steps_sum):
            reset_bfgs_history();
            if print_flag:
                sys.stdout.write("\nResetting the history!") ## reset the history if the objective function is changing
                warnings+="\nResetting the history!"
                sys.stdout.flush()
        constratins_prev=constratins_new
        prev_steps_sum=new_steps_sum
        try:
            full_sequence=sequence(epoch)
        except:
            full_sequence=sequence
        if current_restart_from_vacuum:
            obj=xp.ones_like(obj)
            reset_bfgs_history()
        if current_reset_positions:
            positions           =   1*initial_postions
            positions_correction =   1*initial_postions_correction
        count, save_flag=0, False ## count for measurements
        if save_checkpoints_every_epoch: save_flag= epoch%save_checkpoints_every_epoch==0;
        if current_window_weight>0:
            if len(window)==2:
                this_window=pyptyutils.get_window(probe.shape[0], window[0], window[1])
            else:
                this_window=xp.asarray(window)
        else:
            this_window=None
        if current_beam_current_step: beam_current=pyptyutils.try_to_initialize_beam_current(beam_current,measured_data_shape, default_float, xp);
        full_sequence=np.sort(np.array(full_sequence))
        current_loss, current_sse, constraint_contributions, actual_step, count_linesearch, d_value, new_d_value, warnings =  bfgs_update(algorithm, slice_distances, current_probe_step, current_obj_step, current_probe_pos_step,current_tilts_step, dataset, wavelength, masks, pixel_size_x_A, pixel_size_y_A, current_phase_norm_weight, current_abs_norm_weight, min_step, current_probe_reg_constraint_weight,aperture_mask, recon_type, defocus_array, Cs, alpha_near_field, damping_cutoff_multislice, smooth_rolloff, update_extra_cut,  current_keep_probe_states_orthogonal, current_do_charge_flip,cf_delta_phase, cf_delta_abs, cf_beta_phase, cf_beta_abs, current_phase_only_obj, current_wedge_mu, current_beta_wedge, current_wolfe_c1_constant, current_wolfe_c2_constant, current_atv_weight, atv_q, atv_p, current_tune_only_probe_phase, propmethod, full_sequence, load_one_by_one, data_multiplier,data_pad, phase_plate_in_h5, print_flag, current_loss_weight, max_count, reduce_factor, optimism, current_mixed_variance_weight, current_mixed_variance_sigma, data_bin, data_shift_vector, this_smart_memory, default_float, default_complex, default_int, upsample_pattern, current_static_background_step, tilt_mode, fancy_sigma, current_tune_only_probe_abs, aberration_marker, current_aberrations_array_step, probe_marker, compute_batch, this_window, current_window_weight, current_dynamically_resize_yx_object, lazy_clean, current_apply_gaussian_filter, current_apply_gaussian_filter_amplitude, current_beam_current_step, xp, remove_fft_cache, is_first_epoch, current_hist_length, current_update_step_bfgs, current_fast_axis_reg_weight_positions, current_fast_axis_reg_weight_tilts, current_slow_axis_reg_weight_positions, current_slow_axis_reg_weight_tilts, scan_size, current_deformation_reg_weight_positions, current_deformation_reg_weight_tilts, warnings, current_data_simulation_flag,current_puzzle_positions,allow_subPixel_shift) ## update
        is_first_epoch=False
        pyptyutils.print_recon_state(t0, algorithm, epoch, current_loss, current_sse, current_obj_step, current_probe_step,current_probe_pos_step,current_tilts_step, current_static_background_step, current_aberrations_array_step, current_beam_current_step, current_hist_length, print_flag)
        if save_inter_checkpoints!=0:
            if epoch%save_inter_checkpoints==0: pyptyutils.save_current_checkpoint_obj_probe(output_folder, obj, probe, tilts_correction, positions_correction,positions, tilts, static_background, current_probe_step, current_obj_step, current_probe_pos_step, current_tilts_step, current_static_background_step, current_aberrations_array_step, aberrations_array, beam_current, current_beam_current_step, xp);
        pyptyutils.save_updated_arrays(output_folder, epoch,current_probe_step, current_probe_pos_step, current_tilts_step,current_obj_step, obj, probe, tilts_correction, positions_correction, positions, tilts,static_background, current_aberrations_array_step, current_static_background_step, count, current_loss, current_sse, aberrations_array, beam_current, current_beam_current_step, save_flag, save_loss_log, constraint_contributions, actual_step, count_linesearch, d_value, new_d_value,current_update_step_bfgs,t0, xp, warnings) # <-------------- save the results --------------
    pyptyutils.save_current_checkpoint_obj_probe(output_folder, obj, probe, tilts_correction, positions_correction,positions, tilts, static_background, 1,1,1,0,0,0, aberrations_array, beam_current, 0, xp)
    obj, probe, positions, positions_correction, tilts, tilts_correction,static_background, aberrations_array,  beam_current=None, None, None, None, None,None, None, None, None
    try:
        cp.fft.config.clear_plan_cache()
        pool.free_all_blocks()
        pinned_pool.free_all_blocks()
    except:
        pass
    if print_flag!=0:
        sys.stdout.write("\nDone :)")
        sys.stdout.flush()



def bfgs_update(algorithm_type, this_slice_distances, this_step_probe, this_step_obj, this_step_pos_correction, this_step_tilts, measured_array, this_wavelength, masks, pixel_size_x_A, pixel_size_y_A, phase_norm_weight, abs_norm_weight, stepsize_threshold_low, probe_reg_weight, aperture_mask, recon_type, defocus_array, Cs, alpha_near_field, damping_cutoff_multislice, smooth_rolloff, update_extra_cut, keep_probe_states_orthogonal, do_charge_flip, cf_delta_phase, cf_delta_abs, cf_beta_phase, cf_beta_abs, phase_only_obj, wedge_mu, beta_wedge, wolfe_c1_constant, wolfe_c2_constant, atv_weight, atv_q, atv_p, tune_only_probe_phase, propmethod, this_chopped_sequence, load_one_by_one, data_multiplier,data_pad, phase_plate_in_h5, print_flag, this_loss_weight, max_count, reduce_factor, optimism, mixed_variance_weight, mixed_variance_sigma, data_bin, data_shift_vector, smart_memory, default_float, default_complex, default_int, upsample_pattern, this_step_static_background, tilt_mode, fancy_sigma, tune_only_probe_abs, aberration_marker, this_step_aberrations_array, probe_marker, compute_batch, current_window, current_window_weight, dynamically_resize_yx_object, lazy_clean, current_gaussian_filter, current_apply_gaussian_filter_amplitude, this_beam_current_step, xp, remove_fft_cache, is_first_epoch, hist_length, actual_step, fast_axis_reg_weight_positions, fast_axis_reg_weight_tilts, slow_axis_reg_weight_positions, slow_axis_reg_weight_tilts,  scan_size, current_deformation_reg_weight_positions, current_deformation_reg_weight_tilts, warnings, data_simulation_flag,pos_puzzling,allow_subPixel_shift):
    """
    This is one of the core functions of PyPty. It performs updates of all active reconstruction parameters (object, probe, positions, tilts, etc.) via l-BFGS algorithm.

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
    """
    global obj, probe, pool, pinned_pool, positions, positions_correction, tilts, tilts_correction, static_background, aberrations_array, beam_current, history_bfgs
    empty_hist=history_bfgs["empty_hist"]
    update_obj = this_step_obj>0
    update_probe = this_step_probe>0
    update_tilts = this_step_tilts>0
    update_aberrations_array = this_step_aberrations_array>0
    update_pos_correction = this_step_pos_correction>0
    update_static_background = this_step_static_background>0
    update_beam_current = this_beam_current_step>0
    ## some flags
    smooth_rolloff_loss=0
    is_mixed_state=(probe.shape[-1]>1)
    is_mixed_obj=(obj.shape[-1]>1)
    is_single_tilt=(tilts.shape[0]==1)
    is_single_pos=(positions.shape[0]==1)
    multiple_scenarios=len(probe.shape)==4
    clean_grad_ap=False
    if probe_reg_weight==xp.inf:
        clean_grad_ap=True
        probe_reg_weight=0
        if type(aperture_mask)==xp.ndarray:
            probe=pyptyutils.fourier_clean(probe, mask=aperture_mask, default_float=default_float)
        else:
            probe=pyptyutils.fourier_clean(probe, cutoff=aperture_mask, default_float=default_float)
    if update_probe:
        if is_mixed_state and keep_probe_states_orthogonal:
            probe=pyptyloss_and_direction.make_states_orthogonal(probe)
        probe = pyptyutils.fourier_clean(probe, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float)
    
    if update_obj:
        if do_charge_flip:
            obj=pyptyloss_and_direction.charge_flip(obj, cf_delta_phase, cf_delta_abs, cf_beta_phase, cf_beta_abs, fancy_sigma);
            reset_bfgs_history()
            empty_hist=True
        if not lazy_clean:
            if smart_memory:
                for i in range(obj.shape[2]):
                    for j in range(obj.shape[3]):
                        what=obj[:,:,i,j]
                        obj[:,:,i,j]=pyptyutils.fourier_clean(what, cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
            else:
                obj=pyptyutils.fourier_clean(obj,   cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
    if smart_memory:
        try:
            if remove_fft_cache:
                cp.fft.config.clear_plan_cache()
            pool.free_all_blocks()
            pinned_pool.free_all_blocks()
        except:
            pass
            
            
    if empty_hist:
        total_loss, this_sse, this_object_grad, this_probe_grad, this_pos_grad, this_tilts_grad, static_background_grad, this_grad_aberrations_array, this_beam_current_grad, constraint_contributions = pyptyloss_and_direction.loss_and_direction(obj, probe, positions, positions_correction, tilts, tilts_correction, this_slice_distances,  measured_array,  algorithm_type, this_wavelength, update_probe, update_obj, update_pos_correction, update_tilts, masks, pixel_size_x_A, pixel_size_y_A, recon_type, Cs, defocus_array, alpha_near_field, damping_cutoff_multislice, smooth_rolloff_loss, propmethod, this_chopped_sequence, load_one_by_one, data_multiplier, data_pad, phase_plate_in_h5, this_loss_weight, data_bin, data_shift_vector, upsample_pattern, static_background, update_static_background, tilt_mode, aberration_marker, probe_marker, aberrations_array, compute_batch, phase_only_obj, beam_current, update_beam_current, update_aberrations_array, default_float, default_complex, xp, is_first_epoch, scan_size,fast_axis_reg_weight_positions, slow_axis_reg_weight_positions, slow_axis_reg_weight_tilts, current_deformation_reg_weight_positions, current_deformation_reg_weight_tilts, fast_axis_reg_weight_tilts, aperture_mask, probe_reg_weight, current_window_weight, current_window, phase_norm_weight, abs_norm_weight, atv_weight, atv_q, atv_p, mixed_variance_weight, mixed_variance_sigma, wedge_mu, beta_wedge, smart_memory, print_flag, data_simulation_flag) #get the loss and derivatives
        if clean_grad_ap:
            if type(aperture_mask)==xp.ndarray:
                this_probe_grad=pyptyutils.fourier_clean(this_probe_grad, mask=aperture_mask, default_float=default_float)
            else:
                this_probe_grad=pyptyutils.fourier_clean(this_probe_grad, cutoff=aperture_mask, default_float=default_float)
        if smart_memory:
            try:
                if remove_fft_cache:
                    cp.fft.config.clear_plan_cache()
                pool.free_all_blocks()
                pinned_pool.free_all_blocks()
            except:
                pass
        if not lazy_clean and update_obj:
            if smart_memory:
                for i in range(obj.shape[2]):
                    for j in range(obj.shape[3]):
                        what=this_object_grad[:,:,i,j]
                        this_object_grad[:,:,i,j]=pyptyutils.fourier_clean(what, cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
            else:
                this_object_grad=pyptyutils.fourier_clean(this_object_grad,   cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
        if phase_only_obj:
            this_object_grad=pyptyutils.complex_grad_to_phase_grad(this_object_grad, obj)
        if tune_only_probe_abs:
            this_probe_grad_fourier=pyptyfft.ifft2(this_probe_grad)*this_probe_grad.shape[0]*this_probe_grad.shape[1]
            this_probe_fourier=pyptyfft.fft2(probe)
            probe_fourier_mag,probe_fourier_phase=xp.sqrt(xp.abs(this_probe_fourier)), xp.angle(this_probe_fourier)
            this_probe_grad=pyptyutils.complex_grad_to_mag_grad(this_probe_grad_fourier, probe_fourier_mag,probe_fourier_phase)
            tune_only_probe_phase=False
        if tune_only_probe_phase:
            this_probe_grad_fourier=pyptyfft.ifft2_ishift(this_probe_grad) * this_probe_grad.shape[0] * this_probe_grad.shape[1]
            this_probe_fourier=pyptyfft.shift_fft2(probe)
            probe_fourier_abs, probe_fourier_phase = xp.abs(this_probe_fourier), xp.angle(this_probe_fourier)
            this_probe_grad=pyptyutils.complex_grad_to_phase_grad(this_probe_grad_fourier, this_probe_fourier)
        this_obj_update=-1*this_object_grad if update_obj else 0
        d_val_obj=(2-phase_only_obj)*cp.sum(cp.abs(this_obj_update)**2)
        if d_val_obj==0:
            d_val_obj=1
        this_probe_update= -1*this_probe_grad*d_val_obj/((2-tune_only_probe_abs-tune_only_probe_phase)*cp.sum(cp.abs(this_probe_grad)**2)) if update_probe else 0
        this_pos_update= -1*this_pos_grad*d_val_obj/(cp.sum(this_pos_grad**2)) if update_pos_correction else 0
        this_tilts_update= -1*this_tilts_grad*d_val_obj/(cp.sum(this_tilts_grad**2)) if update_tilts else 0
        this_static_background_update= -1*static_background_grad*d_val_obj/(cp.sum(static_background_grad**2)) if update_static_background else 0
        this_aberrations_array_update =-1*this_grad_aberrations_array*d_val_obj/(cp.sum(this_grad_aberrations_array**2)) if update_aberrations_array else 0
        this_beam_current_update=-1*this_beam_current_grad*d_val_obj/(cp.sum(this_beam_current_grad**2)) if update_beam_current else 0
        history_bfgs["empty_hist"]=False
    else: ### HERE we have a start to construct the BFGS update
        total_loss, this_sse, this_object_grad, this_probe_grad, this_pos_grad, this_tilts_grad, static_background_grad, this_grad_aberrations_array, this_beam_current_grad, constraint_contributions=history_bfgs["prev_loss"], history_bfgs["prev_sse"], history_bfgs["obj_grad"], history_bfgs["probe_grad"], history_bfgs["pos_grad"], history_bfgs["tilt_grad"], history_bfgs["static_background_grad"], history_bfgs["aberrations_grad"], history_bfgs["beam_current_grad"], history_bfgs["constraint_contributions"]
        if clean_grad_ap:
            if type(aperture_mask)==xp.ndarray:
                this_probe_grad=pyptyutils.fourier_clean(this_probe_grad, mask=aperture_mask, default_float=default_float)
            else:
                this_probe_grad=pyptyutils.fourier_clean(this_probe_grad, cutoff=aperture_mask, default_float=default_float)
        rhos_hist=history_bfgs["rho_hist"]
        alphas=[]
        this_obj_update=1*this_object_grad if update_obj else 0
        this_probe_update=1*this_probe_grad if update_probe else 0
        this_pos_update= 1*this_pos_grad if update_pos_correction else 0
        this_tilts_update= 1*this_tilts_grad if update_tilts else 0
        this_static_background_update= 1*static_background_grad if update_static_background else 0
        this_aberrations_array_update =1*this_grad_aberrations_array if update_aberrations_array else 0
        this_beam_current_update=1*this_beam_current_grad if update_beam_current else 0
        
        for subdumb_bfgs_i in range(len(rhos_hist)-1,-1,-1):
            rhoi=rhos_hist[subdumb_bfgs_i]
            siTq=0
            if update_obj:
                siTq+=(2-phase_only_obj)*cp.sum(cp.real(history_bfgs["obj_hist_s"][subdumb_bfgs_i]*cp.conjugate(this_obj_update)))
            if update_probe:
                siTq+=(2-tune_only_probe_abs-tune_only_probe_phase)*cp.sum(cp.real(history_bfgs["probe_hist_s"][subdumb_bfgs_i]*cp.conjugate(this_probe_update)))
            if update_pos_correction:
                siTq+=cp.sum(history_bfgs["positions_hist_s"][subdumb_bfgs_i]*this_pos_update)
            if update_tilts:
                siTq+=cp.sum(history_bfgs["tilts_hist_s"][subdumb_bfgs_i]*this_tilts_update)
            if update_static_background:
                siTq+=cp.sum(history_bfgs["static_background_hist_s"][subdumb_bfgs_i]*this_static_background_update)
            if update_aberrations_array:
                siTq+=cp.sum(history_bfgs["aberrations_hist_s"][subdumb_bfgs_i]*this_aberrations_array_update)
            if  update_beam_current:
                siTq+=cp.sum(history_bfgs["beam_current_hist_s"][subdumb_bfgs_i]*this_beam_current_update)
            alphai=rhoi*siTq
            alphas.append(alphai)
            if update_obj:
                this_obj_update-=alphai*history_bfgs["obj_hist_y"][subdumb_bfgs_i]
            if update_probe:
                this_probe_update-=alphai*history_bfgs["probe_hist_y"][subdumb_bfgs_i]
            if update_pos_correction:
                this_pos_update-=alphai*history_bfgs["positions_hist_y"][subdumb_bfgs_i]
            if update_tilts:
                this_tilts_update-=alphai*history_bfgs["tilts_hist_y"][subdumb_bfgs_i]
            if update_static_background:
                this_static_background_update-=alphai*history_bfgs["static_background_hist_y"][subdumb_bfgs_i]
            if update_aberrations_array:
                this_aberrations_array_update-=alphai*history_bfgs["aberrations_hist_y"][subdumb_bfgs_i]
            if update_beam_current:
                this_beam_current_update-=alphai*history_bfgs["beam_current_hist_y"][subdumb_bfgs_i]
        alphas=alphas[::-1]
        if update_obj:
            gamma=history_bfgs["gamma_obj"]
            if gamma!=0 and gamma==gamma: this_obj_update*=gamma;
        if update_probe:
            gamma=history_bfgs["gamma_probe"]
            if gamma!=0 and gamma==gamma: this_probe_update*=gamma;
        if update_pos_correction:
            gamma=history_bfgs["gamma_pos"]
            if gamma!=0 and gamma==gamma: this_pos_update*=gamma;
        if update_tilts:
            gamma=history_bfgs["gamma_tilts"]
            if gamma!=0 and gamma==gamma:  this_tilts_update*=gamma;
        if update_static_background:
            gamma=history_bfgs["gamma_static"]
            if gamma!=0 and gamma==gamma: this_static_background_update*=gamma;
        if update_aberrations_array:
            gamma=history_bfgs["gamma_aberrations"]
            if gamma!=0 and gamma==gamma: this_aberrations_array_update*=gamma;
        if update_beam_current:
            gamma=history_bfgs["gamma_beam_current"]
            if gamma!=0 and gamma==gamma: this_beam_current_update*=gamma;
        for subdumb_bfgs_i in range(len(rhos_hist)):
            rhoi=rhos_hist[subdumb_bfgs_i]
            alphai=alphas[subdumb_bfgs_i]
            yiTr=0
            if update_obj: yiTr+=(2-phase_only_obj)*cp.sum(cp.real(history_bfgs["obj_hist_y"][subdumb_bfgs_i]*cp.conjugate(this_obj_update)));
            if update_probe: yiTr+=(2-tune_only_probe_abs-tune_only_probe_phase)*cp.sum(cp.real(history_bfgs["probe_hist_y"][subdumb_bfgs_i]*cp.conjugate(this_probe_update)));
            if update_pos_correction: yiTr+=cp.sum(history_bfgs["positions_hist_y"][subdumb_bfgs_i]*this_pos_update);
            if update_tilts: yiTr+=cp.sum(history_bfgs["tilts_hist_y"][subdumb_bfgs_i]*this_tilts_update);
            if update_static_background: yiTr+=cp.sum(history_bfgs["static_background_hist_y"][subdumb_bfgs_i]*this_static_background_update);
            if update_aberrations_array: yiTr+=cp.sum(history_bfgs["aberrations_hist_y"][subdumb_bfgs_i]*this_aberrations_array_update);
            if update_beam_current: yiTr+=cp.sum(history_bfgs["beam_current_hist_y"][subdumb_bfgs_i]*this_beam_current_update);
            beta=alphai-rhoi*yiTr
            if update_obj: this_obj_update+=beta*history_bfgs["obj_hist_s"][subdumb_bfgs_i]
            if update_probe: this_probe_update+=beta*history_bfgs["probe_hist_s"][subdumb_bfgs_i]
            if update_pos_correction: this_pos_update+=beta*history_bfgs["positions_hist_s"][subdumb_bfgs_i]
            if update_tilts: this_tilts_update+=beta*history_bfgs["tilts_hist_s"][subdumb_bfgs_i]
            if update_static_background: this_static_background_update+=beta*history_bfgs["static_background_hist_s"][subdumb_bfgs_i]
            if update_aberrations_array: this_aberrations_array_update+=beta*history_bfgs["aberrations_hist_s"][subdumb_bfgs_i]
            if update_beam_current: this_beam_current_update+=beta*history_bfgs["beam_current_hist_s"][subdumb_bfgs_i]
        if update_obj: this_obj_update*=-1
        if update_probe: this_probe_update*=-1
        if update_pos_correction: this_pos_update*=-1
        if update_tilts: this_tilts_update*=-1
        if update_static_background: this_static_background_update*=-1
        if update_aberrations_array: this_aberrations_array_update*=-1
        if update_beam_current: this_beam_current_update*=-1
        if not(phase_only_obj) and not(lazy_clean):
            if smart_memory:
                for i in range(obj.shape[2]):
                    for j in range(obj.shape[3]):
                        what=this_obj_update[:,:,i,j]
                        this_obj_update[:,:,i,j]=pyptyutils.fourier_clean(what, cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
            else:
                this_obj_update=pyptyutils.fourier_clean(this_obj_update,   cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
        #### END of BFGS update construction
    d_value=0
    if update_obj: d_value+=(2-phase_only_obj)*cp.sum(cp.real(cp.conjugate(this_obj_update)*this_object_grad));
    if update_probe: d_value+=(2-tune_only_probe_phase)*cp.sum(cp.real(cp.conjugate(this_probe_update)*this_probe_grad));
    if update_pos_correction: d_value+=cp.sum(this_pos_update*this_pos_grad);
    if update_tilts: d_value+=cp.sum(this_tilts_update*this_tilts_grad);
    if update_static_background: d_value+=cp.sum(this_static_background_update*static_background_grad);
    if update_aberrations_array: d_value+=cp.sum(this_aberrations_array_update*this_grad_aberrations_array);
    if update_beam_current: d_value+=cp.sum(this_beam_current_update*this_beam_current_grad);
        
    if d_value>=0: ### This should not happen if wolfe 2 is satisfied, but just in case!
        sys.stdout.write("\n\n\nWarning! Positive dir-derivative, using normal gradient-descent! Please notify Anton Gladyshev if this message appears!\n\n\n\n\n\n")
        warnings+="\n\\Positive dir-derivative, using normal gradient-descent!"
        sys.stdout.flush()
        this_obj_update=-1*this_object_grad if update_obj else 0
        this_probe_update= -1*this_probe_grad if update_probe else 0
        this_pos_update= -1*this_pos_grad if update_pos_correction else 0
        this_tilts_update= -1*this_tilts_grad if update_tilts else 0
        this_static_background_update= -1*static_background_grad if update_static_background else 0
        this_aberrations_array_update =-1*this_grad_aberrations_array if update_aberrations_array else 0
        this_beam_current_update=-1*this_beam_current_grad if update_beam_current else 0
        if update_obj:
            gamma=history_bfgs["gamma_obj"]
            if gamma>0 and gamma==gamma: this_obj_update*=gamma;
        if update_probe:
            gamma=history_bfgs["gamma_probe"]
            if gamma>0 and gamma==gamma: this_probe_update*=gamma;
        if update_pos_correction:
            gamma=history_bfgs["gamma_pos"]
            if gamma>0 and gamma==gamma: this_pos_update*=gamma;
        if update_tilts:
            gamma=history_bfgs["gamma_tilts"]
            if gamma>0 and gamma==gamma:  this_tilts_update*=gamma;
        if update_static_background:
            gamma=history_bfgs["gamma_static"]
            if gamma>0 and gamma==gamma: this_static_background_update*=gamma;
        if update_aberrations_array:
            gamma=history_bfgs["gamma_aberrations"]
            if gamma>0 and gamma==gamma: this_aberrations_array_update*=gamma;
        if update_beam_current:
            gamma=history_bfgs["gamma_beam_current"]
            if gamma>0 and gamma==gamma: this_beam_current_update*=gamma;
        d_value=0
        if update_obj: d_value+=(2-phase_only_obj)*cp.sum(cp.real(cp.conjugate(this_obj_update)*this_object_grad));
        if update_probe: d_value+=(2-tune_only_probe_phase)*cp.sum(cp.real(cp.conjugate(this_probe_update)*this_probe_grad));
        if update_pos_correction: d_value+=cp.sum(this_pos_update*this_pos_grad);
        if update_tilts: d_value+=cp.sum(this_tilts_update*this_tilts_grad);
        if update_static_background: d_value+=cp.sum(this_static_background_update*static_background_grad);
        if update_aberrations_array: d_value+=cp.sum(this_aberrations_array_update*this_grad_aberrations_array);
        if update_beam_current: d_value+=cp.sum(this_beam_current_update*this_beam_current_grad);
    ##################### Trying out the new points ###############################################################
    count=-1
    while True:
        count+=1
        if update_obj:
            if phase_only_obj:
                new_obj=xp.exp(1j*(xp.angle(obj)+actual_step*this_obj_update))*xp.abs(obj)
                if not lazy_clean:
                    if smart_memory:
                        for i in range(obj.shape[2]):
                            for j in range(obj.shape[3]):
                                what=new_obj[:,:,i,j]
                                new_obj[:,:,i,j]=pyptyutils.fourier_clean(what, cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
                    else:
                        new_obj=pyptyutils.fourier_clean(new_obj,   cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
            else:
                new_obj=obj+actual_step*this_obj_update
        else:
            new_obj=obj
        if update_probe:
            if tune_only_probe_phase:
                probe_fourier=pyptyfft.shift_fft2(probe)
                probe_fourier_abs, probe_fourier_angle=xp.abs(probe_fourier), xp.angle(probe_fourier)
                new_probe=pyptyfft.ifft2_ishift(probe_fourier_abs * xp.exp(1j*(probe_fourier_angle + actual_step*this_probe_update)))
            else:
                if tune_only_probe_abs:
                    probe_fourier=pyptyfft.shift_fft2(probe)
                    probe_fourier_abs, probe_fourier_angle = xp.abs(probe_fourier), xp.angle(probe_fourier)
                    new_probe=pyptyfft.ifft2_ishift((probe_fourier_abs+ actual_step*this_probe_update) * xp.exp(1j*(probe_fourier_angle)))
                else:
                    new_probe=probe + actual_step*this_probe_update
        else:
            new_probe=probe
        new_positions_correction=positions_correction+actual_step*this_pos_update if update_pos_correction else positions_correction
        new_tilts_correction=tilts_correction+actual_step*this_tilts_update if update_tilts else tilts_correction
        new_static_background=static_background+actual_step*this_static_background_update if update_static_background else static_background
        new_aberrations_array=aberrations_array+actual_step*this_aberrations_array_update if update_aberrations_array else aberrations_array
        new_beam_current=beam_current+actual_step*this_beam_current_update if update_beam_current else beam_current
        new_total_loss, new_sse, new_object_grad, new_probe_grad, new_pos_grad, new_tilts_grad, new_static_background_grad, new_grad_aberrations_array, new_beam_current_grad, new_constraint_contributions = pyptyloss_and_direction.loss_and_direction(new_obj, new_probe, positions, new_positions_correction, tilts, new_tilts_correction, this_slice_distances,  measured_array,  algorithm_type, this_wavelength, update_probe, update_obj, update_pos_correction, update_tilts, masks, pixel_size_x_A, pixel_size_y_A, recon_type, Cs, defocus_array, alpha_near_field, damping_cutoff_multislice, smooth_rolloff_loss, propmethod, this_chopped_sequence, load_one_by_one, data_multiplier, data_pad, phase_plate_in_h5, this_loss_weight, data_bin, data_shift_vector, upsample_pattern, new_static_background, update_static_background, tilt_mode, aberration_marker, probe_marker, new_aberrations_array, compute_batch, phase_only_obj, new_beam_current, update_beam_current, update_aberrations_array, default_float, default_complex, xp, is_first_epoch, scan_size,fast_axis_reg_weight_positions,slow_axis_reg_weight_positions, slow_axis_reg_weight_tilts, current_deformation_reg_weight_positions, current_deformation_reg_weight_tilts, fast_axis_reg_weight_tilts, aperture_mask, probe_reg_weight, current_window_weight, current_window, phase_norm_weight, abs_norm_weight, atv_weight, atv_q, atv_p, mixed_variance_weight, mixed_variance_sigma, wedge_mu, beta_wedge, smart_memory, print_flag, data_simulation_flag)
        if clean_grad_ap:
            if type(aperture_mask)==xp.ndarray:
                new_probe_grad=pyptyutils.fourier_clean(new_probe_grad, mask=aperture_mask, default_float=default_float)
            else:
                new_probe_grad=pyptyutils.fourier_clean(new_probe_grad, cutoff=aperture_mask, default_float=default_float)
        
        if smart_memory:
            try:
                if remove_fft_cache:
                    cp.fft.config.clear_plan_cache()
                pool.free_all_blocks()
                pinned_pool.free_all_blocks()
            except:
                pass
        if not(lazy_clean) and update_obj:
            if smart_memory:
                for i in range(new_obj.shape[2]):
                    for j in range(new_obj.shape[3]):
                        what=new_object_grad[:,:,i,j]
                        new_object_grad[:,:,i,j]=pyptyutils.fourier_clean(what, cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
            else:
                new_object_grad=pyptyutils.fourier_clean(new_object_grad,   cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
        if phase_only_obj:
            new_object_grad=pyptyutils.complex_grad_to_phase_grad(new_object_grad, new_obj)
        if tune_only_probe_abs:
            this_probe_grad_fourier=pyptyfft.ifft2(new_probe_grad)*new_probe_grad.shape[0]*new_probe_grad.shape[1]
            new_probe_fourier=pyptyfft.fft2(new_probe)
            probe_fourier_mag,probe_fourier_phase=xp.sqrt(xp.abs(new_probe_fourier)), xp.angle(new_probe_fourier)
            new_probe_grad=pyptyutils.complex_grad_to_mag_grad(new_probe_fourier, probe_fourier_mag,probe_fourier_phase)
        if tune_only_probe_phase:
            new_probe_grad_fourier=pyptyfft.ifft2_ishift(new_probe_grad) * new_probe_grad.shape[0] * new_probe_grad.shape[1]
            new_probe_fourier=pyptyfft.shift_fft2(new_probe)
            probe_fourier_abs, probe_fourier_phase = xp.abs(new_probe_fourier), xp.angle(new_probe_fourier)
            new_probe_grad=pyptyutils.complex_grad_to_phase_grad(new_probe_grad_fourier, new_probe_fourier)
        new_d_value=0
        if update_obj: new_d_value+=(2-phase_only_obj)*cp.sum(cp.real(cp.conjugate(this_obj_update)*new_object_grad));
        if update_probe: new_d_value+=(2-tune_only_probe_phase)*cp.sum(cp.real(cp.conjugate(this_probe_update)*new_probe_grad));
        if update_pos_correction: new_d_value+=cp.sum(this_pos_update*new_pos_grad);
        if update_tilts: new_d_value+=cp.sum(this_tilts_update*new_tilts_grad);
        if update_static_background: new_d_value+=cp.sum(this_static_background_update*new_static_background_grad);
        if update_aberrations_array: new_d_value+=cp.sum(this_aberrations_array_update*new_grad_aberrations_array);
        if update_beam_current: new_d_value+=cp.sum(this_beam_current_update*new_beam_current_grad);
        this_wolfe_1=pyptyutils.wolfe_1(total_loss, new_total_loss, d_value, actual_step, wolfe_c1_constant)
        this_wolfe_2=pyptyutils.wolfe_2(d_value, new_d_value, wolfe_c2_constant)
        if print_flag>=3:
            sys.stdout.write("\nLinesearch iteration %d. This loss is %.3e. Loss change is %.3e. Dir-derivative is %.3e. New dir-derivative is %.3e, This step is %.3e."%(count, total_loss, total_loss-new_total_loss, d_value, new_d_value, actual_step))
            sys.stdout.flush()
        if this_wolfe_1 and this_wolfe_2:
            break
        else:
            if not(this_wolfe_1) and this_wolfe_2:
                actual_step*=reduce_factor # backtracking
            if not(this_wolfe_2) and this_wolfe_1:
                actual_step*=optimism # boosting the step!
            if not(this_wolfe_2 or this_wolfe_1):
                actual_step=type(actual_step)(np.random.uniform(1e-10,1)) ## pick a random step and try again
            if not(max_count is None):
                if count>max_count:
                    break
            if actual_step<stepsize_threshold_low:
                break
        ##############################################################################################################################
    if new_total_loss>=total_loss + actual_step*wolfe_c1_constant*d_value: ## checking Wolfe_1 again
        sys.stdout.write('\nWARNING! The sufficient loss descrese is not achieved, the update is regected, keeping the the same step! Resetting the history! Please terminate if this message appears during the next epoch!\n')
        warnings+="\nSufficient loss descrese is not achieved, the update is regected"
        sys.stdout.flush()
        reset_bfgs_history()
        return total_loss, this_sse, constraint_contributions, actual_step, count, d_value, new_d_value, warnings
    if print_flag>=2:
        sys.stdout.write("\n-->Update done with %d linesearch steps! This loss is %.3e. Loss change is %.3e. Dir-derivative is %.3e. New dir-derivative is %.3e."%(count,total_loss, total_loss-new_total_loss, d_value, new_d_value))
    history_bfgs["obj_hist_s"].append(actual_step*this_obj_update if update_obj else 0)
    history_bfgs["probe_hist_s"].append(actual_step*this_probe_update if update_probe else 0)
    history_bfgs["positions_hist_s"].append(actual_step*this_pos_update if update_pos_correction else 0)
    history_bfgs["tilts_hist_s"].append(actual_step*this_tilts_update if update_tilts else 0)
    history_bfgs["beam_current_hist_s"].append(actual_step*this_beam_current_update if update_beam_current else 0)
    history_bfgs["aberrations_hist_s"].append(actual_step*this_aberrations_array_update if update_aberrations_array else 0)
    history_bfgs["static_background_hist_s"].append(actual_step*this_static_background_update if update_static_background else 0)
    history_bfgs["obj_hist_y"].append(new_object_grad-this_object_grad if update_obj else 0)
    history_bfgs["probe_hist_y"].append(new_probe_grad-this_probe_grad if update_probe else 0)
    history_bfgs["positions_hist_y"].append(new_pos_grad-this_pos_grad if update_pos_correction else 0)
    history_bfgs["tilts_hist_y"].append(new_tilts_grad-this_tilts_grad if update_tilts else 0)
    history_bfgs["beam_current_hist_y"].append(new_beam_current_grad-this_beam_current_grad if update_beam_current else 0)
    history_bfgs["aberrations_hist_y"].append(new_grad_aberrations_array-this_grad_aberrations_array if update_aberrations_array else 0)
    history_bfgs["static_background_hist_y"].append(new_static_background_grad-static_background_grad if update_static_background else 0)
    history_bfgs["prev_loss"]=new_total_loss
    history_bfgs["prev_sse"]=new_sse
    history_bfgs["obj_grad"]=new_object_grad
    history_bfgs["probe_grad"]=new_probe_grad
    history_bfgs["pos_grad"]=new_pos_grad
    history_bfgs["tilt_grad"]=new_tilts_grad
    history_bfgs["static_background_grad"]=new_static_background_grad
    history_bfgs["beam_current_grad"]=new_beam_current_grad
    history_bfgs["aberrations_grad"]=new_grad_aberrations_array
    history_bfgs["constraint_contributions"]=new_constraint_contributions
    yiTsi_tot=0
    if update_obj:
        yiTyi=cp.sum(cp.abs(history_bfgs["obj_hist_y"][-1])**2);
        yiTsi=(2-phase_only_obj)*cp.sum(cp.real(history_bfgs["obj_hist_y"][-1]*cp.conjugate(history_bfgs["obj_hist_s"][-1])));
        gamma=yiTsi/yiTyi
        yiTsi_tot+=yiTsi
        history_bfgs["gamma_obj"]=gamma
    if update_probe:
        yiTyi=cp.sum(cp.abs(history_bfgs["probe_hist_y"][-1])**2);
        yiTsi=(2-tune_only_probe_abs-tune_only_probe_phase)*cp.sum(cp.real(history_bfgs["probe_hist_y"][-1]*cp.conjugate(history_bfgs["probe_hist_s"][-1])));
        yiTsi_tot+=yiTsi
        gamma=yiTsi/yiTyi
        history_bfgs["gamma_probe"]=gamma
    if update_pos_correction:
        yiTyi=cp.sum((history_bfgs["positions_hist_y"][-1])**2);
        yiTsi=cp.sum(history_bfgs["positions_hist_y"][-1]*history_bfgs["positions_hist_s"][-1]);
        yiTsi_tot+=yiTsi
        gamma=yiTsi/yiTyi
        history_bfgs["gamma_pos"]=gamma
    if update_tilts:
        yiTyi=cp.sum((history_bfgs["tilts_hist_y"][-1])**2);
        yiTsi=cp.sum(history_bfgs["tilts_hist_y"][-1]*history_bfgs["tilts_hist_s"][-1]);
        yiTsi_tot+=yiTsi
        gamma=yiTsi/yiTyi
        history_bfgs["gamma_tilts"]=gamma
    if update_static_background:
        yiTyi=cp.sum((history_bfgs["static_background_hist_y"][-1])**2);
        yiTsi=cp.sum(history_bfgs["static_background_hist_y"][-1]*history_bfgs["static_background_hist_s"][-1]);
        yiTsi_tot+=yiTsi
        gamma=yiTsi/yiTyi
        history_bfgs["gamma_static"]=gamma
    if update_aberrations_array:
        yiTyi=cp.sum((history_bfgs["aberrations_hist_y"][-1])**2);
        yiTsi=cp.sum(history_bfgs["aberrations_hist_y"][-1]*history_bfgs["aberrations_hist_s"][-1]);
        yiTsi_tot+=yiTsi
        gamma=yiTsi/yiTyi
        history_bfgs["gamma_aberrations"]=gamma
    if update_beam_current:
        yiTyi=cp.sum((history_bfgs["beam_current_hist_y"][-1])**2);
        yiTsi=cp.sum(history_bfgs["beam_current_hist_y"][-1]*history_bfgs["beam_current_hist_s"][-1]);
        yiTsi_tot+=yiTsi
        gamma=yiTsi/yiTyi
        history_bfgs["gamma_beam_current"]=gamma
    history_bfgs["rho_hist"].append(1/yiTsi_tot)
    while len(history_bfgs["obj_hist_s"])>hist_length:
        history_bfgs["obj_hist_s"].pop(0)
        history_bfgs["probe_hist_s"].pop(0)
        history_bfgs["positions_hist_s"].pop(0)
        history_bfgs["tilts_hist_s"].pop(0)
        history_bfgs["beam_current_hist_s"].pop(0)
        history_bfgs["aberrations_hist_s"].pop(0)
        history_bfgs["static_background_hist_s"].pop(0)
        history_bfgs["obj_hist_y"].pop(0)
        history_bfgs["probe_hist_y"].pop(0)
        history_bfgs["positions_hist_y"].pop(0)
        history_bfgs["tilts_hist_y"].pop(0)
        history_bfgs["beam_current_hist_y"].pop(0)
        history_bfgs["aberrations_hist_y"].pop(0)
        history_bfgs["static_background_hist_y"].pop(0)
        history_bfgs["rho_hist"].pop(0)
        
    probe=1*new_probe
    obj=1*new_obj
   
    
    if update_obj and (current_gaussian_filter or current_apply_gaussian_filter_amplitude):
        reset_bfgs_history()
        if smart_memory:
            for i in range(obj.shape[2]):
                for j in range(obj.shape[3]):
                    what=obj[:,:,i,j]
                    if current_gaussian_filter:
                        what_angle=xp.angle(what)
                        if current_gaussian_filter:
                            what_angle=ndi.gaussian_filter(what_angle, current_gaussian_filter, order=0, mode='nearest')
                    elif current_apply_gaussian_filter_amplitude:
                        what_angle=xp.angle(what)
                    if current_apply_gaussian_filter_amplitude:
                        what_abs=xp.abs(what)
                        what_abs=-xp.log(what_abs)
                        what_abs=ndi.gaussian_filter(what_abs, current_apply_gaussian_filter_amplitude, order=0, mode='nearest')
                        what_abs=xp.exp(-what_abs)
                    elif (current_gaussian_filter):
                        what_abs=xp.abs(what)
                    if current_apply_gaussian_filter_amplitude or current_gaussian_filter:
                        what=what_abs*xp.exp(1j*what_angle)
                    obj[:,:,i,j]=what
            try:
                if remove_fft_cache:
                    cp.fft.config.clear_plan_cache()
                pool.free_all_blocks()
                pinned_pool.free_all_blocks()
            except:
                pass
        else:
            if current_gaussian_filter:
                what_angle=xp.angle(obj)
                if current_gaussian_filter:
                    what_angle=ndi.gaussian_filter(what_angle, current_gaussian_filter, order=0, mode='nearest')
            elif current_apply_gaussian_filter_amplitude:
                what_angle=xp.angle(obj)
            if current_apply_gaussian_filter_amplitude:
                what_abs=xp.abs(obj)
                what_abs=-xp.log(what_abs)
                what_abs=ndi.gaussian_filter(what_abs, current_apply_gaussian_filter_amplitude, order=0, mode='nearest')
                what_abs=xp.exp(-what_abs)
            elif current_gaussian_filter:
                what_abs=xp.abs(obj)
            if current_gaussian_filter or current_apply_gaussian_filter_amplitude:
                obj=what_abs*xp.exp(1j*what_angle)
    
    if update_pos_correction and dynamically_resize_yx_object>0:
        if xp.max(xp.abs(new_positions_correction))>=dynamically_resize_yx_object:
            probe_shape_y, probe_shape_x=probe.shape[0], probe.shape[1]
            obj_shape_y,obj_shape_x = obj.shape[0], obj.shape[1]
            new_positions_correction[:,1]=(new_positions_correction[:,1]-probe_shape_x//2)%probe_shape_x-probe_shape_x+probe_shape_x//2 ## remove cyclic effect
            new_positions_correction[:,0]=(new_positions_correction[:,0]-probe_shape_y//2)%probe_shape_y-probe_shape_y+probe_shape_y//2 ## remove cyclic effect
            
            actuall_full_grid = positions + new_positions_correction
            positions[this_chopped_sequence,:]=xp.round(actuall_full_grid[this_chopped_sequence,:]).astype(default_int)
            new_positions_correction[this_chopped_sequence,:]=((actuall_full_grid-positions)[this_chopped_sequence,:]).astype(default_float)
            pad_top, pad_left=-1*xp.min(positions[this_chopped_sequence,0]), -1*xp.min(positions[this_chopped_sequence,1])
            pad_bottom, pad_right=xp.max(positions[this_chopped_sequence,0])+probe_shape_y-obj_shape_y, xp.max(positions[this_chopped_sequence,1])+probe_shape_x-obj_shape_x
            if pad_top<0: pad_top=0;
            if pad_left<0:  pad_left=0;
            if pad_bottom<0: pad_bottom=0;
            if pad_right<0: pad_right=0;
            if xp!=np:
                try: pad_top=int(pad_top.get());
                except: pass;
                try: pad_bottom= int(pad_bottom.get());
                except: pass;
                try: pad_left=int(pad_left.get());
                except: pass;
                try: pad_right=int(pad_right.get());
                except: pass;
            pad_width=np.array([[pad_top, pad_bottom],[pad_left, pad_right],[0,0],[0,0]])
            if pad_left+pad_right+pad_top+pad_bottom: ## if we actually have to add pixels, then do following:
                positions[:,0]+=pad_top
                positions[:,1]+=pad_left
                if print_flag:
                    sys.stdout.write("\nWARNING: Adding extra pixels to the object in order to correct the positons! Padding: left %d, right, %d, top %d, bottom %d pixels\n"%(pad_left, pad_right, pad_top, pad_bottom))
                obj=xp.pad(obj, pad_width, mode="edge")
                warnings+="\nAdding extra pixels to the object! left %d, right, %d, top %d, bottom %d pixels"%(pad_left, pad_right, pad_top, pad_bottom)
                prev_grad=xp.pad(history_bfgs["obj_grad"], pad_width, mode="constant", constant_values=0)
                history_bfgs["obj_grad"]=prev_grad
                for itemind, item in enumerate(history_bfgs["obj_hist_s"]):
                    history_bfgs["obj_hist_s"][itemind]=xp.pad(item, pad_width, mode="constant", constant_values=0)
                for itemind, item in enumerate(history_bfgs["obj_hist_y"]):
                    history_bfgs["obj_hist_y"][itemind]=xp.pad(item, pad_width, mode="constant", constant_values=0)
    positions_correction=1*new_positions_correction if update_pos_correction else positions_correction
    tilts_correction=1*new_tilts_correction  if update_tilts else tilts_correction
    aberrations_array=1*new_aberrations_array if update_aberrations_array else aberrations_array
    beam_current=1*new_beam_current if update_beam_current else beam_current
    static_background= new_static_background if update_static_background else static_background
    if pos_puzzling and update_pos_correction:
        full_positions = positions+ positions_correction
        puzzling_worked,positions_puzzled =pyptyutils.position_puzzling(full_positions,scan_size)
        if puzzling_worked:
            obj = pyptyutils.intorplate_puzzled_obj(obj,full_positions,positions_puzzled,order=1,method = 'cubic')
            obj = cp.asarray(obj).astype(default_complex)
            if allow_subPixel_shift:
                positions_correction=(positions_puzzled-np.round(positions_puzzled))
            else:
                positions_correction=np.zeros_like(positions_puzzled)
            
            positions=np.round(positions_puzzled).astype(default_int)
            reset_bfgs_history()

            positions=cp.asarray(positions).astype(default_int)
            positions_correction=cp.asarray(positions_correction).astype(default_float)
            del(positions_puzzled)
            print("positions_puzzeled")

    

    return total_loss, this_sse, constraint_contributions, actual_step, count, d_value, new_d_value, warnings





def reset_bfgs_history():
    """
    Reset a global variable history_bfgs that contains information about previous steps.
    """
    global history_bfgs, pool, pinned_pool
    history_bfgs={
                "obj_hist_s": [],
                "probe_hist_s": [],
                "positions_hist_s": [],
                "tilts_hist_s": [],
                "beam_current_hist_s": [],
                "aberrations_hist_s": [],
                "static_background_hist_s": [],
                "obj_hist_y": [],
                "probe_hist_y": [],
                "positions_hist_y": [],
                "tilts_hist_y": [],
                "beam_current_hist_y": [],
                "aberrations_hist_y": [],
                "static_background_hist_y": [],
                "rho_hist": [],
                "prev_loss": None,
                "prev_sse": None,
                "obj_grad": None,
                "probe_grad": None,
                "pos_grad": None,
                "tilt_grad": None,
                "beam_current_grad": None,
                "aberrations_grad": None,
                "static_background_grad": None,
                "empty_hist": True,
                "gamma_obj": 1,
                "gamma_probe": 1,
                "gamma_pos": 1,
                "gamma_tilts": 1,
                "gamma_static": 1,
                "gamma_aberrations": 1,
                "gamma_beam_current": 1,
                            }
    try:
        pool.free_all_blocks()
        pinned_pool.free_all_blocks()
    except:
        pass

