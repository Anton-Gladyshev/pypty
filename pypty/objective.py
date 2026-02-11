import numpy as np
import sys
import h5py
try:
    import cupyx
    import cupy as cp
except:
    ### if no CuPy is found I define dummy clases for some places in the code
    import numpy as cp
    class DummyDevice:
        def __init__(self, device_id): pass
        def __enter__(self): return self  # <- return self here
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    class DummyRuntime:
        @staticmethod
        def getDeviceCount(): return 1
        @staticmethod
        def deviceCanAccessPeer(*args, **kwargs): return True
    class DummyMemoryPool:
        def free_all_blocks(self): pass
    
    class DummyStream:
        def __init__(self, non_blocking=False):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def synchronize(self):
            pass
    class DummyCupy:
        cuda = type("cuda", (), {"Device": DummyDevice, "Stream": DummyStream, "runtime": DummyRuntime()})
        get_default_memory_pool = lambda: DummyMemoryPool()
        get_default_pinned_memory_pool = lambda: DummyMemoryPool()
    cp.cuda = DummyCupy.cuda
    cp.get_default_memory_pool = DummyCupy.get_default_memory_pool
    cp.get_default_pinned_memory_pool = DummyCupy.get_default_pinned_memory_pool

from pypty import fft as pyptyfft
from pypty import utils as pyptyutils
from pypty import multislice_core as pyptymultislice
import os
import threading
import matplotlib.pyplot as plt

half_master_propagator_phase_space, master_propagator_phase_space, q2, qx, qy, exclude_mask, x_real_grid_tilt, y_real_grid_tilt, shift_probe_mask_x, shift_probe_mask_y, exclude_mask_ishift, probe_runx, probe_runy, yx_real_grid_tilt, shift_probe_mask_yx, aberrations_polynomials=None, None,None, None,None, None,None, None, None, None, None, None,None,None,None,None
def loss_and_direction(this_obj, full_probe, this_pos_array, this_pos_correction, this_tilt_array, this_tilts_correction, this_distances, edge_slice_thickness_multiplier,  measured_array,  algorithm_type, this_wavelength, this_step_probe, this_step_obj, this_step_pos_correction, this_step_tilts, masks, pixel_size_x_A, pixel_size_y_A, recon_type, Cs, defocus_array, alpha_near_field, damping_cutoff_multislice, smooth_rolloff, propmethod, this_chopped_sequence, load_one_by_one, data_multiplier, data_pad, phase_plate_in_h5, this_loss_weight, data_bin, data_shift_vector, upsample_pattern, static_background, this_step_static_background, tilt_mode, aberration_marker, probe_marker, aberrations_array, compute_batch, phase_only_obj, beam_current, this_beam_current_step, this_step_aberrations_array, default_float, default_complex, xp, is_first_epoch, scan_size, scan_step_A, fast_axis_reg_weight_positions,slow_axis_reg_weight_positions, slow_axis_reg_weight_tilts, current_deformation_reg_weight_positions, current_deformation_reg_weight_tilts, fast_axis_reg_weight_tilts, aperture_mask, probe_reg_weight, current_window_weight, current_window, phase_norm_weight, abs_norm_weight, atv_weight, atv_q, atv_p, mixed_variance_weight, mixed_variance_sigma, wedge_mu, beta_wedge, probe_mod_reg_weight, probe_mod_tol, smart_memory, print_flag, data_simulation_flag, data_simulation_exitwave, use_padded_intensity):
    """
    Compute the total loss and gradients for ptychographic reconstruction.

    This is the core function of PyPty that performs forward and backward propagation,
    calculates the loss between measured and simulated patterns, and computes the gradients
    of all active reconstruction parameters (object, probe, positions, tilts, etc.).

    Parameters
    ----------
    this_obj : ndarray
        Complex 4D object (current estimate)
    full_probe : ndarray
        Complex probe (y,x,modes) optionally 4D (y,x,modes, scenatios)
    this_pos_array : ndarray
        Integer beam postions in pixels [[y0,x0],.. [yn, xn]]. Note: units are pixels, not angstrom!
    this_pos_correction : ndarray
        Float sub-pixel postions for more precise beam shift. Note: units are pixels, not angstrom!
    this_tilt_array : ndarray
        Beam tilts in radians, shape should be (N_measurements, 6), where first two tilts are applied before the sample, second and third are applied inside (tilted propagator) and two last are applied after the sample
    this_tilts_correction : ndarray
        legacy paramter, actually is not really required. It is a correction that is added to the tilts array.
    this_distances : ndarray
        either just one value for a common slice spacing or list of values for each slice. If object has N slices, it should have N-1 entries.
    measured_array : ndarray  
        array or h5-dataset with diffraction patterns. Should be 3D, [N_measurements, y,x]
    algorithm_type: string 
        string indicating the loss function (error metric)
    this_wavelength : float 
        Electron wavelength in Angstrom
    this_step_probe : float 
        do you refine the beam?
    this_step_obj : float 
        do you refine the object?
    this_step_pos_correction: float 
        do you refine the positions?
    this_step_tilts : float 
        do you refine the tilts?
    masks : ndarray 
        optional, if the data is compressed, you should provide the 3D array with virtual detectors [N_detectors, y,x].
    pixel_size_x_A : float 
        real-space pixel size in x-direction (Angstrom).
    pixel_size_y_A : float 
        real-space pixel size in y-direction (Angstrom).
    recon_type : string 
        "far_field" or "near_field". Changes the exit-wave propagation regime.
    Cs : float 
        Spherical aberration (Angstrom). Only needed for near-field propagation.
    defocus_array : ndarray 
        Array of exit-wave defocus values (Angstrom). Only needed for near-field propagation.
    alpha_near_field : float 
        Flux-preserving correction for near-field propagation.
    damping_cutoff_multislice : float 
        Cutoff (fraction smaller than 1) beyond which the Fouirer-space is cleaned.
    smooth_rolloff : float 
        Smooth rolloff for Fourier masking
    propmethod : string 
        string indicating the method for split-step integration
    this_chopped_sequence : ndarray 
        sequence of measruement indices used for loss and grad calculation (should be sorted)
    load_one_by_one : bool 
        boolean flag. should be True for lazy loading.
    data_multiplier : int 
        multiplicative factor applied to data on the fly.
    data_pad : int 
        padding factor applied to data on the fly.
    phase_plate_in_h5 : string
        path to h5 dataset containing phase plates for each measurement.
    this_loss_weight : float 
        weight applied to the main part of the loss
    data_bin : int 
        binning factor applied to data on the fly
    data_shift_vector : tuple 
        shift vector in pixels (y,x) applied to data on the fly
    upsample_pattern : int 
        virtual "decompression" of the data used to enlarge the probe window
    static_background : ndarray or float
        real-valued array descripting the square root of the static offset on the diffraction patterns.
    this_step_static_background : float 
        do you refine the static background?
    tilt_mode : int 
        flag for tilting
    aberration_marker : ndarray
        
    probe_marker : ndarray
    
    aberrations_array : ndarray
    
    compute_batch : int 
    
    phase_only_obj : bool 
    
    beam_current : ndarray 
    
    this_beam_current_step : float 
    
    this_step_aberrations_array : float 
    
    default_float : dtype 
    
    default_complex : dtype 
    
    xp : module 
    
    is_first_epoch : bool
    
    scan_size : tuple
    
    fast_axis_reg_weight_positions : float
    
    slow_axis_reg_weight_positions : float 
    
    slow_axis_reg_weight_tilts : float 
    
    current_deformation_reg_weight_positions : float 
    
    current_deformation_reg_weight_tilts : float 
    
    fast_axis_reg_weight_tilts : float 
    
    aperture_mask : ndarray 
    
    probe_reg_weight : float 
    
    current_window_weight : float 
    
    current_window : ndarray 
    
    phase_norm_weight : float 
    
    abs_norm_weight : float 
    
    atv_weight : float 
    
    atv_q : float 
    
    atv_p : float 
    
    mixed_variance_weight : float 
    
    mixed_variance_sigma : float 
        
    smart_memory : bool
        do you want to prevent memory fragmentation? Makes the reconstrcution slightly slower
    print_flag : int
        verbodity level
        
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
    """
    
    global pool, pinned_pool,master_propagator_phase_space, half_master_propagator_phase_space,  q2, qx, qy, exclude_mask,exclude_mask_ishift, x_real_grid_tilt, y_real_grid_tilt,yx_real_grid_tilt, shift_probe_mask_x, shift_probe_mask_y,shift_probe_mask_yx, probe_runx,probe_runy, aberrations_polynomials
    if is_first_epoch:
        half_master_propagator_phase_space, master_propagator_phase_space, q2, qx, qy, exclude_mask,exclude_mask_ishift, x_real_grid_tilt, y_real_grid_tilt,yx_real_grid_tilt, shift_probe_mask_x, shift_probe_mask_y,shift_probe_mask_yx, probe_runx,probe_runy, aberrations_polynomials= None, None,None,None,None,None,None,None,None,None,None,None,None,None,None,None
        
   ################################################################################################################
   ################ checking the situation and prepare a bunch of flags ###########################################
   ################################################################################################################
    devcount, use_device= (cp.cuda.runtime.getDeviceCount(),  True) if cp!=np else (1, False) ## do you have >1 GPU? Or any GPU?
    pattern_number = len(this_chopped_sequence) ## how much data is there?
    n_remain=pattern_number%(compute_batch)     ## computing batches
    step_batch=(pattern_number//devcount)//compute_batch
    ranges_patterns=[0]
    for devind in range(1,devcount+1): # how to split the data across availble GPUs
        ranges_patterns.append(devind*step_batch*compute_batch)
    ranges_patterns[-1]=pattern_number ## if there is a remainder (can't devide patterns evenly across GPUs), pass it to the last gpu
    if cp != np and devcount>1:
        for tt in range(1,devcount):
            try:
                cp.cuda.runtime.deviceCanAccessPeer(tt, 0)
                cp.cuda.runtime.deviceCanAccessPeer(0, tt)
            except cp.cuda.runtime.CUDARuntimeError as e:
                sys.stdout.write("\nCould not enable peer access!")
                sys.stdout.flush()
    gpu_loss_flag = (xp==np and cp!=np) ## if you store your stuff on cpu, send it to GPU #0
    if gpu_loss_flag:
        this_obj=cp.asarray(this_obj)
        full_probe=cp.asarray(full_probe)
        this_pos_array=cp.asarray(this_pos_array)
        this_pos_correction=cp.asarray(this_pos_correction)
        this_tilt_array=cp.asarray(this_tilt_array)
        this_tilts_correction=cp.asarray(this_tilts_correction)
        this_distances=cp.asarray(this_distances)
        defocus_array=cp.asarray(defocus_array)
        try:
            masks=cp.asarray(masks)
        except:
            pass
        try:
            beam_current=cp.asarray(beam_current)
        except:
            pass
        try:
            aperture_mask=cp.asarray(aperture_mask)
        except:
            pass
        try:
            current_window=cp.asarray(current_window)
        except:
            pass
        try:
            if not(static_background is None):
                static_background=cp.asarray(static_background)
        except:
            pass
        try:
            aberrations_array=cp.asarray(aberrations_array)
        except:
            pass
        if default_complex==np.complex128:
            default_float, default_complex, default_int, default_int_cpu, default_float_cpu, default_complex_cpu=cp.float64, cp.complex128, cp.int32, np.int32, np.float64, np.complex128
        else:
            default_float, default_complex, default_int, default_int_cpu, default_float_cpu, default_complex_cpu=cp.float32, cp.complex64, cp.int16, np.int16, np.float32, np.complex64
    ########################################################################################
    ############################ Preparation of Arrays #####################################
    ########################################################################################
    if 'compressed' in algorithm_type:
        masks_len=masks.shape[0] #int
    this_ps, is_single_tilt, is_single_pos, is_single_dist, num_slices, n_obj_modes, n_probe_modes, is_single_defocus, multiple_scenarios, fluctuating_current_flag = full_probe.shape[0], (this_tilt_array.shape[0]==1), (this_pos_array.shape[0]==1), (this_distances.shape[0]==1), this_obj.shape[2], this_obj.shape[3], full_probe.shape[2], False, not(probe_marker is None) and len(full_probe.shape)==4, not(beam_current is None)
    
    if data_simulation_flag:
        if print_flag>1:
            print("\nSimiulating diffraction patterns, the dataset will be saved in %s."%data_simulation_flag)
        if data_simulation_exitwave:
            sim_patterns=cp.zeros((pattern_number, this_ps, this_ps,n_probe_modes, n_obj_modes), dtype=cp.complex128)
        else:
            sim_patterns=cp.zeros((pattern_number, this_ps-2*data_pad*(1-use_padded_intensity), this_ps-2*data_pad*(1-use_padded_intensity)), dtype=cp.float64)
   
    is_fully_coherent=n_probe_modes==1 and n_obj_modes==1
    if probe_runx is None:
        probe_runx,probe_runy=cp.meshgrid(cp.arange(this_ps, dtype=int),cp.arange(this_ps, dtype=int), indexing="xy")
        probe_runx,probe_runy=probe_runx[None,:,:],probe_runy[None,:,:]
  
    if exclude_mask is None:
        q2, qx, qy, exclude_mask, exclude_mask_ishift = pyptyutils.create_spatial_frequencies(pixel_size_x_A, pixel_size_y_A, this_ps, damping_cutoff_multislice, smooth_rolloff, default_float)     # create some arrays with spatial frequencies. Many of them are actually idential (or almost idential, so I will later clean this mess). In any case, the code is currently optimized to create them only once (when configured properly)
        qx,qy,q2, exclude_mask, exclude_mask_ishift=qx[None,:,:],qy[None,:,:],q2[None,:,:], exclude_mask[None,:,:], exclude_mask_ishift[None,:,:]
    if tilt_mode and (x_real_grid_tilt is None): ## if anything but zero
        x_real_grid_tilt, y_real_grid_tilt=cp.meshgrid(pyptyfft.fftshift(pyptyfft.fftfreq(full_probe.shape[1])), pyptyfft.fftshift(pyptyfft.fftfreq(full_probe.shape[0])), indexing="xy")
        x_real_grid_tilt=((x_real_grid_tilt*6.2831855j*full_probe.shape[1]*pixel_size_x_A/this_wavelength)[None,:,:,None]).astype(default_complex)
        y_real_grid_tilt=((y_real_grid_tilt*6.2831855j*full_probe.shape[0]*pixel_size_y_A/this_wavelength)[None,:,:,None]).astype(default_complex)
        yx_real_grid_tilt=cp.stack((y_real_grid_tilt, x_real_grid_tilt), axis=1) # (1, 2, y, x, 1)
    if shift_probe_mask_yx is None:
        shift_probe_mask_x, shift_probe_mask_y=cp.meshgrid(pyptyfft.fftfreq(full_probe.shape[1]), pyptyfft.fftfreq(full_probe.shape[0]), indexing="xy")
        shift_probe_mask_x, shift_probe_mask_y=(-6.2831855j*shift_probe_mask_x[None,:,:]).astype(default_complex), (-6.2831855j*shift_probe_mask_y[None,:,:]).astype(default_complex)
        shift_probe_mask_x=shift_probe_mask_x*exclude_mask_ishift
        shift_probe_mask_y=shift_probe_mask_y*exclude_mask_ishift
        shift_probe_mask_yx=cp.stack((shift_probe_mask_y, shift_probe_mask_x), axis=1) #(1, 2, y, x)
    #is_multislice = num_slices>1 or propmethod!="multislice"
    is_single_defocus = recon_type=="far_field" or defocus_array.shape[0]==1
    helper_flag_1= this_step_probe or this_beam_current_step
    helper_flag_2= helper_flag_1 or this_step_aberrations_array
    helper_flag_3= helper_flag_2 or (this_step_tilts and (tilt_mode==2 or tilt_mode==4))
    helper_flag_4= helper_flag_3 or this_step_pos_correction
    if not(aberration_marker is None):
        num_abs=aberrations_array.shape[1]
        if aberrations_polynomials is None:
            aberrations_polynomials=-1j*pyptyutils.get_ctf_matrix(this_wavelength*qx[0], this_wavelength*qy[0], num_abs, this_wavelength).astype(default_complex)
        local_aberrations_phase_plates=cp.exp(cp.sum(aberrations_polynomials[None,:,:,:]*aberrations_array[:,:,None,None], 1))*exclude_mask_ishift
        morph_incomming_beam=True
    else:
        morph_incomming_beam=False
    same_tilt_propagator_flag = is_single_tilt or (tilt_mode==1 or tilt_mode==2)
    if not(phase_plate_in_h5 is None):
        phase_plate_active = True
        phase_plate_data = h5py.File(phase_plate_in_h5, "r")["configs"]
        N_phase_plates = phase_plate_data.shape[0]
        is_single_phase_plate= (N_phase_plates==1)
        matrix_phase_plates= (len(phase_plate_data.shape)==3)
    else:
        phase_plate_active=False
    if same_tilt_propagator_flag and is_single_dist and (master_propagator_phase_space is None):
        this_tilt=(this_tilt_array+this_tilts_correction)[0,:]
        this_tan_y_inside=this_tilt[2]
        this_tan_x_inside=this_tilt[3]
       
        if propmethod=="multislice":
            master_propagator_phase_space=cp.exp(-3.141592654j*this_distances[0]*(this_wavelength*q2+2*(qx*this_tan_x_inside+qy*this_tan_y_inside)))*exclude_mask_ishift
            master_propagator_phase_space=cp.expand_dims(master_propagator_phase_space,(-1,-2)) ### probe modes
            half_master_propagator_phase_space=None
    
    if not(same_tilt_propagator_flag):
        master_propagator_phase_space, half_master_propagator_phase_space=None,None
    
    this_tilt_array=(this_tilt_array+this_tilts_correction)[:,:,None, None]
    this_pos_array=this_pos_array[:,:,None]
    probe_shift_flag=cp.sum(this_pos_correction**2)!=0.0 or this_step_pos_correction
    tcsl=len(this_chopped_sequence)
    sh0=compute_batch if compute_batch<=tcsl else tcsl
    if multiple_scenarios:
        full_probe=cp.moveaxis(full_probe,3,0)
    else:
        full_probe=full_probe[None, :,:,:]
    static_background_is_there=not(static_background is None)
    if static_background_is_there:
       static_square=(static_background[None,:,:]**2)*exclude_mask
    ################################################################################################################
    ############################# Start of the Pattern Loop ########################################################
    ################################################################################################################
    if cp!=np:
        for i in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(i):
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                
    def worker(device_ind): ### this function is a wrapper for multiple GPUs
        device=cp.cuda.Device(device_ind)
        with device:
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                object_grad, probe_grad, pos_grad, tilts_grad = cp.zeros_like(this_obj), cp.zeros_like(full_probe), cp.zeros_like(this_pos_correction), cp.zeros_like(this_tilts_correction) ### this runs very fast
                static_background_grad = cp.zeros_like(static_background) if type(static_background)==cp.ndarray else 0
                beam_current_grad = cp.zeros_like(beam_current) if not(beam_current is None) else 0
                aberrations_array_grad=cp.zeros_like(aberrations_array) if not(aberrations_array is None) else 0
                loss, sse = 0, 0
                
                fourier_probe_grad=None
                if cp!=np and load_one_by_one:
                    ms0, ms1, ms2=measured_array.shape
                    pinned_measured=cupyx.zeros_pinned((compute_batch, ms1,ms2), dtype=measured_array.dtype)
                    if n_remain!=0 and (devcount==1 or (device_ind==(devcount-1))):
                        pinned_measured_remain=cupyx.zeros_pinned((n_remain, ms1,ms2), dtype=measured_array.dtype);
                if propmethod=="multislice":
                    waves_multislice = cp.zeros((sh0, this_ps,this_ps,num_slices,n_probe_modes,n_obj_modes, 1 + int(this_step_tilts>0 and (tilt_mode==0 or tilt_mode==3 or tilt_mode==4))), dtype=default_complex)
                
                this_exit_wave           = cp.zeros((sh0, this_ps,this_ps,         n_probe_modes,n_obj_modes   ), dtype=default_complex)
                ##### Acutal loop starts here ##############
                for i in range(ranges_patterns[device_ind], ranges_patterns[device_ind+1], compute_batch):
                    next_i=i+compute_batch
                    tcs=this_chopped_sequence[i:next_i]
                    lltcs=len(tcs)
                    if compute_batch>1 and i>0 and next_i>pattern_number:
                        waves_multislice=waves_multislice[:lltcs]
                        waves_multislice=waves_multislice[:lltcs]
                        this_exit_wave=this_exit_wave[:lltcs]
                    if load_one_by_one and cp!=np:
                        measured_chop=measured_array[tcs]
                        if lltcs!=compute_batch:
                            pinned_measured_remain[:]=measured_chop
                            measured=cp.array(pinned_measured_remain)
                        else:
                            pinned_measured[:]=measured_chop
                            measured=cp.array(pinned_measured)
                        measured, *_ = pyptyutils.preprocess_dataset(measured, False, algorithm_type, recon_type, data_shift_vector, data_bin, data_pad*use_padded_intensity, upsample_pattern, data_multiplier, cp, True) ### preprocess
                    else:
                        measured=1*measured_array[tcs]
                        measured, *_ = pyptyutils.preprocess_dataset(measured, False, algorithm_type, recon_type, data_shift_vector, data_bin, data_pad*use_padded_intensity, upsample_pattern, data_multiplier, cp, True)
                    if recon_type=="near_field":
                        if is_single_defocus:
                            defocind=0
                        else:
                            defocind=tcs
                        defocus_near_field=defocus_array[defocind]
                    if is_single_pos:
                        posind=0
                        this_y, this_x=this_pos_array[:,0,:], this_pos_array[:,1,:]
                        this_pos_corr=this_pos_correction
                    else:
                        posind=tcs
                        this_pos=this_pos_array[posind,:]
                        this_pos_corr=this_pos_correction[posind,:]
                        this_y, this_x=cp.split(this_pos, 2, axis=1)
                    if is_single_tilt:
                        tiltind=[0]
                        this_tilt=this_tilt_array
                    else:
                        tiltind=tcs
                        this_tilt=this_tilt_array[tiltind]
                    this_tan_y_before, this_tan_x_before, this_tan_y_inside, this_tan_x_inside,this_tan_y_after, this_tan_x_after=cp.split(this_tilt, 6, axis=1)
                    this_tan_y_inside, this_tan_x_inside=this_tan_y_inside[:,:,:,0], this_tan_x_inside[:,:,:,0]  ## to be changed!
                    masked_pixels_y=this_y+probe_runy
                    masked_pixels_x=this_x+probe_runx
                    this_obj_chopped=this_obj[masked_pixels_y, masked_pixels_x ,:,:] ## chop some vectorized transmission functions
                    this_obj_chopped=pyptyfft.fft2(this_obj_chopped, axes=(1,2), overwrite_x=True)
                    this_obj_chopped*=exclude_mask_ishift[:,:,:,None,None] ## set the cutoff
                    this_obj_chopped=pyptyfft.ifft2(this_obj_chopped, axes=(1,2), overwrite_x=True)
                    this_probe_fourier=None
                    if multiple_scenarios:
                        this_probe=full_probe[probe_marker[tcs], :,:,:]
                    else:
                        this_probe=full_probe[:]
                    if fluctuating_current_flag:
                        this_probe_before_fluctuations=this_probe[:]
                        thisbc=cp.abs(beam_current[tcs])
                        beam_current_values=thisbc[:,None,None,None]
                        this_probe=this_probe*beam_current_values
                    if phase_plate_active:
                        if is_single_phase_plate:
                            indexphase_plate=0
                        else:
                            indexphase_plate=tcs
                        this_phase_plate=cp.asarray(phase_plate_data[indexphase_plate])
                        this_probe_fourier=pyptyfft.fft2(this_probe, axes=(1,2))*this_phase_plate[:,:,:,None]
                        this_probe=pyptyfft.ifft2(this_probe_fourier,  axes=(1,2))
                    if morph_incomming_beam:
                        local_aberrations_phase_plate=local_aberrations_phase_plates[aberration_marker[tcs]]
                        if this_probe_fourier is None: this_probe_fourier=pyptyfft.fft2(this_probe, axes=(1,2));
                        this_fourier_probe_before_local_aberrations=this_probe_fourier*1
                        this_probe=pyptyfft.ifft2(this_fourier_probe_before_local_aberrations*local_aberrations_phase_plate[:,:,:,None], axes=(1,2))
                    if tilt_mode==2 or tilt_mode==4: #before
                        tilting_mask_real_space_before=cp.exp(x_real_grid_tilt*this_tan_x_before+y_real_grid_tilt*this_tan_y_before)
                        this_probe_before_tilt=this_probe[:]
                        this_probe=this_probe_before_tilt*tilting_mask_real_space_before
                    else:
                        tilting_mask_real_space_before=None
                    if probe_shift_flag:
                        if this_probe_fourier is None: this_probe_fourier=pyptyfft.fft2(this_probe, axes=(1,2));
                        shift_probe_mask=cp.exp(shift_probe_mask_x*this_pos_corr[:,1:2,None]+shift_probe_mask_y*this_pos_corr[:,0:1,None])*exclude_mask_ishift
                        this_probe=pyptyfft.ifft2(this_probe_fourier*shift_probe_mask[:,:,:,None], axes=(1,2))
                    if this_probe.shape[0]==1 and lltcs>1:
                        this_probe=cp.repeat(this_probe, lltcs, axis=0)
                    
                   
                    propagator_phase_space, half_propagator_phase_space= master_propagator_phase_space, half_master_propagator_phase_space ### Note for future self: do NOT delete this line, you spent two weeks debuggggging it!
                    ## FORWARD PROPAGATION
                    if propmethod=="multislice":
                        if is_single_dist and not(same_tilt_propagator_flag):
                            propagator_phase_space=cp.exp(-3.141592654j*this_distances[0]*(this_wavelength*q2+2*(qx*this_tan_x_inside+qy*this_tan_y_inside)))*exclude_mask_ishift
                            propagator_phase_space=cp.expand_dims(propagator_phase_space,(-1,-2))
                        
                        waves_multislice, this_exit_wave=pyptymultislice.multislice(this_probe, this_obj_chopped, num_slices, n_obj_modes, n_probe_modes, this_distances, edge_slice_thickness_multiplier, this_wavelength, q2, qx, qy, exclude_mask, is_single_dist, this_tan_x_inside,this_tan_y_inside, damping_cutoff_multislice, smooth_rolloff, propagator_phase_space,  None, exclude_mask_ishift, waves_multislice, this_exit_wave, tilt_mode, this_step_tilts, default_float, default_complex)
                        
                   
                            
                    if tilt_mode==1 or tilt_mode>=3: ## after if tilt mode is 1, 3 or 4
                        tilting_mask_real_space_after=cp.exp(x_real_grid_tilt*this_tan_x_after+y_real_grid_tilt*this_tan_y_after)[:,:,:,:, None]
                        if propmethod=="multislice" and num_slices==1:
                            this_exit_wave=pyptyfft.fft2(this_exit_wave, axes=(1,2), overwrite_x=True)
                            this_exit_wave*=exclude_mask_ishift[:,:,:,None,None]
                            this_exit_wave=pyptyfft.ifft2(this_exit_wave, axes=(1,2), overwrite_x=True)
                        this_exit_wave_before_tilt=this_exit_wave[:]
                        this_exit_wave = this_exit_wave_before_tilt*tilting_mask_real_space_after
                    else:
                        tilting_mask_real_space_after=None
                    ## PROPAGATION FROM THE SPECIMEN PLANE TO THE DETECTOR PLANE
                    if recon_type=="far_field":
                        this_wave=pyptyfft.shift_fft2(this_exit_wave, axes=(1,2))
                        this_wave*=exclude_mask[:,:,:,None,None]
                    else:
                        CTF=(cp.exp(-3.141592654j*(((this_wavelength*q2+2*(qx*this_tan_x_inside+qy*this_tan_y_inside))*defocus_near_field)+(0.5*Cs*this_wavelength**3*q2**2)))*exclude_mask_ishift)[:,:,:,None,None]
                        fourier_exit_wave=pyptyfft.fft2(this_exit_wave, axes=(1,2))
                        this_wave=pyptyfft.ifft2(fourier_exit_wave*CTF, axes=(1,2))
                    this_pattern=cp.abs(this_wave)**2
                    if is_fully_coherent:
                        this_pattern=this_pattern[:,:,:,0,0]
                    else:
                        this_pattern=cp.mean(this_pattern, (-1,-2))
                    if recon_type=="near_field" and alpha_near_field>0: ## the correction to a coherent case
                        E_near_field_not_coherent_correction=(cp.exp(-q2*(3.14152654*alpha_near_field*defocus_near_field)**2))[ None,:,:]
                        this_pattern=cp.real(pyptyfft.ifft2(pyptyfft.fft2(this_pattern, axes=(1,2))*E_near_field_not_coherent_correction, axes=(1,2)))
                    if data_pad!=0 and not(use_padded_intensity):
                        this_pattern=this_pattern[:, data_pad : -data_pad, data_pad : -data_pad]
                    if data_simulation_flag:
                        if data_simulation_exitwave:
                            sim_patterns[tcs]=this_exit_wave[:]
                        else:
                            sim_patterns[tcs]=this_pattern[:]
                    if upsample_pattern!=1:
                        this_pattern=pyptyutils.downsample_something_3d(this_pattern, upsample_pattern, cp)
                    if static_background_is_there:
                        this_pattern+=static_square
                    #### HERE  COMES THE LOSS
                    if algorithm_type == 'lsq_sqrt':
                        sse+= cp.sum((measured-this_pattern)**2)
                        this_pattern_sqrt = cp.sqrt(this_pattern)
                        measured_sqrt     = cp.sqrt(measured)
                        dLoss_dint   = this_pattern_sqrt-measured_sqrt
                        loss_full_pix = dLoss_dint**2
                        loss+=cp.sum(loss_full_pix)
                        
                        this_pattern_sqrt[this_pattern_sqrt==0]=1
                        dLoss_dint/=this_pattern_sqrt
                    else:
                        if algorithm_type=='ml':
                            dLoss_dint=this_pattern-measured
                            sse+=cp.sum(dLoss_dint**2)
                            where_zero=this_pattern==0
                            loss_full_pix=this_pattern - measured*cp.log(this_pattern + where_zero)
                            loss+=cp.sum(loss_full_pix)
                            this_pattern[where_zero]=1
                            dLoss_dint=dLoss_dint/this_pattern
                        if algorithm_type=='lsq':
                            tterm=cp.sum((this_pattern - measured)**2) #, (1,2))
                            loss+= tterm
                            sse+=tterm
                            dLoss_dint=2*(this_pattern-measured)
                        if algorithm_type=='lsq_sqrt_2':
                            term_loss_sse=(measured-this_pattern)**2
                            sse+=cp.sum(term_loss_sse)
                            nonzero_pixels=measured>0.5
                            loss+=0.5 * cp.sum(term_loss_sse[nonzero_pixels]/measured[nonzero_pixels])
                            loss+=cp.sum(this_pattern[(1-nonzero_pixels).astype(bool)])
                            dLoss_dint=cp.ones_like(this_pattern)
                            dLoss_dint[nonzero_pixels]=this_pattern[nonzero_pixels]/measured[nonzero_pixels]-1
                        if algorithm_type=='lsq_compressed':
                            this_coeffs = cp.sum(this_pattern[:,None,:,:]*masks[None,:,:,:], axis=(2,3))
                            this_differences=this_coeffs - measured
                            tterm=cp.sum(this_differences**2)
                            loss+=tterm
                            sse+=tterm
                            this_differences*=2
                            dLoss_dint=cp.sum(masks[None,:,:,:]*this_differences[:,:,None,None], axis=1)
                        if algorithm_type=='gauss_compressed':
                            this_coeffs = cp.sum(this_pattern[:,None,:,:]*masks[None,:,:,:], axis=(2,3))
                            measured_sqrt=cp.sqrt(measured)
                            this_coeffs_sqrt=cp.sqrt(this_coeffs)
                            sse+=cp.sum((this_coeffs-measured)**2)
                            dLoss_dint=this_coeffs_sqrt - measured_sqrt
                            loss+=cp.sum(dLoss_dint**2)
                            this_coeffs_sqrt[this_coeffs_sqrt==0]=1
                            dLoss_dint/=this_coeffs_sqrt
                            dLoss_dint=cp.sum(masks[None,:,:,:]*dLoss_dint[:,:,None,None], axis=1)
                        if algorithm_type=='poisson_compressed':
                            this_coeffs = cp.sum(this_pattern[:,None,:,:]*masks[None,:,:,:], axis=(2,3))
                            dLoss_dint=this_coeffs-measured
                            sse+=cp.sum(dLoss_dint**2)
                            this_coeffs[this_coeffs==0]=1
                            loss+=cp.sum(this_coeffs - measured*cp.log(this_coeffs))
                            dLoss_dint=dLoss_dint/this_coeffs
                            dLoss_dint=cp.sum(masks[None,:,:,:]*dLoss_dint[:,:,None,None], axis=1)
                    if not(is_fully_coherent):
                        dLoss_dint/=(n_probe_modes*n_obj_modes)
                    if this_step_static_background and static_background_is_there:
                        static_background_grad+=cp.sum(dLoss_dint, 0)*2*static_background*this_loss_weight
                    if upsample_pattern!=1:
                        dLoss_dint=pyptyutils.upsample_something_3d(dLoss_dint, upsample_pattern, False, cp)
                    if not(use_padded_intensity) and data_pad!=0:
                        dLoss_dint=cp.pad(dLoss_dint, [[0,0],[data_pad,data_pad],[data_pad, data_pad]])
                    
                    if recon_type=="far_field":
                        dLoss_dWave  =  dLoss_dint[:,:,:,None,None] * this_wave * (this_loss_weight * dLoss_dint.shape[1]* dLoss_dint.shape[2])
                        dLoss_dP_out = pyptyfft.ifft2_ishift(dLoss_dWave, axes=(1,2))
                    else:
                        if alpha_near_field>0:
                            dLoss_dWave=this_wave*cp.expand_dims(cp.real(pyptyfft.ifft2(pyptyfft.fft2(dLoss_dint, axes=(1,2))*E_near_field_not_coherent_correction, axes=(1,2))),(-1,-2))
                        else:
                            dLoss_dWave=this_wave*cp.expand_dims(dLoss_dint, (-1,-2))
                        fouirer_wave_grad=this_loss_weight * pyptyfft.fft2(dLoss_dWave, axes=(1,2))*cp.conjugate(CTF) #(n_m, y,x, mp, mo)
                        dLoss_dP_out=pyptyfft.ifft2(fouirer_wave_grad, axes=(1,2))
                        if this_step_tilts:
                            if tilt_mode==0 or tilt_mode==3 or tilt_mode==4:
                                rsc=-12.566370614359172/(fouirer_wave_grad.shape[1]*fouirer_wave_grad.shape[2])
                                fouirer_wave_grad*=cp.conjugate(fourier_exit_wave)
                                if is_fully_coherent:
                                    dL_dtilt_x=cp.sum(cp.imag(fouirer_wave_grad[:,:,:,0,0]*qx*defocus_near_field), (1,2))
                                    dL_dtilt_y=cp.sum(cp.imag(fouirer_wave_grad[:,:,:,0,0]*qy*defocus_near_field), (1,2))
                                else:
                                    dL_dtilt_x=cp.sum(cp.imag(fouirer_wave_grad[:,:,:,:,:]*qx[:,:,:,None,None]*defocus_near_field), (1,2,3,4))
                                    dL_dtilt_y=cp.sum(cp.imag(fouirer_wave_grad[:,:,:,:,:]*qy[:,:,:,None,None]*defocus_near_field), (1,2,3,4))
                                if is_single_tilt:
                                    dL_dtilt_x=cp.sum(dL_dtilt_x)
                                    dL_dtilt_y=cp.sum(dL_dtilt_y)
                                tilts_grad[tiltind,2]+=dL_dtilt_y*rsc
                                tilts_grad[tiltind,3]+=dL_dtilt_x*rsc
                    if not(tilting_mask_real_space_after is None):
                        dLoss_dP_out = dLoss_dP_out * cp.conjugate(tilting_mask_real_space_after)
                        if this_step_tilts:
                            grad_tilting_kernel = -2 * dLoss_dP_out * cp.conjugate(this_exit_wave_before_tilt)
                            if is_fully_coherent:
                                dL_dtilt=cp.sum(cp.real(grad_tilting_kernel[:,None,:,:,0,0]*yx_real_grid_tilt[:,:,:,:,0]), (2,3), dtype=default_float)#cp.float64).astype(default_float)
                            else:
                                dL_dtilt=cp.sum(cp.real(grad_tilting_kernel[:,None,:,:,:,:]*yx_real_grid_tilt[:,:,:,:,:,None]),(2,3,4,5), dtype=default_float)#cp.float64).astype(default_float)
                            if is_single_tilt:
                                dL_dtilt=cp.sum(dL_dtilt, 0)
                                tilts_grad[0,4:]+=dL_dtilt
                            else:
                                tilts_grad[tiltind,4:]=dL_dtilt
                        if propmethod=="multislice" and num_slices==1:
                            dLoss_dP_out=pyptyfft.fft2(dLoss_dP_out, axes=(1,2), overwrite_x=True)
                            dLoss_dP_out*=exclude_mask_ishift[:,:,:,None,None]
                            dLoss_dP_out=pyptyfft.ifft2(dLoss_dP_out, axes=(1,2), overwrite_x=True)
                    
                    if propmethod=="multislice":
                        object_grad, interm_probe_grad, tilts_grad = pyptymultislice.multislice_grads(dLoss_dP_out, waves_multislice, this_obj_chopped, object_grad, tilts_grad, is_single_dist,this_distances, edge_slice_thickness_multiplier, exclude_mask, this_wavelength, q2, qx, this_tan_x_inside, qy, this_tan_y_inside, num_slices, n_obj_modes,tiltind, propagator_phase_space, this_step_tilts, damping_cutoff_multislice, smooth_rolloff, tilt_mode,  compute_batch, exclude_mask_ishift, this_step_probe, this_step_obj, this_step_pos_correction, masked_pixels_y, masked_pixels_x, default_float, default_complex, helper_flag_4)
                  
                    if helper_flag_4:
                        fourier_probe_grad=None
                        if probe_shift_flag:
                            fourier_probe_grad=pyptyfft.fft2(interm_probe_grad, axes=(1,2))
                            fourier_probe_grad=fourier_probe_grad*cp.conjugate(shift_probe_mask)[:,:,:,None]
                            interm_probe_grad=pyptyfft.ifft2(fourier_probe_grad, axes=(1,2))
                            if this_step_pos_correction:
                                shift_mask_grad=cp.conjugate(this_probe_fourier)*fourier_probe_grad
                                sh = -2/(this_probe_fourier.shape[1]*this_probe_fourier.shape[2]) ## -1 for conj, 2 for real-grad shape for ifft
                                if n_probe_modes==1:
                                    sh_grad = sh*cp.sum(cp.real(shift_probe_mask_yx[:,:,:,:]*shift_mask_grad[:,None,:,:,0]), (2,3), dtype=default_float)#.astype(default_float)
                                else:
                                    sh_grad = sh*cp.sum(cp.real(shift_probe_mask_yx[:,:,:,:,None]*shift_mask_grad[:,None,:,:,:]), (2,3,4), dtype=default_float)#.astype(default_float)
                                if is_single_pos:
                                    sh_grad=cp.sum(sh_grad, 0)
                                    pos_grad[posind,:]+=sh_grad
                                else:
                                    pos_grad[posind,:]=sh_grad
                        if helper_flag_3:
                            if not(tilting_mask_real_space_before is None):
                                interm_probe_grad*=cp.conjugate(tilting_mask_real_space_before)
                                if this_step_tilts:
                                    grad_tilting_kernel=-2 * interm_probe_grad * cp.conjugate(this_probe_before_tilt)
                                    if n_probe_modes==1:
                                        dL_dtilt=cp.sum(cp.real(grad_tilting_kernel[:,None,:,:,0]*yx_real_grid_tilt[:,:,:,0]),(2,3),  dtype=default_float)#cp.float64).astype(default_float)
                                    else:
                                        dL_dtilt=cp.sum(cp.real(grad_tilting_kernel[:,None,:,:,:]*yx_real_grid_tilt),(2,3,4),  dtype=default_float)#cp.float64).astype(default_float)
                                    if is_single_tilt:
                                        dL_dtilt=cp.sum(dL_dtilt, 0)
                                        tilts_grad[0,:2]+=dL_dtilt
                                    else:
                                        tilts_grad[tiltind,:2]=dL_dtilt
                            if helper_flag_2:
                                if morph_incomming_beam:
                                    if fourier_probe_grad is None: fourier_probe_grad=pyptyfft.fft2(interm_probe_grad, axes=(1,2));
                                    fourier_probe_grad*=cp.conjugate(local_aberrations_phase_plate[:,:,:,None])
                                    interm_probe_grad=pyptyfft.ifft2(fourier_probe_grad, axes=(1,2))
                                    if this_step_aberrations_array:
                                        sh = 2/(fourier_probe_grad.shape[1]*fourier_probe_grad.shape[2])
                                        defgr=sh*cp.sum(cp.real((fourier_probe_grad*cp.conjugate(this_fourier_probe_before_local_aberrations))[:,None, :,:,:] * cp.conjugate(aberrations_polynomials[None,:,:,:,None])), axis=(2,3,4), dtype=default_float)
                                        scatteradd_abers(aberrations_array_grad, aberration_marker[tcs], defgr)
                                        #for dumbindex, t in enumerate(tcs):
                                         #   aberrations_array_grad[aberration_marker[t],:]+=defgr[dumbindex,:]
                                if helper_flag_1:
                                    if phase_plate_active:
                                        if fourier_probe_grad is None: fourier_probe_grad=pyptyfft.fft2(interm_probe_grad, axes=(1,2));
                                        cp.conjugate(this_phase_plate, out=this_phase_plate)
                                        interm_probe_grad=pyptyfft.ifft2(fourier_probe_grad*this_phase_plate[:,:,:,None], (1,2))
                                    if fluctuating_current_flag:
                                        if this_beam_current_step:
                                            if n_probe_modes==1:
                                                cp.conjugate(this_probe_before_fluctuations, out=this_probe_before_fluctuations)
                                                
                                                beam_current_grad[tcs]=2*cp.sign(thisbc)*cp.sum(cp.real(interm_probe_grad[:,:,:,0]*this_probe_before_fluctuations[:,:,:,0]), (1,2), dtype=default_float)#cp.float64).astype(default_float)
                                            else:
                                                beam_current_grad[tcs]=2*cp.sign(thisbc)*cp.sum(cp.real(interm_probe_grad*cp.conjugate(this_probe_before_fluctuations)), (1,2,3))
                                        if this_step_probe:
                                            interm_probe_grad=interm_probe_grad*beam_current_values
                                    if this_step_probe:
                                        if multiple_scenarios:
                                            scatteradd_probe(probe_grad, probe_marker[tcs], interm_probe_grad)
                                        else:
                                            probe_grad[0]+=cp.sum(interm_probe_grad,0)
                if smart_memory:
                    try:
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()
                    except:
                        pass
        return loss, sse, object_grad, probe_grad, pos_grad, tilts_grad, static_background_grad, beam_current_grad, aberrations_array_grad
    
    ############################################################################################################
    ######################################## End of the Pattern Loop ###########################################
    ############################################################################################################
    thread_results = [None] * devcount
    def thread_wrapper(device_ind):
        thread_results[device_ind] = worker(device_ind)
    threads = []
    for device_ind in range(devcount):
        t = threading.Thread(target=thread_wrapper, args=(device_ind,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    
    
    loss, sse, object_grad, probe_grad, pos_grad, tilts_grad, static_background_grad, beam_current_grad, aberrations_array_grad = thread_results[0] # thread 0 results are the base
    for r in thread_results[1:]: ## if mutltiGPU -> append the thread results to first one
        main_device=cp.cuda.Device(0)
        with main_device:
            loss_1, sse_1, object_grad_1, probe_grad_1, pos_grad_1, tilts_grad_1, static_background_grad_1, beam_current_grad_1, aberrations_array_grad_1 = r
            loss += loss_1
            sse += sse_1
            object_grad += object_grad_1
            probe_grad += probe_grad_1
            pos_grad += pos_grad_1
            tilts_grad += tilts_grad_1
            static_background_grad += static_background_grad_1
            beam_current_grad += beam_current_grad_1
            aberrations_array_grad += aberrations_array_grad_1

    loss*=this_loss_weight
    if data_simulation_flag:
        try:
            np.save(data_simulation_flag, sim_patterns.get())
        except:
            np.save(data_simulation_flag, sim_patterns)
    constraint_contributions=[]
    loss_print_copy=1*loss;
    this_pos_array=this_pos_array[:,:,0]
    this_tilt_array=this_tilt_array[:,:,0,0]
    
    if this_step_probe:
        if multiple_scenarios:
            probe_grad=cp.moveaxis(probe_grad, 0,3)
        else:
            probe_grad=probe_grad[0]
    if multiple_scenarios: ## put things back where they were!
        full_probe=cp.moveaxis(full_probe,3,0)
    else:
        full_probe=full_probe[0, :,:,:]

        
    #######
    if this_step_pos_correction and fast_axis_reg_weight_positions!=0:
        something=this_pos_array+this_pos_correction
        ind_loss, reg_grad=compute_fast_axis_constraint_on_grid(something, scan_size, fast_axis_reg_weight_positions)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Positions fast axis constaint is %.2e %% of the main loss"%(fast_axis_reg_weight_positions, ind_loss*100/loss_print_copy));
        pos_grad+=reg_grad
        loss+=ind_loss
        constraint_contributions.append(ind_loss)
    else:
        constraint_contributions.append(0)
    #######
    if this_step_pos_correction and current_deformation_reg_weight_positions!=0:
        something=this_pos_array+this_pos_correction
        ind_loss, reg_grad = compute_deformation_constraint_on_grid(something, scan_size, current_deformation_reg_weight_positions)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Positions deformation constaint is %.2e %% of the main loss"%(current_deformation_reg_weight_positions, ind_loss*100/loss_print_copy));
        pos_grad+=reg_grad;
        loss+=ind_loss
        constraint_contributions.append(ind_loss)
    else:
        constraint_contributions.append(0)
    ########
    if this_step_tilts and current_deformation_reg_weight_tilts!=0:
        something=this_tilt_array
        ind_loss, reg_grad=compute_deformation_constraint_on_grid(something, scan_size, current_deformation_reg_weight_tilts)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Tilts deformation constaint is %.2e %% of the main loss"%(current_deformation_reg_weight_tilts, ind_loss*100/loss_print_copy));
        tilts_grad+=reg_grad;
        loss+=ind_loss
        constraint_contributions.append(ind_loss)

    else:
        constraint_contributions.append(0)
    #######
    if this_step_tilts and fast_axis_reg_weight_tilts!=0:
        something=this_tilt_array
        ind_loss, reg_grad=compute_fast_axis_constraint_on_grid(something, scan_size, fast_axis_reg_weight_tilts)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Tilts fast axis constaint is %.2e %% of the main loss"%(fast_axis_reg_weight_tilts, ind_loss*100/loss_print_copy));
        tilts_grad+=reg_grad
        loss+=ind_loss
        constraint_contributions.append(ind_loss)
      
    else:
        constraint_contributions.append(0)
    #######
    if phase_norm_weight!=0: # l_1 norm of the potential
        l1_reg_term, l1_object_grad=compute_full_l1_constraint(this_obj, 0, phase_norm_weight, None, True, smart_memory)
        if print_flag==4:
            sys.stdout.write("\nWith phase weight of %.3e l1 constaint is %.2e %% of the main loss"%(phase_norm_weight, l1_reg_term*100/loss_print_copy));
        loss+=l1_reg_term
        object_grad+=l1_object_grad
        constraint_contributions.append(l1_reg_term)
        del l1_reg_term,l1_object_grad # forget about it
    else:
        constraint_contributions.append(0)
    if abs_norm_weight!=0: # l_1 norm of the absorp. potential
        l1_reg_term, l1_object_grad=compute_full_l1_constraint(this_obj, abs_norm_weight, 0, None, True, smart_memory)
        if print_flag==4:
            sys.stdout.write("\nWith abs weight of %.3e l1 constaint (abs) is %.2e %% of the main loss"%(abs_norm_weight, l1_reg_term*100/loss_print_copy));
        loss+=l1_reg_term
        object_grad+=l1_object_grad
        constraint_contributions.append(l1_reg_term)
        del l1_reg_term,l1_object_grad # forget about it
    else:
        constraint_contributions.append(0)
    ######
    if probe_reg_weight!=0 and this_step_probe:
        probe_reg_term, reg_probe_grad = compute_probe_constraint(full_probe, aperture_mask, probe_reg_weight, True)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Probe recprocal-space constaint is %.2e %% of the main loss"%(probe_reg_weight, probe_reg_term*100/loss_print_copy));
        #print("Debug:", reg_probe_grad.shape, probe_grad.shape, aperture_mask.shape, full_probe.shape)
        loss+=probe_reg_term
        probe_grad+=reg_probe_grad
        constraint_contributions.append(probe_reg_term)
        del reg_probe_grad, probe_reg_term
    else:
        constraint_contributions.append(0)
    ########
    if this_step_probe and current_window_weight!=0:
        probe_reg_term, reg_probe_grad = compute_window_constraint(full_probe, current_window, current_window_weight)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Probe real-space constaint is %.2e %% of the main loss"%(current_window_weight, probe_reg_term*100/loss_print_copy));
        loss+=probe_reg_term
        probe_grad+=reg_probe_grad
        constraint_contributions.append(probe_reg_term)
        del reg_probe_grad, probe_reg_term #forget about it
        
    else:
        constraint_contributions.append(0)
    ####
    if atv_weight!=0:
        atv_reg_term, atv_object_grad = compute_atv_constraint(this_obj, atv_weight, atv_q, atv_p, pixel_size_x_A, pixel_size_y_A, None, True, smart_memory)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, ATV constaint is %.2e %% of the main loss"%(atv_weight, atv_reg_term*100/loss_print_copy));
        loss+=atv_reg_term
        object_grad+=atv_object_grad
        constraint_contributions.append(atv_reg_term)
        del atv_object_grad, atv_reg_term
    else:
        constraint_contributions.append(0)
    #######
    if mixed_variance_weight!=0 and this_obj.shape[-1]>1:
        mixed_variance_reg_term, mixed_variance_grad=compute_mixed_object_variance_constraint(this_obj, mixed_variance_weight, mixed_variance_sigma, True, smart_memory)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Mixed variance constaint is %.2e %% of the main loss"%(mixed_variance_weight, mixed_variance_reg_term*100/loss_print_copy));
        loss+=mixed_variance_reg_term
        object_grad+=mixed_variance_grad
        constraint_contributions.append(mixed_variance_reg_term)
        del mixed_variance_reg_term, mixed_variance_grad # forget about it
    else:
        constraint_contributions.append(0)
    #######
    if wedge_mu!=0:
        mw_reg_term,mw_reg_grad = compute_missing_wedge_constraint(this_obj, pixel_size_x_A, pixel_size_y_A, this_distances[0], beta_wedge, wedge_mu)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Missing wedge constaint is %.2e %% of the main loss"%(wedge_mu, mw_reg_term*100/loss_print_copy))
        loss+=mw_reg_term
        object_grad+=mw_reg_grad
        constraint_contributions.append(mw_reg_term)
        del mw_reg_term, mw_reg_grad # forget about it
    else:
        constraint_contributions.append(0)
    ########
    if this_step_pos_correction and slow_axis_reg_weight_positions!=0:
        something=this_pos_array+this_pos_correction
        ind_loss, reg_grad=compute_slow_axis_constraint_on_grid(something, scan_size, slow_axis_reg_weight_positions)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Positions slow axis constaint is %.2e %% of the main loss"%(slow_axis_reg_weight_positions, ind_loss*100/loss_print_copy));
        pos_grad+=reg_grad
        loss+=ind_loss
        constraint_contributions.append(ind_loss)
    else:
        constraint_contributions.append(0)
    #######
    if this_step_tilts and slow_axis_reg_weight_tilts!=0:
        something=this_tilt_array
        ind_loss, reg_grad=compute_slow_axis_constraint_on_grid(something, scan_size, slow_axis_reg_weight_tilts)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Tilts slow axis constaint is %.2e %% of the main loss"%(slow_axis_reg_weight_tilts, ind_loss*100/loss_print_copy));
        tilts_grad+=reg_grad
        loss+=ind_loss
        constraint_contributions.append(ind_loss)
    else:
        constraint_contributions.append(0)
    #######
    if this_step_probe and probe_mod_reg_weight!=0:
        probe_reg_term, reg_probe_grad = probe_modulation_constraint(full_probe, scan_step_A, pixel_size_x_A, pixel_size_y_A, probe_mod_tol, probe_mod_reg_weight, damping_cutoff_multislice)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Probe modulation constaint is %.2e %% of the main loss"%(probe_mod_reg_weight, probe_reg_term*100/loss_print_copy));
        loss+=probe_reg_term
        probe_grad+=reg_probe_grad
        constraint_contributions.append(probe_reg_term)
        del reg_probe_grad, probe_reg_term #forget about it
    else:
        constraint_contributions.append(0)
    
    ######
    if print_flag==4:
        sys.stdout.flush()
    if loss!=loss:
        raise ValueError('A very specific bad thing. Loss is Nan.')
    if this_step_probe:
        probe_grad=pyptyfft.fft2(probe_grad, (0,1), overwrite_x=True)
        if multiple_scenarios:
            probe_grad*=exclude_mask_ishift[0,:,:,None, None]
        else:
            probe_grad*=exclude_mask_ishift[0,:,:,None]
        probe_grad=pyptyfft.ifft2(probe_grad, (0,1), overwrite_x=True)
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except:
        pass
    
    if gpu_loss_flag:
        object_grad=object_grad.get()
        probe_grad=probe_grad.get()
        pos_grad=pos_grad.get()
        tilts_grad=tilts_grad.get()
        loss = loss.get()
        sse  = sse.get()
        try:
            static_background_grad=static_background_grad.get()
        except:
            pass
        try:
            aberrations_array_grad=aberrations_array_grad.get()
        except:
            pass
        try:
            beam_current_grad=beam_current_grad.get()
        except:
            pass
    for dumbi1 in range(len(constraint_contributions)):
        try:
            constraint_contributions[dumbi1]=constraint_contributions[dumbi1].get()
        except:
            pass
    
    constraint_contributions=np.array(constraint_contributions)
    
    return loss, sse, object_grad,  probe_grad, pos_grad, tilts_grad, static_background_grad, aberrations_array_grad, beam_current_grad, constraint_contributions



        
def scatteradd_probe(full, indic, batches):
    """
    Adds batched probe updates to their respective positions in the full probe array. This wrapper is needed to support older CuPy version.

    Parameters
    ----------
    full : ndarray
        Full probe gradient array.
    indic : array_like
        Indices specifying where to add each batch.
    batches : ndarray
        Batched gradients to scatter-add.
    """
    try:
        cp.add.at(full, (indic), batches)
    except:
        cupyx.scatter_add(full.real, (indic), batches.real)
        cupyx.scatter_add(full.imag, (indic), batches.imag)
        
        
        
def scatteradd_abers(full, indic, batches):
    """
    Adds batched aberration updates to their respective positions in the full aberration array. This wrapper is needed to support older CuPy version.

    Parameters
    ----------
    full : ndarray
        Full aberration gradient array.
    indic : array_like
        Indices specifying where to add each batch.
    batches : ndarray
        Batched gradients to scatter-add.
    """
    try:
        cp.add.at(full, (indic), batches)
    except:
        cupyx.scatter_add(full, (indic), batches)
        
        
        
        
def charge_flip(a, delta_phase = 0.03, delta_abs = 0.14, beta_phase = -0.95, beta_abs = -0.95, fancy_sigma=None, xp=cp):
    """
    Perform charge-flipping style object regularization to enhance phase and absorption contrast.

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
    """
    phase=xp.angle(a)
    absorption=-xp.log(xp.abs(a)+1e-10)
    def fftconvolve(a,b):
        return pyptyfft.fftshift(pyptyfft.ifftn(pyptyfft.fftn(a, xp=xp) * pyptyfft.fftn(b, xp=xp), xp=xp), xp=xp).real
    def richardson_lucy(image, psf, tolerance):
        result = xp.copy(image)
        count, doflag, prev_estimate=0, True, None
        while doflag:
            convolved_result = fftconvolve(result, psf)
            relative_blur = image / convolved_result
            error_estimate=fftconvolve(relative_blur, xp.conjugate(psf))
            if count>=1:
                progress=xp.sum((error_estimate-prev_estimate)**2)
                if progress<=tolerance:
                    doflag=False
            prev_estimate=error_estimate
            count+=1
            result *= error_estimate
        return result
    if not(fancy_sigma is None):
        do_phase_things=not(fancy_sigma[0] is None)
        do_abs_things=not(fancy_sigma[1] is None)
        shx,shy=phase.shape[1], phase.shape[0]
        r2=xp.sum(xp.array(xp.meshgrid(xp.arange(-shx//2, shx+(-shx//2), 1), xp.arange(-shy//2, shy+(-shy//2), 1), indexing="xy"))**2, 0)
        if do_phase_things:
            sigma_1_phase=fancy_sigma[0][0]
            sigma_2_phase=fancy_sigma[0][1]
            do_richardson_lucy_phase=type(sigma_1_phase)==str
            sigma_1_phase=float(sigma_1_phase)
            psf_1_phase=xp.exp(-r2/sigma_1_phase**2)
            psf_2_phase=xp.exp(-r2/sigma_2_phase**2)
            psf_1_phase/=xp.sum(psf_1_phase)
            psf_2_phase/=xp.sum(psf_2_phase)
            
            if not(do_richardson_lucy_phase):
                psf_1_phase=pyptyfft.shift_fft2(psf_1_phase, axes=(0,1), xp=xp) #+1e-20
        if do_abs_things:
            sigma_1_abs=fancy_sigma[1][0]
            sigma_2_abs=fancy_sigma[1][1]
            do_richardson_lucy_abs=type(sigma_1_abs)==str
            sigma_1_abs=float(sigma_1_abs)
            psf_1_abs =xp.exp(-r2/sigma_1_abs**2)
            psf_2_abs =xp.exp(-r2/sigma_2_abs**2)
            psf_1_abs/=xp.sum(psf_1_abs)
            psf_2_abs/=xp.sum(psf_2_abs)
            if not(do_richardson_lucy_abs):
                psf_1_abs=pyptyfft.shift_fft2(psf_1_abs, axes=(0,1), xp=xp)#+1e-20
        for i in range(phase.shape[-1]):
            for j in range(phase.shape[-2]):
                if do_phase_things:
                    phase[:,:,j,i]-=xp.min(phase[:,:,j,i])
                    if do_richardson_lucy_phase:
                        phase[:,:,j,i]=richardson_lucy(phase[:,:,j,i], psf_1_phase, tolerance=1e-3)
                    else:
                        phase[:,:,j,i]=xp.real(pyptyfft.ifft2_ishift(pyptyfft.shift_fft2(phase[:,:,j,i], xp=xp)/psf_1_phase, xp=xp))
            
                if do_abs_things:
                    absorption[:,:,j,i]-=xp.min(absorption[:,:,j,i])
                    if do_richardson_lucy_abs:
                        absorption[:,:,j,i]=richardson_lucy(absorption[:,:,j,i], psf_1_abs, tolerance=1e-3)
                    else:
                        absorption[:,:,j,i]=xp.real(pyptyfft.fftshift(pyptyfft.ifft2_ishift(pyptyfft.shift_fft2(absorption[:,:,j,i], xp=xp)/psf_1_abs, xp=xp), xp=xp))
    for i in range(phase.shape[-1]):
        for j in range(phase.shape[-2]):
            p=phase[:,:,j,i]
            a=absorption[:,:,j,i]
            p[p < delta_phase*xp.max(p)] *= beta_phase
            a[a < delta_abs*xp.max(a)] *= beta_abs
            phase[:,:,j,i]=p
            absorption[:,:,j,i]=a
    if not(fancy_sigma is None):
        for i in range(phase.shape[-1]):
            for j in range(phase.shape[-2]):
                if do_phase_things:
                    phase[:,:,j,i]=fftconvolve(phase[:,:,j,i], psf_2_phase)
                if do_abs_things:
                    absorption[:,:,j,i]=fftconvolve(absorption[:,:,j,i], psf_2_abs)
    return xp.exp(-absorption+1j*phase)



def make_states_orthogonal(probe_states, xp=cp):
    """
    Apply Gram-Schmidt orthogonalization to probe modes.

    Parameters
    ----------
    probe_states : ndarray
        Probe array with multiple modes.

    Returns
    -------
    ndarray
        Orthogonalized probe states.
    """
    n_states=probe_states.shape[-1]
    multiple_scenarios=len(probe_states.shape)==4
    for ind1 in range(n_states):
        for ind2 in range(ind1):
            if multiple_scenarios:
                probe_states[:,:,ind1,:]=probe_states[:,:,ind1,:]-probe_states[:,:,ind2,:]*(xp.sum(xp.conjugate(probe_states[:,:,ind2,:])*probe_states[:,:,ind1,:]))/(1e-10+xp.sum(xp.abs(probe_states[:,:,ind2,:])**2))
            else:
                probe_states[:,:,ind1]=probe_states[:,:,ind1]-probe_states[:,:,ind2]*(xp.sum(xp.conjugate(probe_states[:,:,ind2])*probe_states[:,:,ind1]))/(1e-10+xp.sum(xp.abs(probe_states[:,:,ind2])**2))
    return probe_states
    
def make_basis_orthogonal(vectors, xp=cp):
    """
    Orthogonalize a set of 1D basis vectors using Gram-Schmidt.

    Parameters
    ----------
    vectors : ndarray
        2D array of vectors to orthogonalize.

    Returns
    -------
    ndarray
        Orthogonalized basis.
    """
    n=vectors.shape[0]
    for ind1 in range(n):
        for ind2 in range(ind1):
            vectors[ind1]=vectors[ind1]-vectors[ind2]*(xp.sum(xp.conjugate(vectors[ind2])*vectors[ind1]))/(1e-20+xp.sum(xp.abs(vectors[ind2])**2))
    return vectors



def compute_fast_axis_constraint_on_grid(something, scan_size, tv_reg_weight):
    """
    Apply second-order TV regularization along the fast scan axis.

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
    """
    something_scan_size = something.reshape(scan_size[0], scan_size[1],something.shape[-1])
    grad=cp.zeros_like(something_scan_size)
    something_x_roll_p1=cp.roll(something_scan_size,  1, axis=1)
    something_x_roll_m1=cp.roll(something_scan_size, -1, axis=1)
    laplace=something_x_roll_p1+something_x_roll_m1 -2*something_scan_size
    laplace[:, 0]=0
    laplace[:,-1]=0
    reg_term=tv_reg_weight*cp.sum(laplace**2)
    laplace*=2*tv_reg_weight
    grad+=-2*laplace
    grad+=cp.roll(laplace,   1, axis=1)
    grad+=cp.roll(laplace,  -1, axis=1)
    grad=grad.reshape(something.shape)
    return reg_term, grad


def compute_slow_axis_constraint_on_grid(something, scan_size, tv_reg_weight):
    """
    Apply second-order TV regularization along the slow scan axis.

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
    """
    something_scan_size = something.reshape(scan_size[0], scan_size[1],something.shape[-1])
    something_scan_size_fast_avg=cp.sum(something_scan_size, axis=1)
    
    grad_fast_avg=cp.zeros_like(something_scan_size_fast_avg)
    
    something_y_roll_p1=cp.roll(something_scan_size_fast_avg,  1, axis=0)
    something_y_roll_m1=cp.roll(something_scan_size_fast_avg, -1, axis=0)
    laplace=something_y_roll_p1+something_y_roll_m1 -2*something_scan_size_fast_avg
    laplace[0,:]=0
    laplace[-1,:]=0
    reg_term=tv_reg_weight*cp.sum(laplace**2)
    laplace*=2*tv_reg_weight
    grad_fast_avg+=-2*laplace
    grad_fast_avg+=cp.roll(laplace,   1, axis=0)
    grad_fast_avg+=cp.roll(laplace,  -1, axis=0)
    grad_fast_avg=cp.repeat(grad_fast_avg[:,None,:], scan_size[1], axis=1)
    grad_fast_avg=grad_fast_avg.reshape(something.shape)
    return reg_term, grad_fast_avg


    
    
def compute_deformation_constraint_on_grid(something, scan_size, reg_weight):
    """
    Penalize deviations from affine transformations in local scan patches.

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
    """
    x_perfect, y_perfect = cp.meshgrid(cp.arange(scan_size[1]), cp.arange(scan_size[0]))
    yxo_perf = cp.stack([y_perfect.flatten(), x_perfect.flatten(), cp.ones(y_perfect.size)], axis=1)
    matrix_1=yxo_perf@(cp.linalg.inv(yxo_perf.T @ yxo_perf)).T
    reg_term=0
    grad_full=cp.zeros_like(something)
    for subdumb in range(something.shape[1]//2):
        something_chop=something[:,subdumb*2:2*(subdumb+1)]
        matrix_2= yxo_perf.T @ something_chop
        deformation_desired=matrix_1 @ matrix_2
        
        grad= something_chop-deformation_desired
        reg_term+=cp.sum(grad**2)*reg_weight
        grad*=2*reg_weight
        dR_dDef=-1*grad
        dR_dDef=yxo_perf @ (matrix_1.T @ dR_dDef)
        grad_full[:,subdumb*2:2*(subdumb+1)]=grad+dR_dDef
    return reg_term, grad_full

    
    
    
    
    
def compute_full_l1_constraint(object, abs_norm_weight, phase_norm_weight, grad_mask, return_direction, smart_memory):
    """
    Apply L1 norm regularization to the object's phase and absorption.

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
    """
    reg_term=0
    if return_direction:
        grad=cp.zeros_like(object)
    else:
        grad=None
    if phase_norm_weight>0:
        this_potential=cp.angle(object)#*grad_mask[:,:,None,None]
        reg_term+=phase_norm_weight*cp.sum(cp.abs(this_potential))
        if return_direction:
            grad+=1j*phase_norm_weight*cp.sign(this_potential)
    if abs_norm_weight>0:
        this_abs_potential=cp.log(cp.abs(object))#*grad_mask[:,:,None,None] # actually a negative of it, but its irrelevant for us!
        reg_term+=abs_norm_weight*cp.sum(cp.abs(this_abs_potential))
        if return_direction:
            grad+=abs_norm_weight*cp.sign(this_abs_potential)
    if return_direction:
        grad=0.5*grad/cp.conjugate(object)
    return reg_term, grad
    
def compute_window_constraint(to_reg_probe, current_window, current_window_weight):
    """
    Penalize probe values outside a predefined window region in real-space.

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
    """
    to_reg_probe_2=cp.abs(to_reg_probe)**2
    if len(to_reg_probe.shape)==3:
        current_window=current_window[:,:,None]
    else:
        current_window=current_window[:,:,None, None]
    current_window*=current_window_weight
    to_reg_probe_2_window=to_reg_probe_2*current_window
    reg_term=cp.sum(to_reg_probe_2_window)
    reg_grad=current_window*to_reg_probe
    return reg_term, reg_grad
           
    
    
def compute_probe_constraint(to_reg_probe, aperture, weight, return_direction):
    """
    Apply reciprocal space constraint to the probe using an aperture mask. Penalize probe values outside an aperture.

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
    """
    if type(aperture)==cp.ndarray:
        aperture=1-cp.expand_dims(aperture.astype(bool),-1) ### actually a mask
    else:
        ## if aperture is a float, then you should construct a circular mask yourself! Here the aperture is supposed to be a ratio between the convergence and collection angles!
        ffx=pyptyfft.fftshift(pyptyfft.fftfreq(to_reg_probe.shape[1]))
        ffy=pyptyfft.fftshift(pyptyfft.fftfreq(to_reg_probe.shape[0]))
        ffx,ffy=cp.meshgrid(ffx,ffy, indexing="xy")
        ffr=cp.expand_dims((ffx**2+ffy**2)**0.5,-1)
        aperture=ffr >(0.5*aperture)
        del ffx,ffy, ffr
    if len(to_reg_probe.shape)==4:
        aperture=cp.expand_dims(aperture,-1)
    probe_fft=pyptyfft.shift_fft2(to_reg_probe, axes=(0,1))
    probe_fft= probe_fft * aperture
    reg_term=weight*cp.sum(cp.abs(probe_fft))
    probe_fft=aperture*cp.exp(1j*cp.angle(probe_fft))* (0.5*weight*probe_fft.shape[0]*probe_fft.shape[1])
    probe_fft=pyptyfft.ifft2_ishift(probe_fft, axes=(0,1))
    return reg_term, probe_fft
    
    

def probe_modulation_constraint(to_reg_probe, scan_step_A, pixel_size_x_A, pixel_size_y_A, probe_mod_tol=0.01, probe_mod_reg_weight=1, damping_cutoff_multislice=0.63):
    probe_fft=pyptyfft.shift_fft2(to_reg_probe, axes=(0,1))
    
    probe_fft_abs=cp.abs(probe_fft)
    
    ffx=pyptyfft.fftshift(pyptyfft.fftfreq(to_reg_probe.shape[1], pixel_size_x_A))*scan_step_A
    ffy=pyptyfft.fftshift(pyptyfft.fftfreq(to_reg_probe.shape[0], pixel_size_y_A))*scan_step_A
    ffx, ffy=cp.meshgrid(ffx,ffy, indexing="xy")
    max_int_frequency=int(cp.ceil(0.5*scan_step_A/(pixel_size_x_A)))
    
    target_freq_x = cp.arange(-max_int_frequency,max_int_frequency+1,1)
    target_freq_x, target_freq_y = cp.meshgrid(target_freq_x,target_freq_x, indexing="xy")
    target_freq_r = (target_freq_x**2+target_freq_y**2)<=(damping_cutoff_multislice*max_int_frequency)**2
    target_freq_x, target_freq_y=target_freq_x[target_freq_r], target_freq_y[target_freq_r]
    
    laplace = ((target_freq_x[:,None,None] - ffx[None,:,:])**2 + (target_freq_y[:,None,None] - ffy[None,:,:])**2) ## this is a distance to peaks
    laplace = ((laplace <= probe_mod_tol**2).sum(0)).astype(bool) ## this is a mask
    
    if len(to_reg_probe.shape)==3:
        laplace=laplace[:,:,None]
    else:
        laplace=laplace[:,:,None, None]
    

    laplace = (laplace * (-4*probe_fft_abs + cp.roll(probe_fft_abs, 1, axis=1) + cp.roll(probe_fft_abs, -1, axis=1) + cp.roll(probe_fft_abs, 1, axis=0) +  cp.roll(probe_fft_abs, -1, axis=0))) ## now this is the laplace we need
    
    reg_term = probe_mod_reg_weight * cp.sum(laplace**2)
    
    laplace*=2*probe_mod_reg_weight
    
    probe_fft_abs=-4*laplace + cp.roll(laplace, 1, axis=1) + cp.roll(laplace, -1, axis=1) + cp.roll(laplace, 1, axis=0) + cp.roll(laplace, -1, axis=0)
    
    probe_fft_abs=probe_fft_abs*0.5*cp.exp(1j*cp.angle(probe_fft))
    
    probe_fft_abs=pyptyfft.ifft2_ishift(probe_fft_abs, axes=(0,1))*(to_reg_probe.shape[0])*(to_reg_probe.shape[1])
    
    return reg_term, probe_fft_abs
    
    

def compute_atv_constraint(obj, atv_weight, atv_q, atv_p, pixel_size_x_A, pixel_size_y_A, atv_grad_mask, return_direction, smart_memory):
    """
    Apply adaptive total variation (ATV) regularization to the object.

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
    """
    dx =cp.roll(obj, 1, axis=1)
    dy=cp.roll(obj, 1, axis=0)
    dx-=obj
    dy-=obj
    dx_abs=cp.abs(dx)
    dy_abs=cp.abs(dy)
    if not(atv_grad_mask is None):
        dx_abs = dx_abs*atv_grad_mask[:,:,None,None]
        dy_abs = dy_abs*atv_grad_mask[:,:,None,None]
    dx=cp.exp(1j*cp.angle(dx))
    dy=cp.exp(1j*cp.angle(dy))
    term=(dx_abs**atv_p)+(dy_abs**atv_p)+1e-20
    reg_term=atv_weight*cp.sum(term**(atv_q/atv_p))
    if return_direction:
        dR_dTerm=atv_weight*0.5*atv_q*(term**((atv_q/atv_p) - 1))
        if not(atv_grad_mask is None):
            dR_dTerm = dR_dTerm*atv_grad_mask[:,:,None,None]
        dx_abs=dR_dTerm*(dx_abs**(atv_p-1))*dx
        dy_abs=dR_dTerm*(dy_abs**(atv_p-1))*dy
        dR_dTerm=cp.zeros_like(obj)
        dR_dTerm+=cp.roll(dx_abs, -1, axis=1)
        dR_dTerm-=dx_abs
        dR_dTerm+=cp.roll(dy_abs, -1, axis=0)
        dR_dTerm-=dy_abs
    else:
        dR_dTerm=cp.zeros_like(obj)
    return reg_term, dR_dTerm


def compute_missing_wedge_constraint(obj, pixel_size_x_A, pixel_size_y_A, slice_distance, beta_wedge, wedge_mu):
    """
    Enforce missing wedge constraint in 3D reciprocal space.

    Parameters
    ----------
    obj : ndarray
        3D complex object.
    pixel_size_x_A : float
        Pixel size along x (Å).
    pixel_size_y_A : float
        Pixel size along y (Å).
    slice_distance : float
        Slice spacing (Å).
    beta_wedge : float
        Cone sharpness.
    wedge_mu : float
        Regularization weight.
    Returns
    -------
    loss_term : float
        Regularization loss.
    grad_obj : ndarray
        Gradient of the loss with respect to the object.
    """
    qx=pyptyfft.fftfreq(obj.shape[1],pixel_size_x_A)
    qy=pyptyfft.fftfreq(obj.shape[0],pixel_size_y_A)
    qz=pyptyfft.fftfreq(obj.shape[2],slice_distance)
    qx,qy,qz=cp.meshgrid(qx,qy,qz)
    qr=qx**2+qy**2
    qr[qr==0]=1
    weight=(beta_wedge**2)*(qz**2)/qr
    weight=(wedge_mu*0.63661977236)*cp.arctan(weight)
    fft_object_3d=pyptyfft.fftn(obj, axes=(0,1,2))
    loss_term= cp.sum( (cp.abs(fft_object_3d)**2) * weight[:,:,:,None] )
    weight*=fft_object_3d.shape[0]*fft_object_3d.shape[1]*fft_object_3d.shape[2]
    fft_object_3d = pyptyfft.ifftn(fft_object_3d * weight[:,:,:,None], axes=(0,1,2))
    del weight
    return loss_term, fft_object_3d
    
def compute_mixed_object_variance_constraint(this_obj, weight, sigma, return_direction, smart_memory):
    """
    Regularize variance across object modes by penalizing their differences.

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
    """
    mask=cp.exp(-cp.sum(cp.array(cp.meshgrid(pyptyfft.fftfreq(this_obj.shape[1]), pyptyfft.fftfreq(this_obj.shape[0]), indexing="xy"))**2, axis=0)/sigma**2)
    if smart_memory:
        this_obj_blur=cp.copy(this_obj)
        for i in range(this_obj.shape[2]):
            for j in range(this_obj.shape[3]):
                this_obj_blur[:,:,i,j]=pyptyfft.fft2(this_obj[:,:,i,j])
                this_obj_blur[:,:,i,j]*=mask
        for i in range(this_obj.shape[2]):
            for j in range(this_obj.shape[3]):
                this_obj_blur[:,:,i,j]=pyptyfft.ifft2(this_obj_blur[:,:,i,j])
    else:
        this_obj_blur=pyptyfft.ifft2(pyptyfft.fft2(this_obj, axes=(0,1))*mask[:,:,None,None], axes=(0,1))
    n_states=this_obj.shape[-1]
    mean_obj=cp.mean(this_obj_blur, axis=-1)
    mean_obj=this_obj_blur-mean_obj[:,:,:,None] ## not the mean object anymore
    del this_obj_blur
    reg_term=weight*cp.sum(cp.abs(mean_obj)**2)/n_states ## variance
    if return_direction:
        mean_obj*=(weight/n_states) ## actually a grad with respect to the blurred object
        if smart_memory:
            for i in range(this_obj.shape[2]):
                for j in range(this_obj.shape[3]):
                    mean_obj[:,:,i,j]=pyptyfft.fft2(mean_obj[:,:,i,j])*mask
            for i in range(this_obj.shape[2]):
                for j in range(this_obj.shape[3]):
                    mean_obj[:,:,i,j]=pyptyfft.ifft2(mean_obj[:,:,i,j])
        else:
            mean_obj=pyptyfft.ifft2(pyptyfft.fft2(mean_obj, axes=(0,1))*mask[:,:,None,None], axes=(0,1)) ## I am not dumb, the above disaster should be less memory hungry than this beauty
    else:
        del mean_obj, mask
        mean_obj=None
    return reg_term, mean_obj
    


