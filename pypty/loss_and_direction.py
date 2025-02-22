import numpy as np
import sys
import h5py

try:
    import cupyx
    import cupy as cp
except:
    import numpy as cp

from pypty.fft import *
from pypty.utils import *
from pypty.multislice import *

half_master_propagator_phase_space, master_propagator_phase_space, q2, qx, qy, exclude_mask, x_real_grid_tilt, y_real_grid_tilt, shift_probe_mask_x, shift_probe_mask_y, exclude_mask_ishift, probe_runx, probe_runy, yx_real_grid_tilt, shift_probe_mask_yx, aberrations_polynomials=None, None,None, None,None, None,None, None, None, None, None, None,None,None,None,None
def loss_and_direction(this_obj, full_probe, this_pos_array, this_pos_correction, this_tilt_array, this_tilts_correction, this_distances,  measured_array,  algorithm_type, this_wavelength, this_step_probe, this_step_obj, this_step_pos_correction, this_step_tilts, masks, pixel_size_x_A, pixel_size_y_A, recon_type, Cs, defocus_array, alpha_near_field, damping_cutoff_multislice, smooth_rolloff, propmethod, this_chopped_sequence, load_one_by_one, data_multiplier, data_pad, phase_plate_in_h5, this_loss_weight, data_bin, data_shift_vector, upsample_pattern, static_background, this_step_static_background, tilt_mode, aberration_marker, probe_marker, aberrations_array, compute_batch, phase_only_obj, beam_current, this_beam_current_step, this_step_aberrations_array, default_float, default_complex, xp, is_first_epoch,
                       scan_size,fast_axis_reg_weight_positions, current_hp_reg_weight_positions,current_hp_reg_coeff_positions, current_hp_reg_weight_tilts,current_hp_reg_coeff_tilts, fast_axis_reg_weight_tilts, aperture_mask, probe_reg_weight, current_window_weight, current_window, phase_norm_weight, abs_norm_weight, atv_weight, atv_q, atv_p, mixed_variance_weight, mixed_variance_sigma, smart_memory, print_flag):
    """
    This is an internal PyPty function for iterative Ptychography. It does both forward and backward propagations. Inputs are the parameters of the experiment and outputs are loss, SSE and the gradients of the loss with respect to refinable arrays. 
    """
    
    global pool, pinned_pool, half_master_propagator_phase_space, master_propagator_phase_space, q2, qx, qy, exclude_mask,exclude_mask_ishift, x_real_grid_tilt, y_real_grid_tilt,yx_real_grid_tilt, shift_probe_mask_x, shift_probe_mask_y,shift_probe_mask_yx, probe_runx,probe_runy, aberrations_polynomials
    if is_first_epoch:
        half_master_propagator_phase_space, master_propagator_phase_space, q2, qx, qy, exclude_mask,exclude_mask_ishift, x_real_grid_tilt, y_real_grid_tilt,yx_real_grid_tilt, shift_probe_mask_x, shift_probe_mask_y,shift_probe_mask_yx, probe_runx,probe_runy, aberrations_polynomials= None, None,None,None,None,None,None,None,None,None,None,None,None,None,None,None
   # checking the situation and prepare a bunch of flags
   # start_gpu = cp.cuda.Event()
   # end_gpu = cp.cuda.Event()
   # start_gpu.record()
    if 'lsq_compressed'==algorithm_type:
        masks_len=masks.shape[0] #int
    pattern_number, loss, sse=len(this_chopped_sequence),0,0
    fourier_probe_grad=None
    ###prepare loss, sse and arrays for grads of object, probe, positions and tilts
    object_grad, probe_grad, pos_grad, tilts_grad = cp.zeros_like(this_obj), cp.zeros_like(full_probe), cp.zeros_like(this_pos_correction), cp.zeros_like(this_tilt_array) ### this runs very fast
    static_background_grad= cp.zeros_like(static_background) if type(static_background)==cp.ndarray else 0
    beam_current_grad=cp.zeros_like(beam_current) if not(beam_current is None) else 0
    aberrations_array_grad=cp.zeros_like(aberrations_array) if not(aberrations_array is None) else 0
    
    this_ps, is_single_tilt, is_single_pos, is_single_dist, num_slices, n_obj_modes, n_probe_modes, is_single_defocus, multiple_scenarios, fluctuating_current_flag = full_probe.shape[0], (this_tilt_array.shape[0]==1), (this_pos_array.shape[0]==1), (this_distances.shape[0]==1), this_obj.shape[2], this_obj.shape[3], full_probe.shape[2], False, not(probe_marker is None) and len(full_probe.shape)==4, not(beam_current is None)
    if xp!=np and load_one_by_one:
        stream1=cp.cuda.Stream(non_blocking=True)
        ms0, ms1, ms2=measured_array.shape
        pinned_measured=cupyx.empty_pinned((compute_batch, ms1,ms2), dtype=measured_array.dtype)
        if pattern_number%compute_batch!=0: pinned_measured_remain=cupyx.empty_pinned((pattern_number%compute_batch, ms1,ms2), dtype=measured_array.dtype);
    is_fully_coherent=n_probe_modes==1 and n_obj_modes==1
    if probe_runx is None:
        probe_runx,probe_runy=cp.meshgrid(cp.arange(this_ps, dtype=int),cp.arange(this_ps, dtype=int), indexing="xy")
        probe_runx,probe_runy=probe_runx[None,:,:],probe_runy[None,:,:]
    if exclude_mask is None:
        q2, qx, qy, exclude_mask, exclude_mask_ishift = create_spatial_frequencies(pixel_size_x_A, pixel_size_y_A, this_ps, damping_cutoff_multislice, smooth_rolloff, default_float)     # create some arrays with spatial frequencies. Many of them are actually idential (or almost idential, so i will later clean this mess). In any case, the code is currently optimized to create them only once (when configured properly)
        qx,qy,q2, exclude_mask, exclude_mask_ishift=qx[None,:,:],qy[None,:,:],q2[None,:,:], exclude_mask[None,:,:], exclude_mask_ishift[None,:,:]
    if tilt_mode and (x_real_grid_tilt is None): ## if anything but zero
        x_real_grid_tilt, y_real_grid_tilt=cp.meshgrid(fftshift(fftfreq(full_probe.shape[1])), fftshift(fftfreq(full_probe.shape[0])), indexing="xy")
        x_real_grid_tilt=((x_real_grid_tilt*6.2831855j*full_probe.shape[1]*pixel_size_x_A/this_wavelength)[None,:,:,None]).astype(default_complex)
        y_real_grid_tilt=((y_real_grid_tilt*6.2831855j*full_probe.shape[0]*pixel_size_y_A/this_wavelength)[None,:,:,None]).astype(default_complex)
        yx_real_grid_tilt=cp.stack((y_real_grid_tilt, x_real_grid_tilt), axis=1) # (1, 2, y, x, 1)
    if shift_probe_mask_yx is None:
        shift_probe_mask_x, shift_probe_mask_y=cp.meshgrid(fftfreq(full_probe.shape[1]), fftfreq(full_probe.shape[0]), indexing="xy")
        shift_probe_mask_x, shift_probe_mask_y=(-6.2831855j*shift_probe_mask_x[None,:,:]).astype(default_complex), (-6.2831855j*shift_probe_mask_y[None,:,:]).astype(default_complex)
        shift_probe_mask_x=shift_probe_mask_x*exclude_mask_ishift
        shift_probe_mask_y=shift_probe_mask_y*exclude_mask_ishift
        shift_probe_mask_yx=cp.stack((shift_probe_mask_y, shift_probe_mask_x), axis=1) #(1, 2, y, x)
    is_multislice = num_slices>1 or propmethod=="better_multislice" or propmethod=="yoshida"
    is_single_defocus = recon_type=="far_field" or defocus_array.shape[0]==1
    helper_flag_1= this_step_probe or this_beam_current_step
    helper_flag_2= helper_flag_1 or this_step_aberrations_array
    helper_flag_3= helper_flag_2 or (this_step_tilts and (tilt_mode==2 or tilt_mode==4))
    helper_flag_4= helper_flag_3 or this_step_pos_correction
    if not(aberration_marker is None):
        num_abs=aberrations_array.shape[1]
        if aberrations_polynomials is None:
            aberrations_polynomials=(-1j*get_ctf_matrix(qx[0]*this_wavelength, qy[0]*this_wavelength, num_abs, this_wavelength)).astype(default_complex)
        local_aberrations_phase_plates=cp.exp(cp.sum(aberrations_polynomials[None,:,:,:]*aberrations_array[:,:,None,None], 1))
        morph_incomming_beam=True
    else:
        morph_incomming_beam=False
    individual_propagator_flag = not(is_single_tilt) and (tilt_mode==0 or tilt_mode==3 or tilt_mode==4)
    if not(phase_plate_in_h5 is None):
        phase_plate_active = True
        phase_plate_data = h5py.File(phase_plate_in_h5, "r")["configs"]
        N_phase_plates = phase_plate_data.shape[0]
        is_single_phase_plate= (N_phase_plates==1)
        matrix_phase_plates= (len(phase_plate_data.shape)==3)
    else:
        phase_plate_active=False
    if not(individual_propagator_flag) and is_single_dist and is_multislice and (master_propagator_phase_space is None):
        this_tilt=(this_tilt_array+this_tilts_correction)[0,:]
        this_tan_y_inside=this_tilt[2]
        this_tan_x_inside=this_tilt[3]
        if propmethod=="better_multislice":
            half_master_propagator_phase_space=cp.exp(-3.141592654j*0.5*this_distances[0]*(this_wavelength*q2+2*(qx*this_tan_x_inside+qy*this_tan_y_inside)))*exclude_mask_ishift
            half_master_propagator_phase_space=cp.expand_dims(half_master_propagator_phase_space,(-1,-2))
            master_propagator_phase_space=half_master_propagator_phase_space**2
        if propmethod=="yoshida":
            sigma_yoshida=(2+2**(-1/3)+2**(1/3))/3
            half_master_propagator_phase_space=cp.expand_dims(cp.exp(-3.141592654j*sigma_yoshida*this_distances[0]*(this_wavelength*q2+2*(qx*this_tan_x_inside+qy*this_tan_y_inside)))*exclude_mask_ishift,(-1,-2))
            master_propagator_phase_space=cp.expand_dims(cp.exp(-3.141592654j*(1-2*sigma_yoshida)*this_distances[0]*(this_wavelength*q2+2*(qx*this_tan_x_inside+qy*this_tan_y_inside)))*exclude_mask_ishift,(-1,-2))
        if propmethod=="multislice":
            master_propagator_phase_space=cp.exp(-3.141592654j*this_distances[0]*(this_wavelength*q2+2*(qx*this_tan_x_inside+qy*this_tan_y_inside)))*exclude_mask_ishift
            master_propagator_phase_space=cp.expand_dims(master_propagator_phase_space,(-1,-2)) ### probe modes
            half_master_propagator_phase_space=None
    else:
        master_propagator_phase_space=None
        half_master_propagator_phase_space=None
    this_tilt_array=(this_tilt_array+this_tilts_correction)[:,:,None, None]
    this_pos_array=this_pos_array[:,:,None]
    if type(static_background)==cp.ndarray:
        static_square=(static_background[None,:,:]**2)*exclude_mask
        static_background_is_there=cp.max(static_square)>1e-20
    else:
        static_background_is_there=False
    probe_shift_flag=cp.sum(this_pos_correction**2)!=0.0 or this_step_pos_correction
    tcsl=len(this_chopped_sequence)
    sh0=compute_batch if compute_batch<=tcsl else tcsl
    if propmethod=="multislice":
        waves_multislice = cp.empty((sh0, this_ps,this_ps,num_slices,n_probe_modes,n_obj_modes, 2), dtype=default_complex)
    else:
        if propmethod=="yoshida":
            waves_multislice = cp.empty((sh0, this_ps,this_ps,num_slices,n_probe_modes,n_obj_modes, 7), dtype=default_complex)
        else:
            waves_multislice = cp.empty((sh0, this_ps,this_ps,num_slices,n_probe_modes,n_obj_modes, 10), dtype=default_complex)
    this_exit_wave           = cp.empty((sh0, this_ps,this_ps,         n_probe_modes,n_obj_modes   ), dtype=default_complex)
    if multiple_scenarios:
        full_probe=cp.moveaxis(full_probe,3,0)
        if this_step_probe:
            probe_grad=cp.moveaxis(probe_grad, 3,0)
    else:
        full_probe=full_probe[None, :,:,:]
        
    for i in range(0, pattern_number, compute_batch):     ### start of the pattern loop
        next_i=i+compute_batch
        tcs=this_chopped_sequence[i:next_i]
        lltcs=len(tcs)
        if compute_batch>1 and i>0 and next_i>pattern_number:
            waves_multislice=waves_multislice[:lltcs]
            waves_multislice=waves_multislice[:lltcs]
            this_exit_wave=this_exit_wave[:lltcs]
        if xp!=np and load_one_by_one:
            with stream1:
                measured_chop=measured_array[tcs]
                if next_i>pattern_number:
                    pinned_measured_remain[:]=measured_chop
                    measured=cp.array(pinned_measured_remain)
                else:
                    pinned_measured[:]=measured_chop
                    measured=cp.array(pinned_measured)
                measured, *_ = preprocess_dataset(measured, False, algorithm_type, recon_type, data_shift_vector, data_bin, data_pad, upsample_pattern, data_multiplier, xp, True) ### preprocess
        else:
            measured=measured_array[tcs]
            measured, *_ = preprocess_dataset(measured, False, algorithm_type, recon_type, data_shift_vector, data_bin, data_pad, upsample_pattern, data_multiplier, xp, True)
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
        this_obj_chopped=fft2(this_obj_chopped, axes=(1,2), overwrite_x=True)
        this_obj_chopped*=exclude_mask_ishift[:,:,:,None,None] ## set the cutoff
        this_obj_chopped=ifft2(this_obj_chopped, axes=(1,2), overwrite_x=True)
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
            this_probe_fourier=fft2(this_probe, axes=(1,2))*this_phase_plate[:,:,:,None]
            this_probe=ifft2(this_probe_fourier,  axes=(1,2))
        if morph_incomming_beam:
            local_aberrations_phase_plate=local_aberrations_phase_plates[aberration_marker[tcs]]
            if this_probe_fourier is None: this_probe_fourier=fft2(this_probe, axes=(1,2));
            this_fourier_probe_before_local_aberrations=this_probe_fourier*1
            this_probe=ifft2(this_fourier_probe_before_local_aberrations*local_aberrations_phase_plate[:,:,:,None], axes=(1,2))
        if tilt_mode==2 or tilt_mode==4: #before
            tilting_mask_real_space_before=cp.exp(x_real_grid_tilt*this_tan_x_before+y_real_grid_tilt*this_tan_y_before)
            this_probe_before_tilt=this_probe[:]
            this_probe=this_probe_before_tilt*tilting_mask_real_space_before
        else:
            tilting_mask_real_space_before=None
        if probe_shift_flag:
            if this_probe_fourier is None: this_probe_fourier=fft2(this_probe, axes=(1,2));
            shift_probe_mask=cp.exp(shift_probe_mask_x*this_pos_corr[:,1:2,None]+shift_probe_mask_y*this_pos_corr[:,0:1,None])*exclude_mask_ishift
            this_probe=ifft2(this_probe_fourier*shift_probe_mask[:,:,:,None], axes=(1,2))
        if this_probe.shape[0]==1 and lltcs>1:
            this_probe=xp.repeat(this_probe, lltcs, axis=0)
        ## FORWARD PROPAGATION
        if propmethod=="multislice":
            if is_multislice and is_single_dist and not(individual_propagator_flag):
                master_propagator_phase_space=cp.exp(-3.141592654j*this_distances[0]*(this_wavelength*q2+2*(qx*this_tan_x_inside+qy*this_tan_y_inside)))*exclude_mask_ishift
                master_propagator_phase_space=cp.expand_dims(master_propagator_phase_space,(-1,-2))
            else:
                master_propagator_phase_space,half_master_propagator_phase_space=None,None
            waves_multislice, this_exit_wave=multislice(this_probe, this_obj_chopped, num_slices, n_obj_modes, n_probe_modes, this_distances, this_wavelength, q2, qx, qy, exclude_mask, is_single_dist, this_tan_x_inside,this_tan_y_inside, damping_cutoff_multislice, smooth_rolloff, master_propagator_phase_space,  None, exclude_mask_ishift, waves_multislice, this_exit_wave, default_float, default_complex)
        else:
            if propmethod=="better_multislice":
                if is_single_dist and not(individual_propagator_flag):
                    half_master_propagator_phase_space=cp.exp(-3.141592654j*0.5*this_distances[0]*(this_wavelength*q2+2*(qx*this_tan_x_inside+qy*this_tan_y_inside)))*exclude_mask_ishift
                    half_master_propagator_phase_space=cp.expand_dims(half_master_propagator_phase_space,(-1,-2))
                    master_propagator_phase_space=half_master_propagator_phase_space**2
                waves_multislice, this_exit_wave=better_multislice(this_probe, this_obj_chopped, num_slices, n_obj_modes, n_probe_modes, this_distances, this_wavelength, q2, qx, qy, exclude_mask, is_single_dist, this_tan_x_inside,this_tan_y_inside, damping_cutoff_multislice, smooth_rolloff, master_propagator_phase_space,  half_master_propagator_phase_space, exclude_mask_ishift, waves_multislice,this_exit_wave, default_float, default_complex)
            elif propmethod=="yoshida":
                if is_single_dist and not(individual_propagator_flag):
                    sigma_yoshida=(2+2**(-1/3)+2**(1/3))/3
                    half_master_propagator_phase_space=cp.expand_dims(cp.exp(-3.141592654j*(sigma_yoshida)*this_distances[0]*(this_wavelength*q2+2*(qx*this_tan_x_inside+qy*this_tan_y_inside)))*exclude_mask_ishift,(-1,-2))
                    master_propagator_phase_space=cp.expand_dims(cp.exp(-3.141592654j*(1-2*sigma_yoshida)*this_distances[0]*(this_wavelength*q2+2*(qx*this_tan_x_inside+qy*this_tan_y_inside)))*exclude_mask_ishift,(-1,-2))
                waves_multislice, this_exit_wave=yoshida_multislice(this_probe, this_obj_chopped, num_slices, n_obj_modes, n_probe_modes, this_distances, this_wavelength, q2, qx, qy, exclude_mask, is_single_dist, this_tan_x_inside,this_tan_y_inside, damping_cutoff_multislice, smooth_rolloff, master_propagator_phase_space,  half_master_propagator_phase_space, exclude_mask_ishift, waves_multislice,this_exit_wave, default_float, default_complex)
                
        if tilt_mode==1 or tilt_mode>=3: ## after if tilt mode is 1, 3 or 4
            tilting_mask_real_space_after=cp.exp(x_real_grid_tilt*this_tan_x_after+y_real_grid_tilt*this_tan_y_after)[:,:,:,:, None]
            if propmethod=="multislice" and num_slices==1:
                this_exit_wave=fft2(this_exit_wave, axes=(1,2), overwrite_x=True)
                this_exit_wave*=exclude_mask_ishift[:,:,:,None,None]
                this_exit_wave=ifft2(this_exit_wave, axes=(1,2), overwrite_x=True)
            this_exit_wave_before_tilt=this_exit_wave[:]
            this_exit_wave = this_exit_wave_before_tilt*tilting_mask_real_space_after
        else:
            tilting_mask_real_space_after=None
        ## PROPAGATION FROM THE SPECIMEN PLANE TO THE DETECTOR PLANE
        if recon_type=="far_field":
            this_wave=shift_fft2(this_exit_wave, axes=(1,2))
            this_wave*=exclude_mask[:,:,:,None,None]
        else:
            CTF=(cp.exp(-3.141592654j*(((this_wavelength*q2+2*(qx*this_tan_x_inside+qy*this_tan_y_inside))*defocus_near_field)+(0.5*Cs*this_wavelength**3*q2**2)))*exclude_mask_ishift)[:,:,:,None,None]
            fourier_exit_wave=fft2(this_exit_wave, axes=(1,2))
            this_wave=ifft2(fourier_exit_wave*CTF, axes=(1,2))
        this_pattern=cp.abs(this_wave)**2
        if is_fully_coherent:
            this_pattern=this_pattern[:,:,:,0,0]
        else:
            this_pattern=cp.mean(this_pattern, (-1,-2))
        if recon_type=="near_field" and alpha_near_field>0: ## the correction to a coherent case
            E_near_field_not_coherent_correction=(cp.exp(-q2*(3.14152654*alpha_near_field*defocus_near_field)**2))[ None,:,:]
            this_pattern=cp.real(ifft2(fft2(this_pattern, axes=(1,2))*E_near_field_not_coherent_correction, axes=(1,2)))
        if upsample_pattern!=1:
            this_pattern=downsample_something_3d(this_pattern, upsample_pattern, xp)
        if static_background_is_there:
            this_pattern+=static_square
        try:
            stream1.synchronize()
        except:
            pass
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
            if algorithm_type=='custom_strange_loss':
                mnonzero=(measured>0)
                loss_full_pix=(this_pattern -2*(measured*this_pattern)**0.5 + measured)*mnonzero
                loss+=cp.sum(loss_full_pix)
                sse+=cp.sum(cp.abs(measured-this_pattern)**2)
                this_pattern[this_pattern==0]=1e-10
                dLoss_dint=(1-(measured/this_pattern)**0.5)*mnonzero
            if algorithm_type=='ml':
                dLoss_dint=this_pattern-measured
                sse+=cp.sum(dLoss_dint**2)
                this_pattern[this_pattern==0]=1
                loss_full_pix=this_pattern - measured*cp.log(this_pattern)
                loss+=cp.sum(loss_full_pix)
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
                this_coeffs=cp.empty((this_pattern.shape[0], masks_len), dtype=default_float)
                flat_pattern=this_pattern.reshape(this_pattern.shape[0], this_pattern.shape[1]*this_pattern.shape[2])
                for ind_masks in range(masks_len):
                    this_flat_mask=masks[ind_masks].flatten()
                    sort=this_flat_mask!=0
                    this_pixels, this_mask_pixels = flat_pattern[:,sort], this_flat_mask[sort]
                    this_coeffs[:,ind_masks] = cp.sum(this_pixels*this_mask_pixels[None,:], axis=1, dtype=cp.float64).astype(default_float)
                this_differences=this_coeffs - measured
                tterm=cp.sum(this_differences**2, axis=1)
                loss+=tterm
                sse+=tterm
                dLoss_dint=cp.zeros(this_pattern.shape, dtype=default_float)
                for ind_loss_grad_comp in range(masks_len):
                    dLoss_dint+=(masks[None, ind_loss_grad_comp]*(this_differences[:,ind_loss_grad_comp]))
                dLoss_dint*=2
        if not(is_fully_coherent):
            dLoss_dint/=(n_probe_modes*n_obj_modes)
        if this_step_static_background and static_background_is_there:
            static_background_grad+=cp.sum(dLoss_dint,0)*2*static_background*this_loss_weight
        if upsample_pattern!=1:
            dLoss_dint=upsample_something_3d(dLoss_dint, upsample_pattern, False, xp)
        if recon_type=="far_field":
            dLoss_dWave  =  dLoss_dint[:,:,:,None,None] * this_wave * (this_loss_weight * dLoss_dint.shape[1]* dLoss_dint.shape[2])
            dLoss_dP_out = ifft2_ishift(dLoss_dWave, axes=(1,2))
        else:
            if alpha_near_field>0:
                dLoss_dWave=this_wave*cp.expand_dims(cp.real(ifft2(fft2(dLoss_dint, axes=(1,2))*E_near_field_not_coherent_correction, axes=(1,2))),(-1,-2))
            else:
                dLoss_dWave=this_wave*cp.expand_dims(dLoss_dint, (-1,-2))
            fouirer_wave_grad=this_loss_weight * fft2(dLoss_dWave, axes=(1,2))*cp.conjugate(CTF) #(n_m, y,x, mp, mo)
            dLoss_dP_out=ifft2(fouirer_wave_grad, axes=(1,2))
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
                    dL_dtilt=cp.sum(cp.real(grad_tilting_kernel[:,None,:,:,0,0]*yx_real_grid_tilt[:,:,:,:,0]), (2,3), dtype=cp.float64).astype(default_float)
                else:
                    dL_dtilt=cp.sum(cp.real(grad_tilting_kernel[:,None,:,:,:,:]*yx_real_grid_tilt[:,:,:,:,:,None]),(2,3,4,5), dtype=cp.float64).astype(default_float)
                if is_single_tilt:
                    dL_dtilt=cp.sum(dL_dtilt, 0)
                    tilts_grad[0,4:]+=dL_dtilt
                else:
                    tilts_grad[tiltind,4:]=dL_dtilt
            if propmethod=="multislice" and num_slices==1:
                dLoss_dP_out=fft2(dLoss_dP_out, axes=(1,2), overwrite_x=True)
                dLoss_dP_out*=exclude_mask_ishift[:,:,:,None,None]
                dLoss_dP_out=ifft2(dLoss_dP_out, axes=(1,2), overwrite_x=True)
        if propmethod=="multislice":
            object_grad, interm_probe_grad, tilts_grad = multislice_grads(dLoss_dP_out, waves_multislice, this_obj_chopped, object_grad, tilts_grad, is_single_dist,this_distances, exclude_mask, this_wavelength, q2, qx, this_tan_x_inside, qy, this_tan_y_inside, num_slices, n_obj_modes,tiltind, master_propagator_phase_space, this_step_tilts, damping_cutoff_multislice, smooth_rolloff, tilt_mode,  compute_batch, exclude_mask_ishift, this_step_probe, this_step_obj, this_step_pos_correction, masked_pixels_y, masked_pixels_x, default_float, default_complex, helper_flag_4)
        else:
            if propmethod=="better_multislice":
                object_grad,  interm_probe_grad, tilts_grad = better_multislice_grads(dLoss_dP_out, waves_multislice, this_obj_chopped, object_grad, tilts_grad, is_single_dist, this_distances, exclude_mask, this_wavelength, q2, qx, this_tan_x_inside, qy, this_tan_y_inside, num_slices, n_probe_modes, n_obj_modes,tiltind, this_step_tilts,  master_propagator_phase_space, half_master_propagator_phase_space, damping_cutoff_multislice, smooth_rolloff, tilt_mode,  compute_batch, exclude_mask_ishift, masked_pixels_y, masked_pixels_x, default_float, default_complex)
            if propmethod=="yoshida":
                object_grad, interm_probe_grad, tilts_grad = yoshida_multislice_grads(dLoss_dP_out, waves_multislice, this_obj_chopped, object_grad, tilts_grad, is_single_dist, this_distances, exclude_mask, this_wavelength, q2, qx, this_tan_x_inside, qy, this_tan_y_inside, num_slices, n_probe_modes, n_obj_modes,tiltind, this_step_tilts,  master_propagator_phase_space, half_master_propagator_phase_space, damping_cutoff_multislice, smooth_rolloff, tilt_mode,  compute_batch, exclude_mask_ishift, masked_pixels_y, masked_pixels_x, default_float, default_complex)
        if helper_flag_4:
            fourier_probe_grad=None
            if probe_shift_flag:
                fourier_probe_grad=fft2(interm_probe_grad, axes=(1,2))
                fourier_probe_grad=fourier_probe_grad*cp.conjugate(shift_probe_mask)[:,:,:,None]
                interm_probe_grad=ifft2(fourier_probe_grad, axes=(1,2))
                if this_step_pos_correction:
                    shift_mask_grad=cp.conjugate(this_probe_fourier)*fourier_probe_grad
                    sh = -2/(this_probe_fourier.shape[1]*this_probe_fourier.shape[2]) ## -1 for conj, 2 for real-grad shape for ifft
                    if n_probe_modes==1:
                        sh_grad=sh*cp.sum(cp.real(shift_probe_mask_yx[:,:,:,:]*shift_mask_grad[:,None,:,:,0]), (2,3), dtype=cp.float64).astype(default_float)
                    else:
                        sh_grad= sh*cp.sum(cp.real(shift_probe_mask_yx[:,:,:,:,None]*shift_mask_grad[:,None,:,:,:]), (2,3,4), dtype=cp.float64).astype(default_float)
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
                            dL_dtilt=cp.sum(cp.real(grad_tilting_kernel[:,None,:,:,0]*yx_real_grid_tilt[:,:,:,0]),(2,3),  dtype=cp.float64).astype(default_float)
                        else:
                            dL_dtilt=cp.sum(cp.real(grad_tilting_kernel[:,None,:,:,:]*yx_real_grid_tilt),(2,3,4),  dtype=cp.float64).astype(default_float)
                        if is_single_tilt:
                            dL_dtilt=cp.sum(dL_dtilt, 0)
                            tilts_grad[0,:2]+=dL_dtilt
                        else:
                            tilts_grad[tiltind,:2]=dL_dtilt
                if helper_flag_2:
                    if morph_incomming_beam:
                        if fourier_probe_grad is None: fourier_probe_grad=fft2(interm_probe_grad, axes=(1,2));
                        fourier_probe_grad*=cp.conjugate(local_aberrations_phase_plate[:,:,:,None])
                        interm_probe_grad=ifft2(fourier_probe_grad, axes=(1,2))
                        if this_step_aberrations_array:
                            sh = 2/(fourier_probe_grad.shape[1]*fourier_probe_grad.shape[2])
                            defgr=sh*cp.sum(cp.real((fourier_probe_grad*cp.conjugate(this_fourier_probe_before_local_aberrations))[:,None, :,:,:]*cp.conjugate(aberrations_polynomials[None,:,:,:,None])), axis=(2,3,4), dtype=cp.float64).astype(default_float)
                            scatteradd_abers(aberrations_array_grad, aberration_marker[tcs], defgr)
                            #for dumbindex, t in enumerate(tcs):
                             #   aberrations_array_grad[aberration_marker[t],:]+=defgr[dumbindex,:]
                    if helper_flag_1:
                        if phase_plate_active:
                            if fourier_probe_grad is None: fourier_probe_grad=fft2(interm_probe_grad, axes=(1,2));
                            cp.conjugate(this_phase_plate, out=this_phase_plate)
                            interm_probe_grad=ifft2(fourier_probe_grad*this_phase_plate[:,:,:,None], (1,2))
                        if fluctuating_current_flag:
                            if this_beam_current_step:
                                if n_probe_modes==1:
                                    cp.conjugate(this_probe_before_fluctuations, out=this_probe_before_fluctuations)
                                    
                                    beam_current_grad[tcs]=2*cp.sign(thisbc)*cp.sum(cp.real(interm_probe_grad[:,:,:,0]*this_probe_before_fluctuations[:,:,:,0]), (1,2), dtype=cp.float64).astype(default_float)
                                else:
                                    beam_current_grad[tcs]=2*cp.sign(thisbc)*cp.sum(cp.real(interm_probe_grad*cp.conjugate(this_probe_before_fluctuations)), (1,2,3))
                            if this_step_probe:
                                interm_probe_grad=interm_probe_grad*beam_current_values
                        if this_step_probe:
                            if multiple_scenarios:
                                scatteradd_probe(probe_grad, probe_marker[tcs], interm_probe_grad)
                            else:
                                probe_grad+=cp.sum(interm_probe_grad,0)
    loss*=this_loss_weight
   # end_gpu.record()
   # end_gpu.synchronize()
   # t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
   # print("\n", t_gpu)
    constraint_contributions=[]
    updated_fast_axis_reg_weight_positions, updated_current_hp_reg_weight_positions, updated_current_hp_reg_weight_tilts, updated_fast_axis_reg_weight_tilts, updated_phase_norm_weight, updated_abs_norm_weight, updated_probe_reg_weight, updated_current_window_weight, updated_atv_weight, updated_mixed_variance_weight= None, None, None, None, None, None, None, None, None, None
    loss_print_copy=1*loss;
    this_pos_array=this_pos_array[:,:,0]
    this_tilt_array=this_tilt_array[:,:,0,0]
    if this_step_probe and multiple_scenarios: probe_grad=cp.moveaxis(probe_grad, 0,3);
    #######
    if this_step_pos_correction and fast_axis_reg_weight_positions!=0:
        something=this_pos_array+this_pos_correction
        if type(fast_axis_reg_weight_positions)==str:
            fraction=float(fast_axis_reg_weight_positions)
            fast_axis_reg_weight_positions=1
            recompute=True
        else:
            recompute=False
        ind_loss, reg_grad=compute_fast_axis_constraint_on_grid(something, scan_size, fast_axis_reg_weight_positions)
        if recompute:
            updated_fast_axis_reg_weight_positions=loss_print_copy*fraction/ind_loss
            ind_loss*=updated_fast_axis_reg_weight_positions
            reg_grad*=updated_fast_axis_reg_weight_positions
        pos_grad+=reg_grad
        loss+=ind_loss
        constraint_contributions.append(ind_loss)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Positions fast axis constaint is %.2f %% of the main loss"%(fast_axis_reg_weight_positions, ind_loss*100/loss_print_copy));
    else:
        constraint_contributions.append(0)
    #######
    if this_step_pos_correction and current_hp_reg_weight_positions!=0:
        something=this_pos_array+this_pos_correction
        if type(current_hp_reg_weight_positions)==str:
            fraction=float(current_hp_reg_weight_positions)
            current_hp_reg_weight_positions=1
            recompute=True
        else:
            recompute=False
        ind_loss, reg_grad = compute_hp_constraint_on_grid(something, scan_size, current_hp_reg_weight_positions, current_hp_reg_coeff_positions)
        if recompute:
            updated_current_hp_reg_weight_positions=loss_print_copy*fraction/ind_loss
            ind_loss*=updated_current_hp_reg_weight_positions
            reg_grad*=updated_current_hp_reg_weight_positions
        pos_grad+=reg_grad;
        loss+=ind_loss
        constraint_contributions.append(ind_loss)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Positions slow axis constaint is %.2f %% of the main loss"%(current_hp_reg_weight_positions, ind_loss*100/loss_print_copy));
    else:
        constraint_contributions.append(0)
    ########
    if this_step_tilts and current_hp_reg_weight_tilts!=0:
        something=this_tilt_array
        if type(current_hp_reg_weight_tilts)==str:
            fraction=float(current_hp_reg_weight_tilts)
            current_hp_reg_weight_tilts=1
            recompute=True
        else:
            recompute=False
        ind_loss, reg_grad=compute_hp_constraint_on_grid(something, scan_size, current_hp_reg_weight_tilts, current_hp_reg_coeff_tilts)
        if recompute:
            updated_current_hp_reg_weight_tilts=loss_print_copy*fraction/ind_loss
            ind_loss*=updated_current_hp_reg_weight_tilts
            reg_grad*=updated_current_hp_reg_weight_tilts
        tilts_grad+=reg_grad;
        loss+=ind_loss
        constraint_contributions.append(ind_loss)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Tilts slow axis constaint is %.2f %% of the main loss"%(current_hp_reg_weight_tilts, ind_loss*100/loss_print_copy));
    else:
        constraint_contributions.append(0)
    #######
    if this_step_tilts and fast_axis_reg_weight_tilts!=0:
        something=this_tilt_array
        if type(fast_axis_reg_weight_tilts)==str:
            fraction=float(fast_axis_reg_weight_tilts)
            fast_axis_reg_weight_tilts=1
            recompute=True
        else:
            recompute=False
        ind_loss, reg_grad=compute_fast_axis_constraint_on_grid(something, scan_size, fast_axis_reg_weight_tilts)
        if recompute:
            updated_fast_axis_reg_weight_tilts=loss_print_copy*fraction/ind_loss
            ind_loss*=updated_fast_axis_reg_weight_tilts
            reg_grad*=updated_fast_axis_reg_weight_tilts
        tilts_grad+=reg_grad
        loss+=ind_loss
        constraint_contributions.append(ind_loss)
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Tilts fast axis constaint is %.2f %% of the main loss"%(fast_axis_reg_weight_tilts, ind_loss*100/loss_print_copy));
    else:
        constraint_contributions.append(0)
    #######
    if phase_norm_weight!=0: # l_1 norm of the potential
        grad_mask=generate_mask_for_grad_from_pos(this_obj.shape[1], this_obj.shape[0], this_pos_array, full_probe.shape[1],full_probe.shape[0], 0)
        if type(phase_norm_weight)==str:
            fraction=float(phase_norm_weight)
            phase_norm_weight=1
            recompute=True
        else:
            recompute=False
        l1_reg_term, l1_object_grad=compute_full_l1_constraint(this_obj, 0, phase_norm_weight, grad_mask, True, smart_memory)
        if recompute:
            updated_phase_norm_weight=loss_print_copy*fraction/l1_object_grad
            l1_reg_term*=updated_phase_norm_weight
            l1_object_grad*=updated_phase_norm_weight
        loss+=l1_reg_term
        object_grad+=l1_object_grad
        constraint_contributions.append(l1_reg_term)
        del grad_mask,l1_reg_term,l1_object_grad # forget about it
        if print_flag==4:
            sys.stdout.write("\nWith abs weight of %.3e and phase weight of %.3e, l1 constaint is %.2f %% of the main loss"%(abs_norm_weight, phase_norm_weight, l1_reg_term*100/loss_print_copy));
    else:
        constraint_contributions.append(0)
    ######

    if probe_reg_weight!=0 and this_step_probe:
        if type(probe_reg_weight)==str:
            fraction=float(probe_reg_weight)
            probe_reg_weight=1
            recompute=True
        else:
            recompute=False
        probe_reg_term, reg_probe_grad = compute_probe_constraint(full_probe, aperture_mask, probe_reg_weight, True)
        if recompute:
            updated_probe_reg_weight=loss_print_copy*fraction/probe_reg_term
            probe_reg_term*=updated_probe_reg_weight
            reg_probe_grad*=updated_probe_reg_weight
        loss+=probe_reg_term
        probe_grad+=reg_probe_grad
        constraint_contributions.append(probe_reg_term)
        del reg_probe_grad, probe_reg_term
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Probe recprocal-space constaint is %.2f %% of the main loss"%(probe_reg_weight, probe_reg_term*100/loss_print_copy));
    else:
        constraint_contributions.append(0)
    ########
    if this_step_probe and current_window_weight!=0:
        if type(current_window_weight)==str:
            fraction=float(current_window_weight)
            current_window_weight=1
            recompute=True
        else:
            recompute=False
        probe_reg_term, reg_probe_grad = compute_window_constraint(full_probe, current_window, current_window_weight)
        if recompute:
            updated_current_window_weight=loss_print_copy*fraction/reg_probe_grad
            probe_reg_term*=updated_current_window_weight
            reg_probe_grad*=updated_current_window_weight
        loss+=probe_reg_term
        probe_grad+=reg_probe_grad
        constraint_contributions.append(probe_reg_term)
        del reg_probe_grad, probe_reg_term #forget about it
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Probe real-space constaint is %.2f %% of the main loss"%(current_window_weight, probe_reg_term*100/loss_print_copy));
    else:
        constraint_contributions.append(0)
    ####
    if atv_weight!=0:
        if type(atv_weight)==str:
            fraction=float(atv_weight)
            atv_weight=1
            recompute=True
        else:
            recompute=False
        atv_reg_term, atv_object_grad = compute_atv_constraint(this_obj, atv_weight, atv_q, atv_p, pixel_size_x_A, pixel_size_y_A, None, True, smart_memory)
        if recompute:
            updated_atv_weight=loss_print_copy*fraction/atv_reg_term
            atv_reg_term*=updated_atv_weight
            atv_object_grad*=updated_atv_weight
        loss+=atv_reg_term
        object_grad+=atv_object_grad
        constraint_contributions.append(atv_reg_term)
        del atv_object_grad, atv_reg_term
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, ATV constaint is %.2f %% of the main loss"%(atv_weight, atv_reg_term*100/loss_print_copy));
    else:
        constraint_contributions.append(0)
    #######
    if mixed_variance_weight!=0 and this_obj.shape[-1]>1:
        if type(mixed_variance_weight)==str:
            fraction=float(mixed_variance_weight)
            mixed_variance_weight=1
            recompute=True
        else:
            recompute=False
        mixed_variance_reg_term, mixed_variance_grad=compute_mixed_object_variance_constraint(this_obj, mixed_variance_weight, mixed_variance_sigma, True, smart_memory)
        if recompute:
            updated_mixed_variance_weight=loss_print_copy*fraction/mixed_variance_reg_term
            mixed_variance_grad*=updated_mixed_variance_weight
            mixed_variance_reg_term*=updated_mixed_variance_weight
        loss+=mixed_variance_reg_term
        object_grad+=mixed_variance_grad
        constraint_contributions.append(mixed_variance_reg_term)
        del mixed_variance_reg_term, mixed_variance_grad # forget about it
        if print_flag==4:
            sys.stdout.write("\nWith weight %.3e, Mixed variance constaint is %.2f %% of the main loss"%(mixed_variance_weight, mixed_variance_reg_term*100/loss_print_copy));
    else:
        constraint_contributions.append(0)
    ######
    if print_flag==4:
        sys.stdout.flush()
    if loss!=loss:
        raise ValueError('A very specific bad thing. Loss is Nan.')
    if this_step_probe:
        probe_grad=fft2(probe_grad, (0,1), overwrite_x=True)
        if multiple_scenarios:
            probe_grad*=exclude_mask_ishift[0,:,:,None, None]
        else:
            probe_grad*=exclude_mask_ishift[0,:,:,None]
        probe_grad=ifft2(probe_grad, (0,1), overwrite_x=True)
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except:
        pass
    return loss, sse, object_grad,  probe_grad, pos_grad, tilts_grad, static_background_grad, aberrations_array_grad, beam_current_grad, constraint_contributions, updated_fast_axis_reg_weight_positions, updated_current_hp_reg_weight_positions, updated_current_hp_reg_weight_tilts, updated_fast_axis_reg_weight_tilts, updated_phase_norm_weight, updated_abs_norm_weight, updated_probe_reg_weight, updated_current_window_weight, updated_atv_weight, updated_mixed_variance_weight




        
def scatteradd_probe(full, indic, batches):
    try:
        cupyx.scatter_add(full.real, (indic), batches.real)
        cupyx.scatter_add(full.imag, (indic), batches.imag)
    except:
        cp.add.at(full, (indic), batches)
        
        
def scatteradd_abers(full, indic, batches):
    try:
        cupyx.scatter_add(full, (indic), batches)
    except:
        cp.add.at(full, (indic), batches)
        
        
        
def charge_flip(a, delta_phase = 0.03, delta_abs = 0.14, beta_phase = -0.95, beta_abs = -0.95, fancy_sigma=None):
    phase=cp.angle(a)
    absorption=-cp.log(cp.abs(a)+1e-10)
    def fftconvolve(a,b):
        return cp.fft.fftshift(cp.fft.ifftn(cp.fft.fftn(a) * cp.fft.fftn(b))).real
    def richardson_lucy(image, psf, tolerance):
        result = cp.copy(image)
        count, doflag, prev_estimate=0, True, None
        while doflag:
            convolved_result = fftconvolve(result, psf)
            relative_blur = image / convolved_result
            error_estimate=fftconvolve(relative_blur, cp.conjugate(psf))
            if count>=1:
                progress=cp.sum((error_estimate-prev_estimate)**2)
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
        r2=cp.sum(cp.array(cp.meshgrid(cp.arange(-shx//2, shx+(-shx//2), 1), cp.arange(-shy//2, shy+(-shy//2), 1), indexing="xy"))**2, 0)
        if do_phase_things:
            sigma_1_phase=fancy_sigma[0][0]
            sigma_2_phase=fancy_sigma[0][1]
            do_richardson_lucy_phase=type(sigma_1_phase)==str
            sigma_1_phase=float(sigma_1_phase)
            psf_1_phase=cp.exp(-r2/sigma_1_phase**2)
            psf_2_phase=cp.exp(-r2/sigma_2_phase**2)
            psf_1_phase/=cp.sum(psf_1_phase)
            psf_2_phase/=cp.sum(psf_2_phase)
            
            if not(do_richardson_lucy_phase):
                psf_1_phase=shift_fft2(psf_1_phase)#+1e-20
        if do_abs_things:
            sigma_1_abs=fancy_sigma[1][0]
            sigma_2_abs=fancy_sigma[1][1]
            do_richardson_lucy_abs=type(sigma_1_abs)==str
            sigma_1_abs=float(sigma_1_abs)
            psf_1_abs =cp.exp(-r2/sigma_1_abs**2)
            psf_2_abs =cp.exp(-r2/sigma_2_abs**2)
            psf_1_abs/=cp.sum(psf_1_abs)
            psf_2_abs/=cp.sum(psf_2_abs)
            if not(do_richardson_lucy_abs):
                psf_1_abs=shift_fft2(psf_1_abs)#+1e-20
        for i in range(phase.shape[-1]):
            for j in range(phase.shape[-2]):
                if do_phase_things:
                    phase[:,:,j,i]-=cp.min(phase[:,:,j,i])
                    if do_richardson_lucy_phase:
                        phase[:,:,j,i]=richardson_lucy(phase[:,:,j,i], psf_1_phase, tolerance=1e-3)
                    else:
                        phase[:,:,j,i]=cp.real((ifft2_ishift(shift_fft2(phase[:,:,j,i])/psf_1_phase)))
            
                if do_abs_things:
                    absorption[:,:,j,i]-=cp.min(absorption[:,:,j,i])
                    if do_richardson_lucy_abs:
                        absorption[:,:,j,i]=richardson_lucy(absorption[:,:,j,i], psf_1_abs, tolerance=1e-3)
                    else:
                        absorption[:,:,j,i]=cp.real(fftshift(ifft2_ishift(shift_fft2(absorption[:,:,j,i])/psf_1_abs)))
    for i in range(phase.shape[-1]):
        for j in range(phase.shape[-2]):
            p=phase[:,:,j,i]
            a=absorption[:,:,j,i]
            p[p < delta_phase*cp.max(p)] *= beta_phase
            a[a < delta_abs*cp.max(a)] *= beta_abs
            phase[:,:,j,i]=p
            absorption[:,:,j,i]=a
    if not(fancy_sigma is None):
        for i in range(phase.shape[-1]):
            for j in range(phase.shape[-2]):
                if do_phase_things:
                    phase[:,:,j,i]=fftconvolve(phase[:,:,j,i], psf_2_phase)
                if do_abs_things:
                    absorption[:,:,j,i]=fftconvolve(absorption[:,:,j,i], psf_2_abs)
    return cp.exp(-absorption+1j*phase)



def make_states_orthogonal(probe_states):
    n_states=probe_states.shape[-1]
    multiple_scenarios=len(probe_states.shape)==4
    for ind1 in range(n_states):
        for ind2 in range(ind1):
            if multiple_scenarios:
                probe_states[:,:,ind1,:]=probe_states[:,:,ind1,:]-probe_states[:,:,ind2,:]*(cp.sum(cp.conjugate(probe_states[:,:,ind2,:])*probe_states[:,:,ind1,:]))/(1e-10+cp.sum(cp.abs(probe_states[:,:,ind2,:])**2))
            else:
                probe_states[:,:,ind1]=probe_states[:,:,ind1]-probe_states[:,:,ind2]*(cp.sum(cp.conjugate(probe_states[:,:,ind2])*probe_states[:,:,ind1]))/(1e-10+cp.sum(cp.abs(probe_states[:,:,ind2])**2))
    return probe_states
    
def make_basis_orthogonal(vectors):
    n=vectors.shape[0]
    for ind1 in range(n):
        for ind2 in range(ind1):
            vectors[ind1]=vectors[ind1]-vectors[ind2]*(cp.sum(cp.conjugate(vectors[ind2])*vectors[ind1]))/(1e-20+cp.sum(cp.abs(vectors[ind2])**2))
    return vectors

        
def clear_missing_wedge(obj, px_size_x_A, px_size_y_A, slice_distance, beta_wedge):
    """Basically it is very similar to the Fourier_clean(), but the mask is cone-shaped and the FFT is done along the xyz axes (and not only along xy)!"""
    qx=fftfreq(obj.shape[1],px_size_x_A)
    qy=fftfreq(obj.shape[0],px_size_y_A)
    qz=fftfreq(obj.shape[2],slice_distance)
    qx,qy,qz=cp.meshgrid(qx,qy,qz, indexing="xy")
    qr=qx**2+qy**2
    qr[qr==0]=1e-10
    weight=(beta_wedge**2)*(qz**2)/qr
    weight=1-0.63661977236*cp.arctan(weight)
    del qx,qy,qz
    fft_times_weight=fftn(obj, axes=(0,1,2))*weight[:,:,:,None]
    fft_times_weight=ifftn(fft_times_weight, axes=(0,1,2))
    return fft_times_weight
    


def compute_fast_axis_constraint_on_grid(something, scan_size, tv_reg_weight):
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





def compute_hp_constraint_on_grid(something, scan_size, reg_weight, a_coeff):
    kr = cp.sum(cp.array(cp.meshgrid(fftfreq(scan_size[1]), fftfreq(scan_size[0]), indexing="xy"))**2, 0)
    kr/=cp.max(kr)
    weight=(1-cp.exp(-0.5*kr/a_coeff**2))*reg_weight
    something_scan_size = 1*something.reshape(scan_size[0], scan_size[1],something.shape[-1])
    grad=fft2(something_scan_size, axes=(0,1))
    grad[0,:,:]=0
    grad[:,0,:]=0
    reg_term=cp.sum((cp.abs(grad)**2) * weight[:,:,None])
    grad=ifft2(grad*weight[:,:,None], axes=(0,1))
    grad=2*cp.real(grad)*scan_size[0]*scan_size[1]
    grad=grad.reshape(something.shape)
    return reg_term, grad
    
    
    
    
    
def compute_full_l1_constraint(object, abs_norm_weight, phase_norm_weight, grad_mask, return_direction, smart_memory):
    reg_term=0
    if return_direction:
        grad=cp.zeros_like(object)
    else:
        grad=None
    if phase_norm_weight>0:
        this_potential=cp.angle(object)*grad_mask[:,:,None,None]
        reg_term+=phase_norm_weight*cp.sum(cp.abs(this_potential))
        if return_direction:
            grad+=1j*phase_norm_weight*cp.sign(this_potential)
    if abs_norm_weight>0:
        this_abs_potential=cp.log(cp.abs(object))*grad_mask[:,:,None,None] # actually a negative of it, but its irrelevant for us!
        reg_term+=abs_norm_weight*cp.sum(cp.abs(this_abs_potential))
        if return_direction:
            grad+=abs_norm_weight*cp.sign(this_abs_potential)
    if return_direction:
        grad=0.5*grad/cp.conjugate(object)
    return reg_term, grad
    
def compute_window_constraint(to_reg_probe, current_window, current_window_weight):
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
    if type(aperture)==cp.ndarray:
        aperture=1-cp.expand_dims(aperture.astype(bool),-1) ### actually a mask
    else:
        ## if aperture is a float, then you should construct a circular mask yourself! Here the aperture is supposed to be a ratio between the convergence and collection angles!
        ffx=fftshift(fftfreq(to_reg_probe.shape[1]))
        ffy=fftshift(fftfreq(to_reg_probe.shape[0]))
        ffx,ffy=cp.meshgrid(ffx,ffy, indexing="xy")
        ffr=cp.expand_dims((ffx**2+ffy**2)**0.5,-1)
        aperture=ffr>(0.5*aperture)
        del ffx,ffy, ffr
    if len(to_reg_probe.shape)==4:
        aperture=cp.expand_dims(aperture,-1)
    probe_fft=shift_fft2(to_reg_probe, axes=(0,1))
    probe_fft=probe_fft * aperture
    reg_term=weight*cp.sum(cp.abs(probe_fft)**2)
    if return_direction:
        probe_fft=ifft2_ishift(probe_fft, axes=(0,1))*(weight*probe_fft.shape[0]*probe_fft.shape[1])
    else:
        probe_fft=None
    return reg_term, probe_fft


def compute_atv_constraint(obj, atv_weight, atv_q, atv_p, pixel_size_x_A, pixel_size_y_A, atv_grad_mask, return_direction, smart_memory):
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


def compute_missing_wedge_constraint(obj, px_size_x_A, px_size_y_A, slice_distance, beta_wedge, wegde_mu):
    qx=fftfreq(obj.shape[1],px_size_x_A)
    qy=fftfreq(obj.shape[0],px_size_y_A)
    qz=fftfreq(obj.shape[2],slice_distance)
    qx,qy,qz=cp.meshgrid(qx,qy,qz)
    weight=0.63661977236*cp.arctan((beta_wedge**2)*(qz**2)/(1e-20+qx**2+qy**2))
    fft_times_weight=fftn(obj, axes=(0,1,2))*weight[:,:,:,None]
    loss_term=wegde_mu*cp.sum(cp.abs(fft_times_weight)**2)
    grad_obj=wegde_mu*ifftn(fft_times_weight*weight[:,:,:,None], axes=(0,1,2))*fft_times_weight.shape[0]*fft_times_weight.shape[1]*fft_times_weight.shape[2]
    return loss_term, grad_obj
    
def compute_mixed_object_variance_constraint(this_obj, weight, sigma, return_direction, smart_memory):
    mask=cp.exp(-cp.sum(cp.array(cp.meshgrid(fftfreq(this_obj.shape[1]), fftfreq(this_obj.shape[0]), indexing="xy"))**2, axis=0)/sigma**2)
    if smart_memory:
        this_obj_blur=cp.copy(this_obj)
        for i in range(this_obj.shape[2]):
            for j in range(this_obj.shape[3]):
                this_obj_blur[:,:,i,j]=fft2(this_obj[:,:,i,j])
                this_obj_blur[:,:,i,j]*=mask
        try:
            cp.fft.config.clear_plan_cache()
        except:
            pass
        for i in range(this_obj.shape[2]):
            for j in range(this_obj.shape[3]):
                this_obj_blur[:,:,i,j]=ifft2(this_obj_blur[:,:,i,j])
        try:
            cp.fft.config.clear_plan_cache()
        except:
            pass
    else:
        this_obj_blur=ifft2(fft2(this_obj, axes=(0,1))*mask[:,:,None,None], axes=(0,1))
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
                    mean_obj[:,:,i,j]=fft2(mean_obj[:,:,i,j])*mask
            try:
                cp.fft.config.clear_plan_cache()
            except:
                pass
            for i in range(this_obj.shape[2]):
                for j in range(this_obj.shape[3]):
                    mean_obj[:,:,i,j]=ifft2(mean_obj[:,:,i,j])
            try:
                cp.fft.config.clear_plan_cache()
            except:
                pass
        else:
            mean_obj=ifft2(fft2(mean_obj, axes=(0,1))*mask[:,:,None,None], axes=(0,1)) ## I am not dumb, the above disaster should be less memory hungry than this beauty
    else:
        del mean_obj, mask
        mean_obj=None
    return reg_term, mean_obj
    


