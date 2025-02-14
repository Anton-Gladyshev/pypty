try:
    import cupyx
    import cupy as cp
    cpu_mode=False
except:
    import numpy as cp
    cpu_mode=True
from pypty.fft import *
from pypty.utils import *

def better_multislice(full_probe, this_obj_chopped, num_slices, n_obj_modes, n_probe_modes, this_distances, this_wavelength, q2, qx, qy, exclude_mask, is_single_dist, this_tan_x,this_tan_y, damping_cutoff_multislice, smooth_rolloff, master_propagator_phase_space,  half_master_propagator_phase_space, mask_clean, waves_multislice, wave, default_float, default_complex):
    if n_obj_modes==1:
        wave=full_probe[:,:,:,:,None]
    else:
        wave=cp.repeat(full_probe[:,:,:,:,None], n_obj_modes, axis=-1)
    for ind_multislice in range(0,num_slices,1):
        transmission_func=this_obj_chopped[:, :,:, ind_multislice:ind_multislice+1, :]
        half_transmission_func=transmission_func**0.5
        half_transmission_func=fourier_clean_3d(half_transmission_func, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        if master_propagator_phase_space is None:
            half_propagator_phase_space=cp.exp(-1.570796326794j*this_distances[ind_multislice]*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*mask_clean
            half_propagator_phase_space=cp.expand_dims(half_propagator_phase_space,(-1,-2))
            propagator_phase_space=half_propagator_phase_space**2 # actually a full propagator over one slice
        else:
            half_propagator_phase_space=half_master_propagator_phase_space
            propagator_phase_space=master_propagator_phase_space
        fourier_wave=fft2(wave,        axes=(1,2)) # Fourier
        
        help_psi_11=wave*half_transmission_func
        help_psi_11=fft2(help_psi_11, axes=(1,2)) ## Fourier
        help_psi_12=ifft2(help_psi_11*half_propagator_phase_space, axes=(1,2)) ## Real
        help_psi_13=help_psi_12*half_transmission_func
        help_psi_13=fft2(help_psi_13, axes=(1,2)) ## Fourier
        help_psi_14=ifft2(help_psi_13*half_propagator_phase_space, axes=(1,2)) ## Real
        
        help_psi_21=ifft2(fourier_wave*half_propagator_phase_space, axes=(1,2)) ## Real
        help_psi_22=help_psi_21*half_transmission_func
        help_psi_22=fft2(help_psi_22, axes=(1,2)) # Forier
        help_psi_23=ifft2(help_psi_22*half_propagator_phase_space, axes=(1,2)) ## Real
        help_psi_24=help_psi_23*half_transmission_func # Real
        help_psi_24=fourier_clean_3d(help_psi_24, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        
        help_psi_31=ifft2(fourier_wave * propagator_phase_space, axes=(1,2)) # Real
        help_psi_32=help_psi_31 * transmission_func # Real
        help_psi_32=fourier_clean_3d(help_psi_32, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp) # Real
        help_psi_41=wave * transmission_func
        help_psi_41=fft2(help_psi_41, axes=(1,2)) ## Fourier
        help_psi_42=ifft2(help_psi_41*propagator_phase_space, axes=(1,2)) ## Real
        
        waves_multislice[:,:,:,ind_multislice, :,:,0]=help_psi_11 ### Fourier
        waves_multislice[:,:,:,ind_multislice, :,:,1]=help_psi_12 ## Real
        waves_multislice[:,:,:,ind_multislice, :,:,2]=help_psi_13 ### Fourier
        waves_multislice[:,:,:,ind_multislice, :,:,3]=help_psi_21 ## Real
        waves_multislice[:,:,:,ind_multislice, :,:,4]=help_psi_22 ### Fourier
        waves_multislice[:,:,:,ind_multislice, :,:,5]=help_psi_23 ## Real
        waves_multislice[:,:,:,ind_multislice, :,:,6]=help_psi_31 ## Real
        waves_multislice[:,:,:,ind_multislice, :,:,7]=help_psi_41 ### Fourier
        waves_multislice[:,:,:,ind_multislice, :,:,8]=wave        # input_wave for this slice!!!
        waves_multislice[:,:,:,ind_multislice, :,:,9]=fourier_wave#Fourier
        wave= (2/3)*(help_psi_14+help_psi_24)-(1/6)*(help_psi_32+help_psi_42)
    return cp.conjugate(waves_multislice), wave ## all waves and just the exit wave
    
def better_multislice_grads(dLoss_dP_out, waves_multislice, this_obj_chopped, object_grad, tilts_grad, is_single_dist, this_distances, exclude_mask, this_wavelength, q2, qx, this_tan_x, qy, this_tan_y, num_slices, n_probe_modes, n_obj_modes,tiltind, this_step_tilts,  master_propagator_phase_space, half_master_propagator_phase_space, damping_cutoff_multislice, smooth_rolloff, tilt_mode,  compute_batch, mask_clean, masked_pixels_y, masked_pixels_x, default_float, default_complex):
    sh0=this_obj_chopped.shape[0]
    if is_single_dist:
        prop_distance=this_distances[0]
        half_propagator_phase_space=cp.conjugate(cp.copy(half_master_propagator_phase_space))
        propagator_phase_space=cp.conjugate(cp.copy(master_propagator_phase_space))
    backshifted_exclude=cp.expand_dims(mask_clean, (-1,-2))
    for i_update in range(num_slices-1,-1,-1):
        if not(is_single_dist):
            prop_distance=this_distances[i_update]
            half_propagator_phase_space=cp.exp(1.570796326794j*this_distances[ind_multislice]*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*ifftshift(exclude_mask)
            propagator_phase_space=propagator_phase_space**2 # actually a full propagator over one slice
        ### get the derivatives of the four sub-exitwaves
        dLoss_dpsi_14=dLoss_dP_out*(2/3)
        dLoss_dpsi_24=dLoss_dP_out*(2/3)
        dLoss_dpsi_32=dLoss_dP_out*(-1/6)
        dLoss_dpsi_42=dLoss_dP_out*(-1/6)
        ## get the  object and half of the object (conjugated)
        transmission_func=cp.conjugate(this_obj_chopped[:,:,:, i_update:i_update+1, :])
        half_transmission_func=transmission_func**0.5
        half_transmission_func=fourier_clean_3d(half_transmission_func, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        #unpack the waves needed for the gradinets
        help_psi_11         = waves_multislice[:,:,:, i_update, :,:,0] ### Fourier , conjugated
        help_psi_12         = waves_multislice[:,:,:, i_update, :,:,1] ### Real    , conjugated
        help_psi_13         = waves_multislice[:,:,:, i_update, :,:,2] ### Fourier , conjugated
        help_psi_21         = waves_multislice[:,:,:, i_update, :,:,3] ### Real    , conjugated
        help_psi_22         = waves_multislice[:,:,:, i_update, :,:,4] ### Fourier , conjugated
        help_psi_23         = waves_multislice[:,:,:, i_update, :,:,5] ### Real    , conjugated
        help_psi_31         = waves_multislice[:,:,:, i_update, :,:,6] ### Real    , conjugated
        help_psi_41         = waves_multislice[:,:,:, i_update, :,:,7] ### Fourier , conjugated
        input_wave0         = waves_multislice[:,:,:, i_update, :,:,8] ### Real    , conjugated
        psi_input_fourier   = waves_multislice[:,:,:, i_update, :,:,9] ### Fourier    , conjugated
        # LETS GO   !!!
        dLoss_dpsi_14_fourier=fft2(dLoss_dpsi_14, axes=(1,2))
        dLoss_dpsi_13=ifft2(dLoss_dpsi_14_fourier*half_propagator_phase_space, axes=(1,2)) ### <<< fourir cleaned 2/3 cutoff
        dLoss_dpsi_12=dLoss_dpsi_13*half_transmission_func ###<<< aliasling!
        dLoss_dpsi_12_fourier=fft2(dLoss_dpsi_12, axes=(1,2))*backshifted_exclude
        dLoss_dpsi_11=ifft2(dLoss_dpsi_12_fourier*half_propagator_phase_space, axes=(1,2)) ###<< 2/3 cutoff
        dLoss_dpsi_23=dLoss_dpsi_24*half_transmission_func
        dLoss_dpsi_23_fourier=fft2(dLoss_dpsi_23, axes=(1,2))*backshifted_exclude
        dLoss_dpsi_22=ifft2(dLoss_dpsi_23_fourier*half_propagator_phase_space, axes=(1,2)) ##2/3 cutoff
        dLoss_dpsi_21=dLoss_dpsi_22*half_transmission_func
        dLoss_dpsi_21_fourier=fft2(dLoss_dpsi_21, axes=(1,2))*backshifted_exclude
        dLoss_dpsi_31=dLoss_dpsi_32*transmission_func
        dLoss_dpsi_31_fourier=fft2(dLoss_dpsi_31, axes=(1,2))*backshifted_exclude
        dLoss_dpsi_42_fourier=fft2(dLoss_dpsi_42, axes=(1,2))*backshifted_exclude
        dLoss_dpsi_41=ifft2(dLoss_dpsi_42_fourier*propagator_phase_space, axes=(1,2))
        dLoss_dP_out=ifft2((fft2(dLoss_dpsi_41 * transmission_func + dLoss_dpsi_11 * half_transmission_func, axes=(1,2))+ dLoss_dpsi_21_fourier*half_propagator_phase_space + dLoss_dpsi_31_fourier * propagator_phase_space)*backshifted_exclude, axes=(1,2)) ### << actually a gradient with respect to an input wave of this slice!
        #dLoss_dP_out=ifft2((fft2(dLoss_dpsi_41 * transmission_func, axes=(0,1))+ dLoss_dpsi_31_fourier * propagator_phase_space)*backshifted_exclude, axes=(0,1))
        dLoss_dObject=dLoss_dpsi_13*help_psi_12 + dLoss_dpsi_11*input_wave0 + dLoss_dpsi_24*help_psi_23 + dLoss_dpsi_22*help_psi_21 # << object half
        dLoss_dObject=(dLoss_dObject*0.5)/half_transmission_func
        dLoss_dObject=cp.sum(dLoss_dObject+dLoss_dpsi_32*help_psi_31+dLoss_dpsi_41*input_wave0, -2) #  i<<full object, summing over the probe modes!!!
        dLoss_dObject=fourier_clean_3d(dLoss_dObject, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        scatteradd(object_grad[:,:,i_update, :], masked_pixels_y, masked_pixels_x, dLoss_dObject)
        if this_step_tilts>0 and (tilt_mode==0 or tilt_mode==3 or tilt_mode==4):
            sh=help_psi_11.shape[1]*help_psi_11.shape[2]
            dLoss_dPropagator=dLoss_dpsi_14_fourier*help_psi_13 + dLoss_dpsi_12_fourier*help_psi_11 + dLoss_dpsi_23_fourier*help_psi_22+ dLoss_dpsi_21_fourier*psi_input_fourier #<<< half propagator
            dLoss_dPropagator=dLoss_dPropagator*0.5*cp.conjugate(half_propagator_phase_space) ## now with respect to the full propagator
            dLoss_dPropagator=dLoss_dPropagator + dLoss_dpsi_31_fourier*psi_input_fourier + dLoss_dpsi_42_fourier*help_psi_41###<<< add the contributions from the third and fourth terms full propagator
            dLoss_dPropagator=dLoss_dPropagator*propagator_phase_space
            dLoss_dPropagator=cp.sum(dLoss_dPropagator, (-1,-2)) ## summing over the modes
            tilts_grad[tiltind,3]+=-12.566370614359172*prop_distance*cp.imag(cp.sum(dLoss_dPropagator*qx))/sh ##prop_distance=-1*distance!
            tilts_grad[tiltind,2]+=-12.566370614359172*prop_distance*cp.imag(cp.sum(dLoss_dPropagator*qy))/sh
    interm_probe_grad=cp.sum(dLoss_dP_out,-1) ## sum over the object modes
    return object_grad,  interm_probe_grad, tilts_grad
def yoshida_multislice(full_probe, this_obj_chopped, num_slices, n_obj_modes, n_probe_modes, this_distances, this_wavelength, q2, qx, qy, exclude_mask, is_single_dist, this_tan_x,this_tan_y, damping_cutoff_multislice, smooth_rolloff, master_propagator_phase_space,  half_master_propagator_phase_space, mask_clean, waves_multislice, wave, default_float, default_complex):
    """yoshida integrator married with multislcie approach"""
    #sh0=this_obj_chopped.shape[0]
    sigma_yoshida=(2+2**(-1/3)+2**(1/3))/3
    if n_obj_modes==1:
        wave=full_probe[:,:,:,:,None]
    else:
        wave=cp.repeat(full_probe[:,:,:,:,None], n_obj_modes, axis=-1)
    for ind_multislice in range(0,num_slices,1):
        transmission_func_1=cp.expand_dims(this_obj_chopped[:, :,:, ind_multislice, :], 3)
        transmission_func_2=transmission_func_1**(0.5-sigma_yoshida/2)
        transmission_func_1=transmission_func_1**(sigma_yoshida/2)
        transmission_func_1=fourier_clean_3d(transmission_func_1, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        transmission_func_2=fourier_clean_3d(transmission_func_2, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        if master_propagator_phase_space is None:
            propagator_phase_space_1=cp.expand_dims(cp.exp(-3.141592653j*sigma_yoshida*this_distances[ind_multislice]*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*mask_clean,(-1,-2))
            propagator_phase_space_2=cp.expand_dims(cp.exp(-3.141592653j*(1-2*sigma_yoshida)*this_distances[ind_multislice]*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*mask_clean,(-1,-2))
        else:
            propagator_phase_space_1=half_master_propagator_phase_space
            propagator_phase_space_2=master_propagator_phase_space
        waves_multislice[:, :,:,ind_multislice, :,:,0]=wave # psi_0 (input), cutoff
        wave=wave*transmission_func_1 #psi1 x2 cutoff
        wave=fft2(wave,        axes=(1,2)) ##psi1_fft x2 cutoff
        waves_multislice[:, :,:,ind_multislice, :,:,1]=wave #x2 cutoff
        wave=ifft2(wave * propagator_phase_space_1, axes=(1,2)) #psi2, cutoff
        waves_multislice[:, :,:,ind_multislice, :,:,2]=wave #cutoff
        wave=wave*transmission_func_2 #psi3 # x2 cutoff
        wave=fft2(wave,        axes=(1,2)) ##psi3_fft # x2 cutoff
        waves_multislice[:, :,:,ind_multislice, :,:,3]=wave # psi3_fft x2 cutoff
        wave=ifft2(wave * propagator_phase_space_2, axes=(1,2)) #psi4 # cutoff
        waves_multislice[:, :,:,ind_multislice, :,:,4]=wave # cutoff
        wave=wave*transmission_func_2 #psi5 # x2 cutoff
        wave=fft2(wave,        axes=(1,2)) ##psi5_fft # x2 cutoff
        waves_multislice[:, :,:,ind_multislice, :,:,5]=wave #  psi5_fft x2 cutoff
        wave=ifft2(wave * propagator_phase_space_1, axes=(1,2)) #psi6 # cutoff
        waves_multislice[:, :,:,ind_multislice, :,:,6]=wave # psi6, cutoff
        wave=wave*transmission_func_1  # exit wave, aka next input wave # x2 cutoff
        wave=fourier_clean_3d(wave, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)# cutoff
    return cp.conjugate(waves_multislice), wave
    
def yoshida_multislice_grads(dLoss_dP_out, waves_multislice, this_obj_chopped, object_grad, tilts_grad, is_single_dist, this_distances, exclude_mask, this_wavelength, q2, qx, this_tan_x, qy, this_tan_y, num_slices, n_probe_modes, n_obj_modes,tiltind, this_step_tilts,  master_propagator_phase_space, half_master_propagator_phase_space, damping_cutoff_multislice, smooth_rolloff, tilt_mode,  compute_batch, mask_clean, masked_pixels_y, masked_pixels_x, default_float, default_complex):
    sigma_yoshida=(2+2**(-1/3)+2**(1/3))/3
    sh0=this_obj_chopped.shape[0]
    if is_single_dist:
        prop_distance=this_distances[0]
        propagator_phase_space_1=cp.conjugate(half_master_propagator_phase_space) #(conjugated)
        propagator_phase_space_2=cp.conjugate(master_propagator_phase_space)# (conjugated)
    for i_update in range(num_slices-1,-1,-1):
        if not(is_single_dist):
            prop_distance=this_distances[i_update]
            propagator_phase_space_1=cp.expand_dims(cp.exp(3.141592653j*sigma_yoshida*this_distances[ind_multislice]*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*mask_clean,(-1,-2)) ### to fix: add clean_mask
            propagator_phase_space_2=cp.expand_dims(cp.exp(3.141592653j*(1-2*sigma_yoshida)*this_distances[ind_multislice]*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*mask_clean,(-1,-2))
        ## get the  object and half of the object (conjugated)
        transmission_func_0=cp.conjugate(cp.expand_dims(this_obj_chopped[:, :,:, i_update, :], 3))
        transmission_func_2=transmission_func_0**(0.5-sigma_yoshida/2)
        transmission_func_1=transmission_func_0**(sigma_yoshida/2)
        transmission_func_3=transmission_func_0**(0.5*sigma_yoshida-1)
        transmission_func_4=transmission_func_0**(0.5+0.5*sigma_yoshida)
        transmission_func_1=fourier_clean_3d(transmission_func_1, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        transmission_func_2=fourier_clean_3d(transmission_func_2, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        transmission_func_3=fourier_clean_3d(transmission_func_3, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        transmission_func_4=fourier_clean_3d(transmission_func_4, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        ## unpack the waves!
        psi0=waves_multislice[:,:,:,i_update, :,:,0] #  all waves are already conjugated!!!     cutoff
        psi1=waves_multislice[:,:,:,i_update, :,:,1] ##  its fft                             x2 cutoff
        psi2=waves_multislice[:,:,:,i_update, :,:,2] #                                          cutoff
        psi3=waves_multislice[:,:,:,i_update, :,:,3] ##  its fft                             x2 cutoff
        psi4=waves_multislice[:,:,:,i_update, :,:,4] #                                          cutoff
        psi5=waves_multislice[:,:,:,i_update, :,:,5] ##  its fft                             x2 cutoff
        psi6=waves_multislice[:,:,:,i_update, :,:,6] #                                          cutoff
        # LETS GO   !!!
        dLoss_dpsi6=dLoss_dP_out*transmission_func_1            #x2 cutoff
        dLoss_dOs=dLoss_dP_out*psi6                             #x2 cutoff
        dLoss_dpsi6_fourier=fft2(dLoss_dpsi6, axes=(1,2))      # x2 cutoff
        dLoss_dpsi5=ifft2(dLoss_dpsi6_fourier*propagator_phase_space_1, axes=(1,2)) #cutoff
        dLoss_dpsi4=dLoss_dpsi5*transmission_func_2              # x2 cutoff
        dLoss_dO2s=dLoss_dpsi5*psi4                             # x2 cutoff
        dLoss_dpsi4_fourier=fft2(dLoss_dpsi4, axes=(1,2))       # x2 cutoff
        dLoss_dpsi3=ifft2(dLoss_dpsi4_fourier*propagator_phase_space_2, axes=(1,2))   ## cutoff
        dLoss_dpsi2=dLoss_dpsi3*transmission_func_2                      # x2 cutoff
        dLoss_dO2s+=dLoss_dpsi3*psi2                                    # x2 cutoff
        dLoss_dpsi2_fourier=fft2(dLoss_dpsi2, axes=(1,2))              # x2 cutoff
        dLoss_dpsi1=ifft2(dLoss_dpsi2_fourier*propagator_phase_space_1, axes=(1,2))   # cutoff
        dLoss_dOs+=dLoss_dpsi1*psi0                                     # x2 cutoff
        dLoss_dP_out=dLoss_dpsi1*transmission_func_1                   # x2 cutoff
        dLoss_dP_out=fourier_clean_3d(dLoss_dP_out, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)  ## cutoff
        dLoss_dOs=fourier_clean_3d(dLoss_dOs, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)  ## cutoff
        dLoss_dO2s=fourier_clean_3d(dLoss_dO2s, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp) ## cutoff
        dLoss_dOs=(0.5*sigma_yoshida)*dLoss_dOs*transmission_func_3+(0.5-0.5*sigma_yoshida)*dLoss_dO2s*transmission_func_4
        dLoss_dObject=cp.sum(dLoss_dOs,-2)
        dLoss_dObject=fourier_clean_3d(dLoss_dObject, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        scatteradd(object_grad[:,:,i_update, :], masked_pixels_y, masked_pixels_x, dLoss_dObject)
        if this_step_tilts>0 and (tilt_mode==0 or tilt_mode==3 or tilt_mode==4):
            sh=psi0.shape[2]*psi0.shape[1]
            dLoss_dPropagator=(dLoss_dpsi4_fourier*psi3*propagator_phase_space_2*(1-2*sigma_yoshida))+(dLoss_dpsi6_fourier*psi5+dLoss_dpsi2_fourier*psi1)*propagator_phase_space_1*sigma_yoshida
            dLoss_dPropagator=cp.sum(dLoss_dPropagator, (-1,-2)) ## summing over the modes
            tilts_grad[tiltind,3]+=-12.566370614359172*prop_distance*cp.imag(cp.sum(dLoss_dPropagator*qx))/sh ##prop_distance=-1*distance!
            tilts_grad[tiltind,2]+=-12.566370614359172*prop_distance*cp.imag(cp.sum(dLoss_dPropagator*qy))/sh
    interm_probe_grad=cp.sum(dLoss_dP_out,-1) ## sum over the object modes
    return object_grad,  interm_probe_grad, tilts_grad

def multislice(full_probe, this_obj_chopped, num_slices, n_obj_modes, n_probe_modes, this_distances, this_wavelength, q2, qx, qy, exclude_mask, is_single_dist, this_tan_x,this_tan_y, damping_cutoff_multislice, smooth_rolloff, master_propagator_phase_space,  half_master_propagator_phase_space, mask_clean, waves_multislice, wave, default_float, default_complex):
    if n_obj_modes==1:
        wave=full_probe[:,:,:,:,None]
    else:
        wave=cp.repeat(full_probe[:,:,:,:,None], n_obj_modes, axis=-1)
    for ind_multislice in range(0,num_slices,1):
        waves_multislice[:,:,:, ind_multislice, :, :,0]=wave ## x1 cutoff
        wave*=this_obj_chopped[:, :,:, ind_multislice:ind_multislice+1, :] # the slice dimension became a singular dimension of the probe modes, x2 cutoff
        if ind_multislice<num_slices-1: ###propagation is done here:
            if is_single_dist:
                propagator_phase_space=master_propagator_phase_space
            else:
                propagator_phase_space=cp.expand_dims((cp.exp(-3.141592654j*this_distances[ind_multislice]*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*mask_clean),(-1,-2))
            wave=fft2(wave, axes=(1,2), overwrite_x=True)
            waves_multislice[:, :,:,ind_multislice, :, :, 1]=wave ### FFT of \psi^{out}_{ind\ multislice}, x2 cutoff, but it is needed only for tilts grads,  we will set the x1 cutoff later!
            wave*=propagator_phase_space
            wave=ifft2(wave, axes=(1,2), overwrite_x=True) # x1 cutoff
    return cp.conjugate(waves_multislice), wave


def multislice_grads(dLoss_dP_out, waves_multislice, this_obj_chopped, object_grad, tilts_grad, is_single_dist,this_distances, exclude_mask, this_wavelength, q2, qx, this_tan_x, qy, this_tan_y, num_slices, n_obj_modes,tiltind, master_propagator_phase_space, this_step_tilts, damping_cutoff_multislice, smooth_rolloff, tilt_mode,  compute_batch, mask_clean, this_step_probe, this_step_obj, this_step_pos_correction, masked_pixels_y, masked_pixels_x, default_float, default_complex, helper_flag_4):
    this_obj_chopped=cp.conjugate(this_obj_chopped)
    for i_update in range(num_slices-1,-1,-1): #backward propagation
        if this_step_obj>0:
            waves_multislice[:, :,:,i_update,:,:,0]*=dLoss_dP_out # we will store the grads here for now. After the slices loop we will clean the grads and add them to the main array
        dLoss_dP_out*=this_obj_chopped[:,:,:,i_update:i_update+1,:] # the slice dimension became a singular dimension of the probe modes, x2 cutoff
        if i_update>0: #backpropgation of the loss gradient to the next (actually previous) slice
            if is_single_dist:
                prop_distance=this_distances[0]
                propagator_phase_space=cp.conjugate(master_propagator_phase_space)
            else:
                prop_distance=this_distances[i_update-1]
                propagator_phase_space=cp.exp(3.141592654j*prop_distance*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*mask_clean ### also the same as a conjugate of forward propagator P*(d)=P(-d)
                propagator_phase_space=cp.expand_dims(propagator_phase_space,(-1,-2)) ## expanding
            dLoss_dP_out=fft2(dLoss_dP_out, axes=(1,2), overwrite_x=True) ## x2 cutoff
            dLoss_dP_out*=propagator_phase_space ### reuse it for tilts update!!!!
            if this_step_tilts>0 and (tilt_mode==0 or tilt_mode==3 or tilt_mode==4):
                sh=-12.566370614359172*prop_distance/(waves_multislice.shape[1]*waves_multislice.shape[2])
                dLoss_dPropagator=cp.sum(waves_multislice[:, :,:,i_update-1,:,:,1]*dLoss_dP_out, (3,4))
                tilts_grad[tiltind,3]+=sh*cp.sum(cp.imag(dLoss_dPropagator*qx), (1,2))
                tilts_grad[tiltind,2]+=sh*cp.sum(cp.imag(dLoss_dPropagator*qy), (1,2))
            dLoss_dP_out=ifft2(dLoss_dP_out, axes=(1,2), overwrite_x=True) #  x1 cutoff
    if this_step_obj>0:
        if waves_multislice.shape[-3]==1:
            this_grad=waves_multislice[:,:,:,:,0,:,0] #just  one probe mode, x2 cutoff
        else:
            this_grad=cp.sum(waves_multislice[:,:,:,:,:,:,0], -2) #sum over all probe modes, x2 cutoff
        this_grad=fft2(this_grad, (1,2), overwrite_x=True)
        this_grad*=mask_clean[:,:,:,None,None]
        this_grad=ifft2(this_grad, (1,2), overwrite_x=True)
        scatteradd(object_grad, masked_pixels_y, masked_pixels_x, this_grad)
    if helper_flag_4:
        if waves_multislice.shape[-2]==1:
            interm_probe_grad=dLoss_dP_out[:,:,:,:,0]
        else:
            interm_probe_grad=cp.sum(dLoss_dP_out,-1) ## sum over obj modes
    else:
        interm_probe_grad=None
    return object_grad, interm_probe_grad, tilts_grad


def scatteradd(full, masky, maskx, chop):
    try:
        cupyx.scatter_add(full.real, (masky, maskx), chop.real)
        cupyx.scatter_add(full.imag, (masky, maskx), chop.imag)
    except:
        cp.add.at(full, (masky, maskx), chop)
