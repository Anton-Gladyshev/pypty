import numpy as np
import sys
import os
import csv
import h5py
import time
import random
import pickle
import types
import copy
import inspect
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brent, least_squares
from scipy.interpolate import CloughTocher2DInterpolator, griddata, RectBivariateSpline
from scipy.ndimage import rotate, binary_closing
from scipy.ndimage import percentile_filter as cpu_percentile
from tqdm import tqdm

try:
    import cupyx.scipy.ndimage as ndi
    import cupyx
    import cupy as cp
    import cupyx.scipy.fft as sf
    cpu_mode=False
except:
    import scipy.ndimage as ndi
    import numpy as cp
    import scipy.fft as sf
    cpu_mode=True
    print("cpu mode!")



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




half_master_propagator_phase_space, master_propagator_phase_space, q2, qx, qy, exclude_mask, x_real_grid_tilt, y_real_grid_tilt, shift_probe_mask_x, shift_probe_mask_y, exclude_mask_ishift, probe_runx, probe_runy, yx_real_grid_tilt, shift_probe_mask_yx, aberrations_polynomials=None, None,None, None,None, None,None, None, None, None, None, None,None,None,None,None
def loss_and_direction(this_obj, full_probe, this_pos_array, this_pos_correction, this_tilt_array, this_tilts_correction, this_distances,  measured_array,  algorithm_type, this_wavelength, this_step_probe, this_step_obj, this_step_pos_correction, this_step_tilts, masks, pixel_size_x_A, pixel_size_y_A, recon_type, Cs, defocus_array, alpha_near_field, damping_cutoff_multislice, smooth_rolloff, propmethod, this_chopped_sequence, load_one_by_one, data_multiplier, data_pad, phase_plate_in_h5, this_loss_weight, data_bin, data_shift_vector, upsample_pattern, static_background, this_step_static_background, tilt_mode, aberration_marker, probe_marker, aberrations_array, compute_batch, phase_only_obj, beam_current, this_beam_current_step, this_step_aberrations_array, default_float, default_complex, xp, is_first_epoch,
    scan_size,fast_axis_reg_weight_positions, current_slow_axis_reg_weight_positions,current_slow_axis_reg_coeff_positions, current_slow_axis_reg_weight_tilts,current_slow_axis_reg_coeff_tilts, fast_axis_reg_weight_tilts, aperture_mask, probe_reg_weight, current_window_weight, current_window, phase_norm_weight, abs_norm_weight, atv_weight, atv_q, atv_p, mixed_variance_weight, mixed_variance_sigma):
    global pool, pinned_pool, half_master_propagator_phase_space, master_propagator_phase_space, q2, qx, qy, exclude_mask,exclude_mask_ishift, x_real_grid_tilt, y_real_grid_tilt,yx_real_grid_tilt, shift_probe_mask_x, shift_probe_mask_y,shift_probe_mask_yx, probe_runx,probe_runy, aberrations_polynomials
    if is_first_epoch:
        half_master_propagator_phase_space, master_propagator_phase_space, q2, qx, qy, exclude_mask,exclude_mask_ishift, x_real_grid_tilt, y_real_grid_tilt,yx_real_grid_tilt, shift_probe_mask_x, shift_probe_mask_y,shift_probe_mask_yx, probe_runx,probe_runy, aberrations_polynomials= None, None,None,None,None,None,None,None,None,None,None,None,None,None,None,None
   # checking the situation and prepare a bunch of flags
   # start_gpu = cp.cuda.Event()
   # end_gpu = cp.cuda.Event()
   # start_gpu.record()
    if 'lsq_compressed'==algorithm_type or algorithm_type=='epie_compressed':
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
    helper_flag_1= this_step_probe>0 or this_beam_current_step>0
    helper_flag_2= helper_flag_1 or this_step_aberrations_array>0
    helper_flag_3= helper_flag_2 or (this_step_tilts>0 and (tilt_mode==2 or tilt_mode==4))
    helper_flag_4= helper_flag_3 or this_step_pos_correction>0
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
    probe_shift_flag=cp.sum(this_pos_correction**2)!=0.0 or this_step_pos_correction>0
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
        if this_step_probe>0:
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
                measured, *_ = preprocess_dataset(measured, False, algorithm_type, recon_type, data_shift_vector, data_bin, data_pad, upsample_pattern, data_multiplier, xp, False) ### preprocess
                if data_pad!=0:
                    measured=xp.pad(measured, [[0,0],[data_pad//upsample_pattern,data_pad//upsample_pattern],[data_pad,data_pad//upsample_pattern]])
        else:
            measured=measured_array[tcs]
            measured, *_ = preprocess_dataset(measured, False, algorithm_type, recon_type, data_shift_vector, data_bin, data_pad, upsample_pattern, data_multiplier, xp, False)
            if data_pad!=0:
                measured=xp.pad(measured, [[0,0],[data_pad//upsample_pattern,data_pad//upsample_pattern],[data_pad//upsample_pattern,data_pad//upsample_pattern]])

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
        if this_step_static_background>0 and static_background_is_there:
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
            if this_step_tilts>0:
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
            if this_step_tilts>0:
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
                if this_step_pos_correction>0:
                    shift_mask_grad=cp.conjugate(this_probe_fourier)*fourier_probe_grad ## vectorize this one as well
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
                    if this_step_tilts>0:
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
                        if this_step_aberrations_array>0:
                            sh = 2/(fourier_probe_grad.shape[1]*fourier_probe_grad.shape[2])
                            defgr=sh*cp.sum(cp.real((fourier_probe_grad*cp.conjugate(this_fourier_probe_before_local_aberrations))[:,None, :,:,:]*cp.conjugate(aberrations_polynomials[None,:,:,:,None])), axis=(2,3,4), dtype=cp.float64).astype(default_float)
                            scatteradd_abers(aberrations_array_grad, aberration_marker[tcs], defgr)
                            #for dumbindex, t in enumerate(tcs):
                             #   aberrations_array_grad[aberration_marker[t],:]+=defgr[dumbindex,:]
                    if helper_flag_1:
                        if phase_plate_active:
                            if fourier_probe_grad is None: fourier_probe_grad=fft2(interm_probe_grad, axes=(1,2));
                            interm_probe_grad=ifft2(fourier_probe_grad*cp.conjugate(this_phase_plate[:,:,:,None]), (1,2))
                        if fluctuating_current_flag:
                            if this_beam_current_step>0:
                                if n_probe_modes==1:
                                    beam_current_grad[tcs]=2*cp.sign(thisbc)*cp.sum(cp.real(interm_probe_grad[:,:,:,0]*cp.conjugate(this_probe_before_fluctuations[:,:,:,0])), (1,2), dtype=cp.float64).astype(default_float)
                                else:
                                    beam_current_grad[tcs]=2*cp.sign(thisbc)*cp.sum(cp.real(interm_probe_grad*cp.conjugate(this_probe_before_fluctuations)), (1,2,3))
                            if this_step_probe>0:
                                interm_probe_grad=interm_probe_grad*beam_current_values
                        if this_step_probe>0:
                            if multiple_scenarios:
                                scatteradd_probe(probe_grad, probe_marker[tcs], interm_probe_grad)
                            else:
                                probe_grad+=cp.sum(interm_probe_grad,0)
    loss*=this_loss_weight
    if this_step_probe>0:
        if multiple_scenarios: probe_grad=cp.moveaxis(probe_grad, 0,3);
        probe_grad=fft2(probe_grad, (0,1), overwrite_x=True)
        if multiple_scenarios:
            probe_grad*=exclude_mask_ishift[0,:,:,None, None]
        else:
            probe_grad*=exclude_mask_ishift[0,:,:,None]
        probe_grad=ifft2(probe_grad, (0,1), overwrite_x=True);
   # end_gpu.record()
   # end_gpu.synchronize()
   # t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
   # print("\n", t_gpu)
    if this_step_pos_correction>0 and fast_axis_reg_weight_positions>0:
        something=this_pos_array+this_pos_correction
        ind_loss, reg_grad=compute_fast_axis_constraint_on_grid(something, scan_size, fast_axis_reg_weight_positions)
        pos_grad+=reg_grad;
        loss+=ind_loss
    if this_step_pos_correction>0 and current_slow_axis_reg_weight_positions>0:
        something=this_pos_array+this_pos_correction
        ind_loss, reg_grad = compute_slow_axis_constraint_on_grid(something, scan_size, current_slow_axis_reg_weight_positions, current_slow_axis_reg_coeff_positions)
        pos_grad+=reg_grad;
        loss+=ind_loss
    if this_step_tilts>0 and current_slow_axis_reg_weight_tilts>0:
        something=this_tilt_array+this_tilts_correction
        for i_t_ind in range(0,6,2):
            ind_loss, reg_grad=compute_slow_axis_constraint_on_grid(something[:,i_t_ind:i_t_ind+2], scan_size, current_slow_axis_reg_weight_tilts, current_slow_axis_reg_coeff_tilts)
            tilts_grad[:,i_t_ind:i_t_ind+2]+=reg_grad;
            loss+=ind_loss
    if this_step_tilts>0 and fast_axis_reg_weight_tilts>0:
        something=this_tilt_array+this_tilts_correction
        ind_loss, reg_grad=compute_fast_axis_constraint_on_grid(something, scan_size, fast_axis_reg_weight_tilts)
        tilts_grad+=reg_grad
        loss+=ind_loss
    if (phase_norm_weight+abs_norm_weight)>0: # l_1 norm of the potentials
        grad_mask=generate_mask_for_grad_from_pos(this_obj.shape[1], this_obj.shape[0], this_pos_array, full_probe.shape[1],full_probe.shape[0], 0)
        l1_reg_term, l1_object_grad=compute_full_l1_constraint(this_obj, abs_norm_weight, phase_norm_weight, grad_mask, True, smart_memory)
        loss+=l1_reg_term
        object_grad+=l1_object_grad
        del grad_mask,l1_reg_term,l1_object_grad # forget about it
    if probe_reg_weight>0 and this_step_probe>0:
        probe_reg_term, reg_probe_grad = compute_probe_constraint(full_probe, aperture_mask, probe_reg_weight, True)
        loss+=probe_reg_term
        probe_grad+=reg_probe_grad
        del reg_probe_grad, probe_reg_term
    if this_step_probe>0 and current_window_weight>0:
        probe_reg_term, reg_probe_grad = compute_window_constraint(full_probe, current_window, current_window_weight)
        loss+=probe_reg_term
        probe_grad+=reg_probe_grad
        del reg_probe_grad, probe_reg_term #forget about it
    if atv_weight>0:
        atv_reg_term, atv_object_grad = compute_atv_constraint(this_obj, atv_weight, atv_q, atv_p, pixel_size_x_A, pixel_size_y_A, None, True, smart_memory)
        loss+=atv_reg_term
        object_grad+=atv_object_grad
        del atv_object_grad, atv_reg_term
    if mixed_variance_weight>0 and this_obj.shape[-1]>1:
        mixed_variance_reg_term, mixed_variance_grad=compute_mixed_object_variance_constraint(this_obj, mixed_variance_weight, mixed_variance_sigma, True, smart_memory)
        loss+=mixed_variance_reg_term
        object_grad+=mixed_variance_grad
        del mixed_variance_reg_term, mixed_variance_grad # forget about it
    if loss!=loss:
        raise ValueError('A very specific bad thing. Loss is Nan.')
    return loss, sse, object_grad,  probe_grad, pos_grad, tilts_grad, static_background_grad, aberrations_array_grad, beam_current_grad

#------- utils.py-----------------------------------------------------------------------------------

def scatteradd(full, masky, maskx, chop):
    try:
        cupyx.scatter_add(full.real, (masky, maskx), chop.real)
        cupyx.scatter_add(full.imag, (masky, maskx), chop.imag)
    except:
        cp.add.at(full, (masky, maskx), chop)
        
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
        

def create_spatial_frequencies(px, py, shape, damping_cutoff_multislice, smooth_rolloff, default_float):
    qx,qy= cp.meshgrid(fftfreq(shape, px), fftfreq(shape, py), indexing="xy")
    qx,qy=qx.astype(default_float),qy.astype(default_float)
    q2=qx**2+qy**2
    if damping_cutoff_multislice<0:
        damping_cutoff_multislice=2/3
    max_x=cp.max(qx)
    if damping_cutoff_multislice is None:
        damping_cutoff_multislice=2
    r_max=(max_x*damping_cutoff_multislice)**2
    exclude_mask_ishift=(q2<=r_max).astype(default_float)
    if smooth_rolloff!=0:
        r0=(max_x*(damping_cutoff_multislice-smooth_rolloff))**2
        exclude_mask_ishift[q2>r0]*=(0.5*(1+cp.cos(3.141592654*(q2-r0)/(r_max-r0))))[q2>r0]
    exclude_mask_ishift=exclude_mask_ishift.astype(default_float)
    exclude_mask=fftshift(exclude_mask_ishift)
    return q2, qx, qy, exclude_mask, exclude_mask_ishift
def shift_probe_fourier(probe, shift_px):
    maskx, masky=cp.meshgrid(fftfreq(probe.shape[1]), fftfreq(probe.shape[0]), indexing="xy")
    mask=cp.exp(-6.283185307179586j*(maskx*shift_px[1]+masky*shift_px[0]))
    phat=fft2(probe, axes=(0,1))
    probe=ifft2(mask[:,:,None]*phat, axes=(0,1))
    return probe, mask, phat, maskx, masky
def generate_mask_for_grad_from_pos(shapex, shapey, positions_list, shape_footprint_x,shape_footprint_y, shrink=0):
    mask=cp.zeros((shapey,shapex))
    for p in positions_list:
        py,px=p
        mask[shrink+py:py+shape_footprint_y-shrink,shrink+px:px+shape_footprint_x-shrink]=1
    return mask
def complex_grad_to_phase_grad(grad, array):
    '''Transform a Wirtinger derivative dL/da* to the graidnet with respect to phase of a, dL/dp, where a=c*exp(ip).'''
    #array_real, array_imag, grad_real, grad_imag=cp.real(array),cp.imag(array), cp.real(grad), cp.imag(grad)
    return 2*cp.real(-1j*cp.conjugate(array)*grad) #2*(grad_imag*array_real-array_imag*grad_real)  #
def complex_grad_to_phase_abs_grad(grad, array):
    '''Transform a Wirtinger derivative dL/dz* to the graidnet with respect to phase dL/dp and negative amplitude dL/da  where z=exp(-a+ip)  '''
    array_real, array_imag, grad_real, grad_imag=cp.real(array),cp.imag(array), cp.real(grad), cp.imag(grad)
    return 2*(grad_imag*array_real-array_imag*grad_real), -2*(array_real*grad_real+array_imag*grad_imag)
    
def complex_grad_to_mag_grad(grad, abs, phase):
    array=abs*cp.exp(-1j*phase)
    return 4*cp.real(grad*array)

def construct_update_abs_proto_phase(object_grad, obj):
    obj_real, obj_imag, grad_real, grad_imag=cp.real(obj),cp.imag(obj), cp.real(object_grad), cp.imag(object_grad)  ### First you compute the gradss with respect to the phase and abs-potnetial.
    phase_grad,abs_grad=2*(grad_imag*obj_real-obj_imag*grad_real), -2*(obj_real*grad_real+obj_imag*grad_imag)
    abs_grad=phase_grad*cp.sum(phase_grad*abs_grad)/(1e-20+cp.sum(phase_grad**2)) ## project the absorption grad on the phase grad!
    real_imag_prod=obj_real*obj_imag
    obj_update=0.5*(1j*abs_grad+phase_grad)*cp.conjugate(obj)/(cp.sign(real_imag_prod)*(1e-20+cp.abs(real_imag_prod)))
    return obj_update
    



def wolfe_1(value, new_value, d_value, step, wolfe_c1=0.5):
    return new_value <= value+wolfe_c1*step*d_value
    
def wolfe_2(d_value, new_d_value, wolfe_c2=0.9):
    return -1*new_d_value <= -1* d_value * wolfe_c2

def upsample_something_3d(something, upsample, scale=True, xp=np):
    if scale:
        something /= (upsample ** 2)
    something_new = xp.repeat(xp.repeat(something, upsample, axis=1), upsample, axis=2)
    return something_new
    


def downsample_something_3d(something, upsample, xp):
    shape = something.shape
    #rem1, rem2=(shape[1]) % upsample, (shape[2]) % upsample
    something=something[:, :upsample*(shape[1] // upsample), :upsample*(shape[1] // upsample)]
    something_reshaped = something.reshape(shape[0], shape[1] // upsample, upsample, shape[2] // upsample, upsample)
    something_new = something_reshaped.sum(axis=(2, 4))
    return something_new


def upsample_something(something, upsample, scale=True, xp=np):
    if scale:
        something /= (upsample ** 2)
    something_new = xp.repeat(xp.repeat(something, upsample, axis=0), upsample, axis=1)
    return something_new
    
def downsample_something(something, upsample, xp):
    shape = something.shape
    something=something[ :upsample*(shape[1] // upsample), :upsample*(shape[1] // upsample)]
    something_reshaped = something.reshape(shape[0] // upsample, upsample, shape[1] // upsample, upsample)
    something_new = something_reshaped.sum(axis=(1, 3))
    return something_new
    

def preprocess_dataset(dataset, load_one_by_one, algorithm_type, recon_type, data_shift_vector, data_bin, data_pad, upsample_pattern, data_multiplier, xp, force_pad):
    if not(load_one_by_one) and not("compressed" in algorithm_type):
        if data_shift_vector[0]!=0:
            dataset=xp.roll(dataset, data_shift_vector[0], axis=1)
            if data_shift_vector[0]>0:
                dataset[:,-data_shift_vector[0]:,:]=0
            else:
                dataset[:,:-data_shift_vector[0],:]=0
        if data_shift_vector[1]!=0:
            dataset=xp.roll(dataset, data_shift_vector[1], axis=2)
            if data_shift_vector[1]>0:
                dataset[:,:,-data_shift_vector[1]:]=0
            else:
                dataset[:,:, :-data_shift_vector[1]]=0
        if data_bin>1:
            dataset2=xp.zeros((dataset.shape[0], dataset.shape[1]//data_bin, dataset.shape[2]//data_bin))
            for i1 in range(data_bin):
                for j1 in range(data_bin):
                    dataset2+=dataset[:,i1::data_bin, i2::data_bin]
            dataset=dataset2
            del dataset2
        if data_multiplier!=1:
            dataset*=data_multiplier
        data_shift_vector, data_bin, data_multiplier=[0,0], 1, 1
        if force_pad:
            data_pad=data_pad//upsample_pattern
            dataset=xp.pad(dataset, [[0,0],[data_pad,data_pad],[data_pad,data_pad]])
            data_pad=0
    return dataset, data_shift_vector, data_bin, data_pad, data_multiplier


def get_window(shape, r0, r_max, inverted=True):
    x=cp.arange(shape)
    x=x-cp.mean(x)
    r_max*=cp.max(x)
    r0*=cp.max(x)
    x,y=cp.meshgrid(x,x, indexing="xy")
    r=(x**2+y**2)**0.5
    mask=cp.ones_like(r)
    if r0<r_max:
        mask[np.abs(x)>r0]*=0.5*(1+cp.cos(3.141592654*(np.abs(x)-r0)/(r_max-r0)))[np.abs(x)>r0]
        mask[np.abs(y)>r0]*=0.5*(1+cp.cos(3.141592654*(np.abs(y)-r0)/(r_max-r0)))[np.abs(y)>r0]
    mask[cp.abs(x)>r_max]=0
    mask[cp.abs(y)>r_max]=0
    if inverted:
        mask=1-mask
    return mask


def convert_num_to_nmab(num_abs):
    max_n=int(np.ceil(0.5*(-5+np.sqrt(25+8*num_abs))))
    allns=np.arange(1,max_n+1,1)
    possible_n=[]
    possible_m=[]
    possible_ab=[]
    for thisn in allns:
        all_m=np.arange((thisn+1)%2, thisn+2,2)
        for thism in all_m:
            if thism==0:
                possible_n.append(thisn)
                possible_ab.append("")
                possible_m.append(thism)
            else:
                possible_ab.append("a")
                possible_m.append(thism)
                possible_n.append(thisn)
                possible_ab.append("b")
                possible_n.append(thisn)
                possible_m.append(thism)
    possible_n=possible_n[:num_abs]
    possible_m=possible_m[:num_abs]
    possible_ab=possible_ab[:num_abs]
    return possible_n, possible_m, possible_ab
    
def nmab_to_strings(possible_n, possible_m, possible_ab):
    stings=[]
    for i in range(len(possible_n)): stings.append("C%d%d%s"%(possible_n[i], possible_m[i], possible_ab[i]));
    return stings
    

def get_ctf_matrix(kx, ky, num_abs, wavelength, xp=cp):
    possible_n, possible_m, possible_ab = convert_num_to_nmab(num_abs)
    k_r = xp.sqrt(kx**2 + ky**2)
    phi = xp.angle(kx + 1j * ky)
    A = xp.zeros((num_abs, k_r.shape[0],k_r.shape[1]))
    for i in range(num_abs):
        this_n, this_m, this_ab = possible_n[i], possible_m[i], possible_ab[i]
        term = (k_r**(this_n + 1)) / (this_n + 1)
        if this_ab == "a":
            term *= xp.cos(this_m * phi)
        elif this_ab == "b":
            term *= xp.sin(this_m * phi)
        A[i,:,:] = term
    A *= 6.283185307179586/wavelength
    return A

def get_ctf(aberrations, kx, ky, wavelength, angle_offset=0):
    ctf=np.zeros_like(kx)
    num_abs=len(aberrations)
    possible_n, possible_m, possible_ab=convert_num_to_nmab(num_abs)
    k_r=(kx**2+ky**2)**0.5
    phi=np.angle(kx+1j*ky) + angle_offset
    for i in range(num_abs):
        this_n, this_m, this_ab, ab=possible_n[i], possible_m[i], possible_ab[i], aberrations[i]
        term=ab*(k_r**(this_n+1))/(this_n+1)
        if this_ab=="a":
            term*=np.cos(this_m*phi)
        elif this_ab=="b":
            term*=np.sin(this_m*phi)
        ctf+=term
    ctf*=6.283185307179586/wavelength
    return ctf
    
def get_ctf_derivatives(aberrations, kx, ky, wavelength, angle_offset=0):
    ctf_kr_der, ctf_phi_der=np.zeros_like(kx), np.zeros_like(ky)
    num_abs=len(aberrations)
    possible_n, possible_m, possible_ab=convert_num_to_nmab(num_abs)
    k_r=(kx**2+ky**2)**0.5
    phi=np.angle(kx+1j*ky) + angle_offset
    kxp=k_r*np.cos(phi)
    kyp=k_r*np.sin(phi)
    for i in range(num_abs):
        this_n, this_m, this_ab, ab=possible_n[i], possible_m[i], possible_ab[i], aberrations[i]
        this_dctf_dkr=ab*k_r**(this_n-1) ## here we omitted one power of k_r because later on we have to divide by kr
        if this_ab=="a":
            ctf_phi_der-=this_dctf_dkr*np.sin(this_m*phi)*this_m/(this_n+1) ## here we have omitted two powers of kr
            ctf_kr_der+=this_dctf_dkr*np.cos(this_m*phi)
        if this_ab=="b":
            ctf_phi_der+=this_dctf_dkr*np.cos(this_m*phi)*this_m/(this_n+1) ## here we have omitted two powers of kr
            ctf_kr_der+=this_dctf_dkr*np.sin(this_m*phi)
        if this_ab=="":
            ctf_kr_der+=this_dctf_dkr
    ctf_x_der=ctf_kr_der*kxp-ctf_phi_der*kyp
    ctf_y_der=ctf_kr_der*kyp+ctf_phi_der*kxp
    ctf_x_der*=6.283185307179586/wavelength
    ctf_y_der*=6.283185307179586/wavelength
    return ctf_x_der, ctf_y_der
    
    
def get_ctf_gradient_rotation_angle(aberrations, kx, ky, wavelength, angle_offset=0):
    num_abs=len(aberrations)
    possible_n, possible_m, possible_ab=convert_num_to_nmab(num_abs)
    k_r=(kx**2+ky**2)**0.5
    phi=np.angle(kx+1j*ky)+angle_offset
    del_chix_del_phi, del_chiy_del_phi=np.zeros_like(kx), np.zeros_like(ky)
    for i in range(num_abs):
        this_n, this_m, this_ab, ab=possible_n[i], possible_m[i], possible_ab[i], aberrations[i]
        mphi=this_m*phi
        r_n=k_r**this_n
        if this_ab=="b":
            del_chix_del_phi+=ab*r_n*(((this_m**2/(1+this_n)-1))*np.sin(phi)*np.sin(mphi)+this_m*(1-1/(this_n+1))*np.cos(phi)*np.cos(mphi))
            del_chiy_del_phi+=ab*r_n*(np.cos(phi)*np.sin(mphi)*(1+(this_m**2/(1+this_n)))+np.sin(phi)*np.cos(mphi)*this_m*(1+1/(this_n+1)))
        else:
            del_chix_del_phi+=ab*r_n*(np.sin(phi)*np.cos(mphi)*(-1+this_m**2/(this_n+1))  +  np.cos(phi)*np.sin(mphi)*this_m*(1+1/(this_n+1)))
            del_chiy_del_phi+=ab*r_n*(np.cos(phi)*np.cos(mphi)*(1+this_m**2/(this_n+1)) - np.sin(phi)*np.sin(mphi)*this_m*(1+1/(this_n+1)))
    return del_chix_del_phi, del_chiy_del_phi

def apply_defocus_probe(probe, distance,acc_voltage, pixel_size_x_A, pixel_size_y_A,default_complex, default_float, xp):
    wavelength=12.4/np.sqrt(acc_voltage*(acc_voltage+2*511))
    q2, qx, qy, exclude_mask, exclude_mask_ishift=create_spatial_frequencies(pixel_size_x_A, pixel_size_y_A,probe.shape[0], 2/3, 0, default_float)
    q2, exclude_mask_ishift, =xp.asarray(q2), xp.asarray(exclude_mask_ishift)
    propagator_phase_space=np.exp(-1j*xp.pi*distance*wavelength*q2)*exclude_mask_ishift
    if len(probe.shape)==3:
        wave_fourier=fft2(probe, axes=(0,1))*propagator_phase_space[:,:,None]
    else:
        wave_fourier=fft2(probe, axes=(0,1))*propagator_phase_space[:,:,None, None]
    return ifft2(wave_fourier, axes=(0,1))


def padfft(array, pad):
    array=np.fft.fftshift(np.fft.fft2(array, axes=(0,1)), axes=(0,1))
    a2=np.zeros((array.shape[0]+2*pad, array.shape[1]+2*pad, array.shape[2]), dtype=np.complex128)
    a2[pad:-pad, pad:-pad,:]=array
    return np.fft.ifft2(np.fft.ifftshift(a2, axes=(0,1)), axes=(0,1))


def padprobetodatafarfield(probe, measured_data_shape, data_pad, upsample_pattern):
    probeshape=probe.shape
    diff0=upsample_pattern*measured_data_shape[1]+2*data_pad-probeshape[0]
    diff1=upsample_pattern*measured_data_shape[1]+2*data_pad-probeshape[1]
    if diff0>0:
        p1=diff0//2
        p2=diff0-p1
        pf=np.fft.fftshift(np.fft.fft2(probe, axes=(0,1)), axes=(0,1))
        pf=np.pad(pf, [[p1, p2], [0,0],[0,0]])
        probe=np.fft.ifft2(np.fft.ifftshift(pf,axes=(0,1)),axes=(0,1))
    if diff0<0:
        diff0*=-1
        p1=diff0//2
        p2=diff0-p1
        pf=np.fft.fftshift(np.fft.fft2(probe, axes=(0,1)), axes=(0,1))
        pf=pf[p1:-p2, :,:]
        probe=np.fft.ifft2(np.fft.ifftshift(pf,axes=(0,1)),axes=(0,1))
    if diff1>0:
        p1=diff1//2
        p2=diff1-p1
        pf=np.fft.fftshift(np.fft.fft2(probe, axes=(0,1)), axes=(0,1))
        pf=np.pad(pf, [[0, 0], [p1,p2],[0,0]])
        probe=np.fft.ifft2(np.fft.ifftshift(pf,axes=(0,1)),axes=(0,1))
    if diff1<0:
        diff1*=-1
        p1=diff1//2
        p2=diff1-p1
        pf=np.fft.fftshift(np.fft.fft2(probe, axes=(0,1)), axes=(0,1))
        pf=pf[:, p1:-p2,:]
        probe=np.fft.ifft2(np.fft.ifftshift(pf,axes=(0,1)),axes=(0,1))
    return probe
    
    
def padprobetodatanearfield(probe, measured_data_shape, data_pad, upsample_pattern):
    probeshape=probe.shape
    diff0=upsample_pattern*measured_data_shape[1]+2*data_pad-probeshape[0]
    diff1=upsample_pattern*measured_data_shape[1]+2*data_pad-probeshape[1]
    if diff0>0:
        p1=diff0//2
        p2=diff0-p1
        probe=np.pad(probe, [[p1, p2], [0,0],[0,0]])
    if diff0<0:
        diff0*=-1
        p1=diff0//2
        p2=diff0-p1
        probe=probe[p1:-p2, :,:]
    if diff1>0:
        p1=diff1//2
        p2=diff1-p1
        probe=np.pad(probe, [[0, 0], [p1,p2],[0,0]])
    if diff1<0:
        diff1*=-1
        p1=diff1//2
        p2=diff1-p1
        probe=probe[:, p1:-p2,:]
    return probe
    
    


##-------------constraints.py-------------------------------------------------------------------------------
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

 
def fourier_clean_3d(array, cutoff=0.66, mask=None, rolloff=0, default_float=cp.float32, xp=cp):
    if not(cutoff is None) or not(mask is None):
        shape=array.shape
        if mask is None:
            x=fftshift(fftfreq(shape[2]))
            y=fftshift(fftfreq(shape[1]))
            x,y=cp.meshgrid(x,y, indexing="xy")
            r=x**2+y**2
            max_r=0.5*cutoff ## maximum freq for fftfreq is 0.5 when the px size is not specified
            mask=(r<=max_r**2).astype(default_float)
            if rolloff!=0:
                r0=(0.5*(cutoff-rolloff))**2
                mask[r>r0]*=0.5*(1+cp.cos(3.141592654*(r-r0)/(max_r**2-r0)))[r>r0]
            del x,y, r
        arrayff=shift_fft2(array, axes=(1,2), overwrite_x=True)
        if len(shape)==6:
            arrayff=arrayff*mask[None, :,:, None,None,None]
        if len(shape)==5:
            arrayff=arrayff*mask[None, :,:, None, None]
        if len(shape)==4:
            arrayff=arrayff*mask[None, :,:, None]
        if len(shape)==3:
            arrayff=arrayff*mask[None, :,:]
        arrayff=ifft2_ishift(arrayff, axes=(1,2), overwrite_x=True)
        del mask
        return arrayff
    else:
        return array
    

def fourier_clean(array, cutoff=0.66, mask=None, rolloff=0, default_float=cp.float32, xp=cp):
    if not(cutoff is None) or not(mask is None):
        shape=array.shape
        if mask is None:
            x=fftshift(fftfreq(shape[1]))
            y=fftshift(fftfreq(shape[0]))
            x,y=cp.meshgrid(x,y, indexing="xy")
            r=x**2+y**2
            max_r=0.5*cutoff ## maximum freq for fftfreq is 0.5 when the px size is not specified
            mask=(r<=max_r**2).astype(default_float)
            if rolloff!=0:
                r0=(0.5*(cutoff-rolloff))**2
                mask[r>r0]*=0.5*(1+cp.cos(3.141592654*(r-r0)/(max_r**2-r0)))[r>r0]
            del x,y, r
        arrayff=shift_fft2(array, axes=(0,1))
        if len(shape)==4:
            arrayff=arrayff*mask[:,:, None, None]
        if len(shape)==3:
            arrayff=arrayff*mask[:,:, None]
        if len(shape)==2:
            arrayff=arrayff*mask
        arrayff=ifft2_ishift(arrayff, axes=(0,1))
        del mask
        return arrayff
    else:
        return array
        
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




def compute_slow_axis_constraint_on_grid(something, scan_size, tv_reg_weight, a_coeff):
    something_scan_size = 1*something.reshape(scan_size[0], scan_size[1],something.shape[-1])
    grad=cp.zeros_like(something_scan_size)
    something_fast_average=cp.mean(something_scan_size, axis=1)
    something_fast_average_p1=cp.roll(something_fast_average, 1, axis=0)
    something_fast_average_m1=cp.roll(something_fast_average, -1, axis=0)
    center_expected=(something_fast_average_p1+something_fast_average_m1)*0.5
    r_max_term_inside=something_fast_average_m1-center_expected
    r_actual_term_inside=something_fast_average-center_expected
    r_max=a_coeff*cp.sum(r_max_term_inside**2, -1)**0.5
    r_actual=cp.sum(r_actual_term_inside**2, -1)**0.5
    reg_vector=r_actual-r_max
    reg_vector[reg_vector<0]=0
    reg_vector[0]=0
    reg_vector[-1]=0
    reg_vector_2=reg_vector**2
    reg_term=tv_reg_weight*cp.sum(reg_vector_2)
    dloss_dReg=2*tv_reg_weight*reg_vector
    r_max[r_max==0]=1
    r_actual[r_actual==0]=1
    dLoss_dr_max_term_inside    =-1*(a_coeff**2)*dloss_dReg[:,None]*r_max_term_inside/r_max[:,None]
    dLoss_dr_actual_term_inside =dloss_dReg[:,None]*r_actual_term_inside/r_actual[:,None]
    dLoss_something_fast_average=dLoss_dr_actual_term_inside
    dLoss_center_expeced=-1*dLoss_dr_max_term_inside-1*dLoss_dr_actual_term_inside
    dLoss_dsomething_fast_average_m1=dLoss_dr_max_term_inside+ 0.5*dLoss_center_expeced
    dLoss_dsomething_fast_average_p1=0.5*dLoss_center_expeced
    dLoss_something_fast_average+=cp.roll(dLoss_dsomething_fast_average_p1, -1, axis=0)
    dLoss_something_fast_average+=cp.roll(dLoss_dsomething_fast_average_m1, 1,  axis=0)
    dLoss_something_fast_average/=scan_size[1]
    grad+=dLoss_something_fast_average[:,None,:]
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
    

#-------------fft.py------------------------------------
### 2D FFT
def fftfreq(length, sampling=1):
    return sf.fftfreq(length, sampling)
def ifftfreq(length, sampling=1):
    return sf.ifftfreq(length, sampling)
def fftshift(array):
    return sf.fftshift(array)
def ifftshift(array):
    return sf.ifftshift(array)
def shift_fft2(arr, axes=(0,1), overwrite_x=False):
    y=sf.fftshift(sf.fft2(arr, axes=axes, overwrite_x=overwrite_x), axes=axes)
    return y
def fft2(arr, axes=(0,1), overwrite_x=False):
    return sf.fft2(arr, axes=axes, overwrite_x=overwrite_x)
def ifft2(arr, axes=(0,1), overwrite_x=False):
    '''ifft2'''
    return sf.ifft2(arr, axes=axes, overwrite_x=overwrite_x)
def ifft2_ishift(arr, axes=(0,1), overwrite_x=False):
    return sf.ifft2(sf.ifftshift(arr, axes=axes), axes=axes, overwrite_x=overwrite_x)
### 3D FFT
def shift_fftn(arr, axes=(0,1,2)):
    y=sf.fftshift(sf.fftn(arr, axes=axes), axes=axes)
    return y
def ifftn_ishift(arr, axes=(0,1,2)):
    y=sf.ifftn(sf.ifftshift(arr, axes=axes), axes=axes)
    return y
def ifftn(arr, axes=(0,1,2)):
    y=sf.ifftn(arr, axes=axes)
    return y
def fftn(arr, axes=(0,1,2)):
    y=sf.fftn(arr, axes=axes)
    return y

################## ################## ################## ################## loop.py ################## ################## ################## ##################



def save_updated_arrays(output_folder, epoch,current_probe_step, current_probe_pos_step, current_tilts_step,current_obj_step, obj, probe, tilts_correction, full_pos_correction, positions, tilts, static_background, current_aberrations_array_step, current_static_background_step,count, current_loss, current_sse, aberrations, beam_current, current_beam_current_step, save_flag, save_loss_log, xp):
    if save_loss_log:
        if epoch%save_loss_log==0:
            with open(output_folder+"loss.csv", mode='a', newline='') as loss_list:
                fieldnames=["epoch", "loss", "sse"]
                write_loss=csv.DictWriter(loss_list,fieldnames=fieldnames)
                write_loss.writerow({"epoch": epoch, "loss": current_loss, "sse": current_sse})
    if save_flag:  ##last update in epoch
        if xp==np:
            o=obj
            p=probe
            pos=positions+full_pos_correction
            t=tilts+tilts_correction
            s=static_background
            a=aberrations
            if not(beam_current is None):
                bc=beam_current
        else:
            o=obj.get()
            p=probe.get()
            pos=(positions+full_pos_correction).get()
            t=(tilts+tilts_correction).get()
            if type(static_background)==xp.ndarray:
                s=static_background.get()
            a=aberrations.get()
            if not(beam_current is None):
                bc=beam_current.get()
        if current_beam_current_step and not(beam_current is None):
            np.save(output_folder+"checkpoint_beam_current_epoch_"+str(epoch)+".npy", bc)
        if current_obj_step:
            np.save(output_folder+"checkpoint_obj_epoch_"+str(epoch)+".npy", o)
        if current_probe_step:
            np.save(output_folder+"checkpoint_probe_epoch_"+str(epoch)+".npy", p)
        if current_probe_pos_step:
            np.save(output_folder+"checkpoint_positions_epoch_"+str(epoch)+".npy", pos)
        if current_tilts_step:
            np.save(output_folder+"checkpoint_tilts_epoch_"+str(epoch)+".npy", t)
        if current_static_background_step and type(static_background)==np.ndarray:
            np.save(output_folder+"checkpoint_stback_epoch_"+str(epoch)+".npy", s)
        if current_aberrations_array_step:
            np.save(output_folder+"checkpoint_aberrations_epcoh_"+str(epoch)+".npy", a)

def create_static_background_from_nothing(static_background, probe, damping_cutoff_multislice, data_pad, upsample_pattern, default_float, recon_type):
    if type(static_background)!=np.ndarray:
        if static_background>1e-7:
            rescale=np.copy(static_background)
            shape0st,shape1st =(probe.shape[0]-2*data_pad)//upsample_pattern, (probe.shape[1]-2*data_pad)//upsample_pattern
            static_background=0.1+0.9*np.random.rand(shape0st, shape1st)
            static_background*=rescale/np.sum(static_background)
            r=np.sum(np.array(np.meshgrid(np.fft.fftshift(np.fft.fftfreq(shape0st)),np.fft.fftshift(np.fft.fftfreq(shape1st)), indexing="xy"))**2, axis=0)**0.5
            static_background[r>damping_cutoff_multislice*0.5]=0
            static_background=np.pad(static_background, data_pad//upsample_pattern)
            static_background=static_background.astype(default_float)
    return static_background
    
def create_probe_from_nothing(probe, data_pad, mean_pattern, aperture_mask, tilt_mode, tilts, dataset, estimate_aperture_based_on_binary, pixel_size_x_A, acc_voltage, data_multiplier, masks, params, data_shift_vector, data_bin, upsample_pattern, default_complex_cpu, print_flag, algorithm, measured_data_shape, n_obj_modes, probe_marker, recon_type, defocus_array, Cs):
    if type(probe)!=np.ndarray:
        if probe is None or probe=="aperture":
            if probe=="aperture":
                if data_pad>0:
                    mean_pattern=aperture_mask[data_pad:-data_pad, data_pad:-data_pad]
                else:
                    mean_pattern=aperture_mask
            else:
                if tilt_mode and not("compressed" in algorithm):
                    sub_tilts=np.copy(tilts[:1000,:])
                    sub_tilts=sub_tilts[:,4:]-sub_tilts[:,:2]
                    sub_data=np.array(dataset[:1000,:,:]).astype(default_complex_cpu)
                    if estimate_aperture_based_on_binary:
                        mean_sub=np.mean(sub_data)
                        sub_data=sub_data>mean_sub*estimate_aperture_based_on_binary
                    sub_data=np.fft.fftshift(np.fft.fft2(sub_data, axes=(1,2)), axes=(1,2))
                    stx,sty=np.meshgrid(np.fft.fftshift(np.fft.fftfreq(dataset.shape[2])), np.fft.fftshift(np.fft.fftfreq(measured_data_shape[1])), indexing="xy")
                    rsc=(data_pad*2+measured_data_shape[2])*pixel_size_x_A*np.sqrt(acc_voltage*(acc_voltage+2*511))/12.4
                    tiltx=sub_tilts[:,1]*rsc
                    tilty=sub_tilts[:,0]*rsc
                    subkernel=np.exp(2j*np.pi*(stx[None,:,:]*tiltx[:,None,None]+sty[None,:,:]*tilty[:,None,None]))
                    sub_data=sub_data*subkernel
                    mean_pattern=np.mean(np.abs(np.fft.ifft2(np.fft.ifftshift(sub_data, axes=(1,2)), axes=(1,2))), axis=0)
                    if estimate_aperture_based_on_binary:
                        mean_pattern_mean=np.mean(mean_pattern)
                        mean_pattern[mean_pattern<mean_pattern_mean*estimate_aperture_based_on_binary]=0
                    del sub_data, subkernel, stx,sty
                else:
                    if mean_pattern is None:
                        mean_pattern=np.mean(dataset[:1000], axis=0)*data_multiplier
                    else:
                        mean_pattern*=data_multiplier
            if not(masks is None):
                mean_pattern=np.sum(masks*mean_pattern[:,None, None], axis=0)
            # Shift, bin, pad, rescale!
            if data_shift_vector[0]!=0:
                mean_pattern=np.roll(mean_pattern, data_shift_vector[0], axis=0)
                if data_shift_vector[0]>0:
                    mean_pattern[-data_shift_vector[0]:,:]=0
                else:
                    mean_pattern[:-data_shift_vector[0],:]=0
            if data_shift_vector[1]!=0:
                mean_pattern=np.roll(mean_pattern, data_shift_vector[1], axis=1)
                if data_shift_vector[1]>0:
                    mean_pattern[:,-data_shift_vector[1]:]=0
                else:
                    mean_pattern[:, :-data_shift_vector[1]]=0
            if data_bin>1:
                mean_pattern2=np.zeros((mean_pattern.shape[0]//data_bin, mean_pattern.shape[1]//data_bin))
                for i1 in range(data_bin):
                    for j1 in range(data_bin):
                        mean_pattern2+=mean_pattern[i1::data_bin, i2::data_bin]
                mean_pattern=mean_pattern2
                del mean_pattern2
            if upsample_pattern!=1:
                x,y=np.meshgrid(np.linspace(0,1,mean_pattern.shape[1]),np.linspace(0,1,mean_pattern.shape[0]))
                points=np.swapaxes(np.array([x.flatten(),y.flatten()]), 0,1)
                x2, y2=np.meshgrid(np.linspace(0,1,upsample_pattern*mean_pattern.shape[1]), np.linspace(0,1,upsample_pattern*mean_pattern.shape[0]))
                mean_pattern=np.abs(griddata(points, mean_pattern.flatten(), (x2, y2), method='cubic'))
            mean_pattern=np.pad(np.abs(mean_pattern),data_pad, mode="constant", constant_values=0)
            if print_flag!=0:
                sys.stdout.write("\nThe probe was generated based on the mean pattern!")
                sys.stdout.flush()
            if recon_type=="near_field":
                probe=np.expand_dims(np.sqrt(mean_pattern),-1).astype(default_complex_cpu)
            else:
                probe=np.expand_dims(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(np.sqrt(mean_pattern)))),-1).astype(default_complex_cpu)
    if not("compressed" in algorithm):
        if recon_type=="far_field":
            probe=padprobetodatafarfield(probe, measured_data_shape, data_pad, upsample_pattern)
        else:
            probe=padprobetodatanearfield(probe, measured_data_shape, data_pad, upsample_pattern)
        
    if recon_type=="far_field":
        probe_counts_must = np.sum(dataset[:1000], axis=(0,1,2))/((dataset[:1000]).shape[0] * probe.shape[0] * probe.shape[1])
    else:
        probe_counts_must = np.sum(dataset[:1000], axis=(0,1,2))/((dataset[:1000]).shape[0])
    if len(probe.shape)==3:
        probe_counts=np.sum(np.abs(probe)**2)/probe.shape[2]
        rescale=np.sqrt(np.abs(probe_counts_must/probe_counts))
        probe*=rescale
    else:
        for indsc in range(probe.shape[3]):
            probe_counts=np.sum(np.abs(probe[:,:,:,indsc])**2)
            rescale=np.sqrt(probe_counts_must/probe_counts)
            probe[:,:,:,indsc]=probe[:,:,:,indsc]*rescale
    if not(probe_marker is None) and len(probe.shape)==3:
        probe_scenarios=np.max(probe_marker)+1
        probe=np.tile(np.expand_dims(probe,-1), probe_scenarios)
    sys.stdout.flush()
    return probe.astype(default_complex_cpu)

def generate_hermite_modes(main_mode, n_herm_x,n_herm_y, default_complex, xp):
    probe_int_px=xp.abs(main_mode)**2
    probe_int_tot=xp.sum(probe_int_px)
    X,Y=xp.meshgrid(xp.linspace(-1,1,main_mode.shape[1], endpoint=True), xp.linspace(-1,1,main_mode.shape[0], endpoint=True))
    x_cent = xp.average(X , weights=probe_int_px)
    y_cent = xp.average(Y , weights=probe_int_px)
    x_var =  xp.average((X-x_cent)**2 , weights=probe_int_px)
    y_var = xp.average((Y-y_cent)**2 , weights=probe_int_px)
    all_modes = (main_mode[:,:,None]) / xp.sqrt(xp.sum(xp.abs(main_mode)**2))
    for yi in range(n_herm_y + 1):
        for xi in range(n_herm_x + 1):
            if yi!= 0 or xi!=0:
                temp = main_mode*  (X - x_cent)**xi * (Y - y_cent)**yi  * xp.exp( -(X - x_cent)**2/(2*x_var) - (Y - y_cent)**2/(2*y_var))
                temp = temp / xp.sqrt(xp.sum(xp.abs(temp)**2))
                all_modes=xp.concatenate((all_modes,temp[:,:,None]), axis=2)
    modes_int=xp.sum(xp.abs(all_modes)**2)
    return (all_modes* xp.sqrt(probe_int_tot/modes_int)).astype(default_complex)
    
def apply_probe_modulation(probe, extra_probe_defocus, acc_voltage, pixel_size_x_A, pixel_size_y_A, aberrations, print_flag, beam_ctf, n_hermite_probe_modes, defocus_spread_modes, probe_marker, default_complex, default_float, xp):
    if extra_probe_defocus!=0: probe=apply_defocus_probe(probe, extra_probe_defocus,acc_voltage, pixel_size_x_A, pixel_size_y_A, default_complex, default_float, xp);
    if not(aberrations is None):
        num_abs=len(aberrations)
        possible_n, possible_m, possible_ab=convert_num_to_nmab(num_abs)
        aber_print, s=nmab_to_strings(possible_n, possible_m, possible_ab), ""
        for i in range(len(aberrations)): s+=aber_print[i]+": %.2e Å; "%aberrations[i];
        if print_flag:
            sys.stdout.write("\nProvided aberrations: %s\n"%s[:-1])
        wavelength=12.4/((2*511.0+acc_voltage)*acc_voltage)**0.5
        kx=np.fft.fftshift(np.fft.fftfreq(probe.shape[1], pixel_size_x_A))*wavelength
        ky=np.fft.fftshift(np.fft.fftfreq(probe.shape[0], pixel_size_y_A))*wavelength
        kx, ky=np.meshgrid(kx,ky, indexing="xy")
        ctf = xp.asarray(get_ctf(aberrations, kx, ky, wavelength))
        if len(probe.shape)==3:
            probe=ifft2_ishift(shift_fft2(probe)*xp.exp(-1j*ctf)[:,:,None])
        else:
            probe=ifft2_ishift(shift_fft2(probe)*xp.exp(-1j*ctf)[:,:,None, None])
    if not(beam_ctf is None):
        beam_ctf= xp.asarray(beam_ctf)
        if beam_ctf.shape[0]!=probe.shape[0]:
            beam_ctf=np.pad(beam_ctf, data_pad)
        if len(probe.shape)==3:
            probe=ifft2_ishift(shift_fft2(probe)*xp.exp(-1j*beam_ctf)[:,:,None])
        else:
            probe=ifft2_ishift(shift_fft2(probe)*xp.exp(-1j*beam_ctf)[:,:,None,None])
        if print_flag:
            sys.stdout.write("\nUsing the provided CTF for the beam initial guess!\n")
    if not(defocus_spread_modes is None):
        p_final=cp.zeros((probe.shape[0], probe.shape[1], len(defocus_spread_modes)), dtype=default_complex)
        for inddef,defocus in enumerate(defocus_spread_modes):
            p_final[:,:,inddef]=apply_defocus_probe(probe[:,:,:1], defocus, acc_voltage, pixel_size_x_A, pixel_size_y_A, default_complex, default_float, xp)[:,:,0]
            probe=p_final
    if not(n_hermite_probe_modes is None):
        n_herm_y, n_herm_x=n_hermite_probe_modes
        if print_flag:
            sys.stdout.write("\nGenerating probe modes based on Hermite polynomials! n_x: %d, n_y: %d\n"%(n_herm_x, n_herm_y))
        if probe_marker is None:
            probe=generate_hermite_modes(probe[:,:,0], n_herm_x,n_herm_y, default_complex, xp)
        else:
            p_temp=generate_hermite_modes(probe[:,:,0,i_sc], n_herm_x,n_herm_y, default_complex, xp)[:,:,:,None]
            for i_sc in range(1, probe.shape[3]):
                modes=generate_hermite_modes(probe[:,:,0,i_sc], n_herm_x,n_herm_y, default_complex, xp)
                p_temp=xp.concatenate((p_temp,modes[:,:,:,None]), axis=3)
            probe=p_temp
            del p_temp
    return probe.astype(default_complex)


def prepare_main_loop_params(algorithm,probe, obj,positions,tilts, measured_data_shape, acc_voltage,allow_subPixel_shift=True, sequence=None, use_full_FOV=False, print_flag=0, default_float_cpu=np.float64, default_complex_cpu=np.complex128, default_int_cpu=np.int64, probe_constraint_mask=None, aperture_mask=None, extra_space_on_side_px=0):
    wavelength  = default_float_cpu(12.398 / np.sqrt((2 * 511.0 + acc_voltage) * acc_voltage))  # angstrom
    if sequence is None:
        sequence=list(np.arange(0, measured_data_shape[0],1))
    try:
        full_sequence=sequence(0)
    except:
        full_sequence=sequence
    if use_full_FOV:
        positions[:,0]-=np.min(positions[:,0])
        positions[:,1]-=np.min(positions[:,1])
    else:
        positions[full_sequence,0]-=np.min(positions[full_sequence,0])
        positions[full_sequence,1]-=np.min(positions[full_sequence,1])
    positions+=extra_space_on_side_px
    if extra_space_on_side_px>0:
        obj=np.vstack((np.max(np.abs(obj))*np.ones((extra_space_on_side_px,obj.shape[1], obj.shape[2],obj.shape[3]), dtype=default_complex_cpu), obj ))
        obj=np.hstack((np.max(np.abs(obj))*np.ones((obj.shape[0],extra_space_on_side_px, obj.shape[2],obj.shape[3]), dtype=default_complex_cpu), obj))
    tilts_correction=np.zeros_like(tilts)
    if allow_subPixel_shift:
        full_pos_correction=(positions-np.round(positions).astype(default_int_cpu)).astype(default_float_cpu)
    else:
        full_pos_correction=np.zeros_like(positions).astype(default_float_cpu)
    positions=np.round(positions).astype(default_int_cpu)
    if use_full_FOV:
        diff_x=np.max(positions[:,1])+extra_space_on_side_px+probe.shape[1]-obj.shape[1]
    else:
        diff_x=np.max(positions[full_sequence,1])+extra_space_on_side_px+probe.shape[1]-obj.shape[1]
    if diff_x>0:
        obj=np.hstack((obj, np.max(np.abs(obj))*np.ones((obj.shape[0],int(diff_x), obj.shape[2],obj.shape[3]), dtype=default_complex_cpu)))
    if use_full_FOV:
        diff_y=np.max(positions[:,0])+extra_space_on_side_px+probe.shape[0]-obj.shape[0]
    else:
        diff_y=np.max(positions[full_sequence,0])+extra_space_on_side_px+probe.shape[0]-obj.shape[0]
    if diff_y>0:
        obj=np.vstack((obj, np.max(np.abs(obj))*np.ones((int(diff_y),obj.shape[1], obj.shape[2],obj.shape[3]), dtype=default_complex_cpu)))
    if (diff_x>0 or diff_y>0) and print_flag>=1:
        sys.stdout.write("\nWARNING! Range of specified scan positions was larger than the shape of the object, adding ones to the right and/or bottom. New shape of the object is %d px in y, %d px in x, %d slice(-s) , %d mode(-s) !!!!!\n" %(obj.shape[0], obj.shape[1], obj.shape[2], obj.shape[3]))
        sys.stdout.flush()
    if not(probe_constraint_mask is None):
        aperture_mask=probe_constraint_mask
    return obj, positions, 0, sequence, wavelength, full_pos_correction, tilts_correction, aperture_mask
def prepare_saving_stuff(output_folder, save_loss_log):
    try:
        os.makedirs(output_folder, exist_ok=True)
    except:
        pass
    try:
        os.remove(output_folder+"params.pkl")
    except:
        pass
    if save_loss_log:
        os.system("touch "+output_folder+"loss.csv")
        with open(output_folder+"loss.csv", 'w+', newline='') as loss_list:
            fieldnames=["epoch", "loss", "sse"]
            write_loss=csv.DictWriter(loss_list,fieldnames=fieldnames)
            write_loss.writeheader()

        
def print_pypty_header(data_path, output_folder, save_loss_log):
    sys.stdout.write("\n***************************************************** *************************\n************************ Starting PyPty Reconstruction ***********************\n******************************************************************************\n")
    sys.stdout.write("\nPath to the dataset: %s"%data_path)
    sys.stdout.write("\nSaving the results in %s" %output_folder)
    sys.stdout.write("\nSaving the parameters in %s" %output_folder+"params.pkl")
    if save_loss_log:
        sys.stdout.write("\nThe log file will be saved as %s" %output_folder+"loss.csv")
    sys.stdout.write("\n******************************************************************************\n******************************************************************************\n******************************************************************************")
    sys.stdout.flush()

        

def save_current_checkpoint_obj_probe(output_folder, obj, probe, tilts_correction, full_pos_correction, positions, tilts, static_background, current_probe_step, current_obj_step, current_probe_pos_step, current_tilts_step, current_static_background_step, current_aberrations_array_step, aberrations_array, beam_current, bcstep, xp):
    if current_obj_step:
        o=obj if xp==np else obj.get();
        np.save(output_folder+"co.npy", o)   ## checkpoint_obj
        del o
    if bcstep and not(beam_current is None):
        bc=beam_current if xp==np else beam_current.get();
        np.save(output_folder+"cb.npy", bc)
        del bc
    if current_probe_step:
        p=probe if xp==np else probe.get();
        np.save(output_folder+"cp.npy", p)   ## checkpoint_probe
        del p
    if current_tilts_step:
        t=tilts+tilts_correction if xp==np else (tilts+tilts_correction).get()
        np.save(output_folder+"ct.npy", t)   ## checkpoint_tilts
        del t
    if current_probe_pos_step:
        pos=positions+full_pos_correction if xp==np else (positions+full_pos_correction).get()
        np.save(output_folder+"cg.npy", pos) ## checkpoint_grid (postions)
        del pos
    if current_static_background_step and type(static_background)==xp.ndarray:
        s=static_background if xp==np else static_background.get()
        np.save(output_folder+"cs.npy", s) ## checkpoint_static_background
        del s
    if current_aberrations_array_step:
        a=aberrations_array if xp==np else aberrations_array.get()
        np.save(output_folder+"ca.npy", a)
        
def print_recon_state(t0, algorithm, epoch,
                         current_loss, current_sse,
                        current_obj_step, current_probe_step,current_probe_pos_step,current_tilts_step,
                        current_static_background_step,current_aberrations_array_step,
                        current_beam_current_step, current_hist_length,  print_flag):
    if print_flag>0:
        t1=time.time()-t0
        hours=t1//3600
        minutes=(t1-3600*hours)//60
        seconds=t1-3600*hours-60*minutes
        string="Updating "
        if current_obj_step: string+="object, ";
        if current_probe_step: string+="probe, ";
        if current_probe_pos_step: string+="scan grid, ";
        if current_tilts_step: string+="tilts, ";
        if current_static_background_step: string+="static background, ";
        if current_aberrations_array_step: string+="aberrations, ";
        if current_beam_current_step: string+="beam current, ";
        string=string[:-2]+";"
        print_optimizer="bfgs-%d"%current_hist_length
        if current_hist_length==0:
            print_optimizer="gradient descent"
        if current_hist_length==1:
            print_optimizer="conjugate gradient"
        if print_flag<2:
            sys.stdout.write("\r---------> Time: %d:%d:%d. Epoch %i. Using %s error metric with %s optimzer. Loss: %.2e. SSE: %.2e. %s" % (hours, minutes, seconds,  epoch, algorithm, print_optimizer, current_loss, current_sse, string))
        else:
            sys.stdout.write("\n---------> Time: %d:%d:%d. Epoch %i. Using %s error metric with %s optimzer. Loss: %.2e. SSE: %.2e. %s" % (hours, minutes, seconds,  epoch, algorithm, print_optimizer, current_loss, current_sse, string))
        sys.stdout.flush()
        
def try_to_gpu(obj, probe, positions,full_pos_correction, tilts, tilts_correction, masks, defocus_array, slice_distances, aperture_mask, dataset, load_one_by_one, static_background, aberrations_array, beam_current, default_float, default_complex, default_int, xp):
    obj=xp.asarray(obj).astype(default_complex)
    probe=xp.asarray(probe).astype(default_complex)
    positions=xp.asarray(positions).astype(default_int)
    full_pos_correction=xp.asarray(full_pos_correction).astype(default_float)
    tilts=xp.asarray(tilts).astype(default_float)
    tilts_correction=xp.asarray(tilts_correction).astype(default_float)
    defocus_array=xp.asarray(defocus_array).astype(default_float)
    slice_distances=xp.asarray(slice_distances).astype(default_float)
    if type(static_background)==np.ndarray:
        static_background=xp.asarray(static_background).astype(default_float)
    beam_current=xp.asarray(beam_current).astype(default_float) if not(beam_current is None) else None
    if not(aberrations_array is None):
        aberrations_array=xp.asarray(aberrations_array).astype(default_float)
    if not(masks is None):
        masks=xp.asarray(masks).astype(default_float)
    if not(aperture_mask is None) and type(aperture_mask)==np.ndarray:
        aperture_mask=xp.asarray(aperture_mask).astype(default_float)
    if not(load_one_by_one):
        dataset=xp.asarray(dataset)
    return obj, probe, positions,full_pos_correction, tilts, tilts_correction, masks, defocus_array, slice_distances, aperture_mask, dataset, static_background, aberrations_array, beam_current
    
def try_to_initialize_beam_current(beam_current,measured_data_shape,default_float, xp):
    if beam_current is None:
        beam_current=beam_current=xp.ones(measured_data_shape[0], dtype=default_float)
    if beam_current.shape[0]<measured_data_shape[0]: beam_current=xp.pad(beam_current, pad_width=measured_data_shape[0]-beam_current.shape[0], mode='constant', constant_values=1)
    return beam_current

def get_value_for_epoch(func_or_value, epoch, default_float):
    out=[]
    for f in func_or_value:
        try:
            x=f(epoch)
        except:
            x=f
        out.append(x)
    return out
    
def get_steps_epoch(steps, epoch, default_float):
    out=[]
    for s in steps:
        try:
            curr_s=s[0]*s[1](epoch)
        except:
            curr_s=s
        out.append(default_float(curr_s))
    return out


def lambda_to_string(f):
    if isinstance(f, types.LambdaType):
        string=inspect.getsourcelines(f)[0][0]
        first=string.find("lambda")
        if "]" in string:
            last=string.find("]")
        else:
            last=string.find(",")
        return string[first:last]
    else:
        return f
        
def convert_to_string(dicti2):
    for key, value in dicti2.items():
        if isinstance(value, types.LambdaType):
            dicti2[key] = lambda_to_string(value)
        elif isinstance(value, list):
            dicti2[key] = [lambda_to_string(item) for item in value]
        elif isinstance(value, dict):
            dicti2[key] = {key2: lambda_to_string(item) for key2, item in value.items()}
    return dicti2
    
def string_to_lambda(lambda_string):
    try:
        return eval(lambda_string)
    except:
        return lambda_string

def load_params(path):
    with open(path, 'rb') as handle:
        params = pickle.load(handle)
    return params
    
def string_params_to_usefull_params(params):
    for key in params.keys():
        item=params[key]
        if type(item)==str and 'lambda' in item:
            item=string_to_lambda(item)
        if type(item)==list:
            if len(item)==2:
                for i in range(2):
                    item[i]=string_to_lambda(item[i])
        params[key]=item
    return params

def save_params(params_path, params):
    if params_path[-4:]==".pkl":
        try:
            os.remove(params_path)
        except:
            pass
    params_pkl=convert_to_string(params)
    with open(params_path, 'wb') as file:
        pickle.dump(params_pkl, file)
    del params_pkl
def reset_bfgs_history():
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
        
history_bfgs=None
obj,probe, positions, positions_correction, tilts, tilts_correction, defocus_array, beam_current = None, None, None, None, None, None,None, None
try:
    pool=cp.get_default_memory_pool()
    pinned_pool=cp.get_default_pinned_memory_pool()
except:
    pool, pinned_pool=None, None


def run_ptychography(pypty_params):
    global obj, probe, pool, pinned_pool, positions, positions_correction, tilts, tilts_correction, defocus_array, beam_current, history_bfgs
    params=pypty_params.copy()
    obj,probe, positions, positions_correction, tilts, tilts_correction, defocus_array, beam_current = None, None, None, None, None, None,None, None
    reset_bfgs_history()
    try:
        pool=cp.get_default_memory_pool()
        pinned_pool=cp.get_default_pinned_memory_pool()
    except:
        pass
    if type(params)==str:
        params=load_params(params)
    xp =  params.get('backend', cp) ## currently not used, but usefull for future:
    default_dtype=params.get('default_dtype', "double")
    if default_dtype=="double":
        default_float, default_complex, default_int, default_int_cpu, default_float_cpu, default_complex_cpu=xp.float64, xp.complex128, xp.int64, np.int64, np.float64, np.complex128
    if default_dtype=="single":
        default_float, default_complex, default_int, default_int_cpu, default_float_cpu, default_complex_cpu=xp.float32, xp.complex64, xp.int32, np.int32, np.float32, np.complex64
    if default_dtype=="half":
        default_float, default_complex, default_int, default_int_cpu, default_float_cpu, default_complex_cpu=xp.float16, xp.complex64, xp.int16, np.int16, np.float16, np.complex64
    ## Dataset
    data_path = params.get('data_path', "")
    masks = params.get('masks', None)
    data_multiplier = default_float_cpu(params.get('data_multiplier', 1))
    data_pad = int(params.get('data_pad', 0))
    data_bin=int(params.get('data_bin', 1))
    data_shift_vector=params.get('data_shift_vector', [0,0])
    upsample_pattern=params.get('upsample_pattern',1)
    sequence = params.get('sequence', None)
    use_full_FOV = params.get('use_full_FOV', True)
    ## Saving and printing
    output_folder = params.get('output_folder', "")
    save_loss_log = params.get('save_loss_log', True)
    prepare_saving_stuff(output_folder, save_loss_log)
    if output_folder[-1]!="/": output_folder+="/";
    save_params(output_folder+"params.pkl", params) ### save the params
    print_pypty_header(data_path, output_folder, save_loss_log)
    params=string_params_to_usefull_params(params) ### here we want to convert some possible strings that may look like 'lambda x: x>1' into real functions
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
    ### ptycho stuff
    num_slices=params.get('num_slices', 1)
    obj = params.get('obj', np.ones((1, 1, num_slices, 1))).astype(default_complex_cpu)
    probe = params.get('probe', None)
    positions = params.get('positions', np.array([[0.0, 0.0]])).astype(default_float_cpu)
    tilts = params.get('tilts', np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])).astype(default_float_cpu) ### tilts are the tilt before specimen (slope in real space), tilts in the specimen (slope in reciprocal space) and tilts after the specimen (slope in real space)
    tilt_mode=params.get('tilt_mode', 0) ## tilt mode 0: tilt only inside of the specimen, 1 tilt only after the specimen; 2 for tilt before and after the specimen;  3 tilt inside and after the specimen; 4 is for tilting before, inside and after;
    static_background=params.get('static_background', 0)
    beam_current=params.get('beam_current', None)
    ## spatial callibration of the object
    slice_distances = params.get('slice_distances', np.array([10]))
    pixel_size_x_A = default_float_cpu(params.get('pixel_size_x_A', 1))
    pixel_size_y_A = default_float_cpu(params.get('pixel_size_y_A', 1))
    scan_size= params.get('scan_size', None)
    
    ## Propagation method, windowing and dinamic resizing
    propmethod = params.get('propmethod', "multislice")
    allow_subPixel_shift = params.get('allow_subPixel_shift', True)
    dynamically_resize_yx_object=params.get('dynamically_resize_yx_object', False)
    extra_space_on_side_px= int(params.get('extra_space_on_side_px', 0))
    
    ## Bandwidth limitation
    damping_cutoff_multislice = default_float_cpu(params.get('damping_cutoff_multislice', 2/3))
    smooth_rolloff=default_float_cpu(params.get('smooth_rolloff', 0))
    update_extra_cut=default_float_cpu(params.get('update_extra_cut', 0.005))
    lazy_clean=params.get('lazy_clean', False)
    
    ## optimization settings
    algorithm = params.get('algorithm', "lsq_sqrt")
    update_batch = params.get('update_batch', "full")
    epoch_max = int(params.get('epoch_max', 200))
    epoch_prev = int(params.get('epoch_prev', 0))
    randomize = params.get('randomize', True)
    wolfe_c1_constant = params.get('wolfe_c1_constant', 0.5)
    wolfe_c2_constant=params.get('wolfe_c2_constant', 0.999999)
    loss_weight = params.get('loss_weight', 1)
    max_count = params.get('max_count', None)
    reduce_factor = default_float_cpu(params.get('reduce_factor', 0.5))
    optimism = params.get('optimism', 2)
    min_step = params.get('min_step', 1e-20)
    
    hist_length=params.get('hist_length', 10)
    update_step_bfgs=params.get('update_step_bfgs', 1)

    reset_history_flag=params.get('reset_history_flag', None)
    
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
    load_one_by_one= params.get('load_one_by_one', True)
    smart_memory = params.get('smart_memory', True)
    remove_fft_cache = params.get('remove_fft_cache', False)
    compute_batch=params.get('compute_batch', 1)
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
    fast_axis_reg_weight_positions = params.get('fast_axis_reg_weight_positions', 0)
    fast_axis_reg_weight_tilts = params.get('fast_axis_reg_weight_tilts', 0)
    slow_axis_reg_weight_positions= params.get('slow_axis_reg_weight_positions', 0)
    slow_axis_reg_coeff_positions= params.get('slow_axis_reg_coeff_positions', 0)
    slow_axis_reg_weight_tilts= params.get('slow_axis_reg_weight_tilts', 0)
    slow_axis_reg_coeff_titls= params.get('slow_axis_reg_coeff_titls', 0)
    
    # Constraints that modify the object and probe 'by hand'
    apply_gaussian_filter=params.get('apply_gaussian_filter', False)
    apply_gaussian_filter_amplitude=params.get('apply_gaussian_filter_amplitude', False)
    
    phase_only_obj = params.get('phase_only_obj', False)
    tune_only_probe_phase = params.get('tune_only_probe_phase', False)
    tune_only_probe_abs=params.get('tune_only_probe_abs', False)
    beta_wedge = params.get('beta_wedge', 0) ## to do for bfgs!!!
    keep_probe_states_orthogonal = params.get('keep_probe_states_orthogonal', False) ## to do for bfgs!!!
    
    do_charge_flip = params.get('do_charge_flip', False)
    cf_delta_phase = params.get('cf_delta_phase', 0.1)
    cf_delta_abs = params.get('cf_delta_abs', 0.01)
    cf_beta_phase = params.get('cf_beta_phase', -0.95)
    cf_beta_abs = params.get('cf_beta_abs', -0.95)
    fancy_sigma=params.get('fancy_sigma', None)
    ### Messing with the reconstruciton
    restart_from_vacuum=params.get('restart_from_vacuum', [])
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
    ############################### done with params, starts other things ###################################
    ### get the data
    if data_path[-2:]=="h5":
        this_file=h5py.File(data_path, "r")
        dataset=this_file['data']
        if preload_to_cpu:
            dataset=np.array(dataset).astype(force_dataset_dtype)
    else:
        dataset=np.load(data_path).astype(force_dataset_dtype)
        if len(dataset.shape==4):
            dataset=dataset.reshape(dataset.shape[0]*dataset.shape[1], dataset.shape[2], dataset.shape[3])
        dataset, data_shift_vector, data_bin, data_pad, data_multiplier = preprocess_dataset(dataset, False, algorithm_type, recon_type, data_shift_vector, data_bin, data_pad, upsample_pattern, data_multiplier, np, False)
    measured_data_shape=dataset.shape
    probe=create_probe_from_nothing(probe, data_pad, mean_pattern, aperture_mask, tilt_mode, tilts, dataset, estimate_aperture_based_on_binary, pixel_size_x_A, acc_voltage, data_multiplier, masks, params, data_shift_vector, data_bin, upsample_pattern, default_complex_cpu, print_flag, algorithm, measured_data_shape, obj.shape[-1], probe_marker, recon_type, defocus_array, Cs) ### create probe from nothing
    static_background=create_static_background_from_nothing(static_background, probe, damping_cutoff_multislice,data_pad,upsample_pattern,  default_float_cpu, recon_type) ## initializing static background
    obj, positions, t, sequence, wavelength, positions_correction, tilts_correction, aperture_mask = prepare_main_loop_params(algorithm,probe, obj,positions,tilts, measured_data_shape, acc_voltage, allow_subPixel_shift, sequence, use_full_FOV, print_flag, default_float_cpu, default_complex_cpu, default_int_cpu, probe_constraint_mask, aperture_mask, extra_space_on_side_px)  # now the we will initilize the object in this function (create from nothing if needed and pad an existing one if needed)
    try:
        obj, probe, positions,positions_correction, tilts, tilts_correction, masks, defocus_array, slice_distances, aperture_mask, dataset, static_background, aberrations_array, beam_current=try_to_gpu(obj, probe, positions,positions_correction, tilts, tilts_correction, masks, defocus_array, slice_distances, aperture_mask, dataset, load_one_by_one, static_background, aberrations_array, beam_current, default_float, default_complex, default_int, xp) ##Convert numpy arrays to cupy arrays
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
        obj, probe, positions,positions_correction, tilts, tilts_correction, masks, defocus_array, slice_distances, aperture_mask, dataset, static_background, aberrations_array, beam_current=try_to_gpu(obj, probe, positions,positions_correction, tilts, tilts_correction, masks, defocus_array, slice_distances, aperture_mask, dataset, load_one_by_one, static_background, aberrations_array, beam_current, default_float, default_complex, default_int, xp) ##Convert numpy arrays to cupy arrays
    probe=apply_probe_modulation(probe, extra_probe_defocus, acc_voltage, pixel_size_x_A, pixel_size_y_A, aberrations, print_flag, beam_ctf, n_hermite_probe_modes, defocus_spread_modes, probe_marker, default_complex, default_float, xp) #Here we will apply aberrations to an existing beam and create multiple modes
    probe=fourier_clean(probe, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=xp) # clean the beam and object just to be on a safe side
    try:
        first_smart_memory=smart_memory(0)
        first_smart_memory=True
    except:
        first_smart_memory=smart_memory
    if first_smart_memory:
        for i in range(obj.shape[2]):
            for j in range(obj.shape[3]):
                obj[:,:,i,j]=fourier_clean(obj[:,:,i,j], cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=xp)
    else:
        obj = fourier_clean(obj, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=xp)
    try:
        if remove_fft_cache:
            cp.fft.config.clear_plan_cache()
        pool.free_all_blocks()
        pinned_pool.free_all_blocks()
    except:
        pass
    save_current_checkpoint_obj_probe(output_folder, obj, probe, tilts_correction, positions_correction,positions, tilts, static_background,1,1,1,1,1,1, aberrations_array, beam_current, 1, xp)
    dataset, data_shift_vector, data_bin, data_pad, data_multiplier = preprocess_dataset(dataset, load_one_by_one, algorithm, recon_type, data_shift_vector, data_bin, data_pad, upsample_pattern, data_multiplier, xp, force_pad)
    #######----------------------------------------------------------------------------------------------------------------------------------
    #######------------------------------------------ HERE is the begin of the actual ptychography ------------------------------------------
    #######----------------------------------------------------------------------------------------------------------------------------------
    t0=time.time()
    is_first_epoch=True ## Just a flag that will enable recomputing of propagation- & shift- meshgrids
    for epoch in range(epoch_prev,epoch_max,1):
        try:
            full_sequence=sequence(epoch)
        except:
            full_sequence=sequence
        try:
            this_smart_memory=smart_memory(epoch)
        except:
            this_smart_memory=smart_memory
        if randomize: full_sequence=random.sample(full_sequence, len(full_sequence));
        if not(reset_history_flag is None):
            if reset_history_flag(epoch): reset_bfgs_history();
        try:
            restart_flag=restart_from_vacuum(epoch)
        except:
            restart_flag=epoch in restart_from_vacuum
        if restart_flag:
            obj=xp.ones_like(obj);
            reset_bfgs_history()
        count, save_flag=0, False ## count for measurements
        if save_checkpoints_every_epoch: save_flag= epoch%save_checkpoints_every_epoch==0;
        current_wolfe_c1_constant,current_wolfe_c2_constant, current_window_weight, current_hist_length, current_slow_axis_reg_weight_tilts, current_slow_axis_reg_coeff_tilts, current_slow_axis_reg_weight_positions, current_slow_axis_reg_coeff_positions, current_fast_axis_reg_weight_positions,current_fast_axis_reg_weight_tilts, current_update_step_bfgs, current_apply_gaussian_filter_amplitude, current_apply_gaussian_filter, current_keep_probe_states_orthogonal, current_loss_weight, current_phase_norm_weight, current_abs_norm_weight, current_probe_reg_constraint_weight, current_do_charge_flip, current_atv_weight, current_beta_wedge, current_tune_only_probe_phase, current_mixed_variance_weight,current_mixed_variance_sigma, current_phase_only_obj, current_tune_only_probe_abs, current_dynamically_resize_yx_object, current_beam_current_step, current_probe_step, current_obj_step, current_probe_pos_step, current_tilts_step, current_static_background_step, current_aberrations_array_step =               get_value_for_epoch([wolfe_c1_constant,wolfe_c2_constant, window_weight, hist_length, slow_axis_reg_weight_tilts, slow_axis_reg_coeff_titls, slow_axis_reg_weight_positions, slow_axis_reg_coeff_positions, fast_axis_reg_weight_positions,fast_axis_reg_weight_tilts, update_step_bfgs, apply_gaussian_filter_amplitude, apply_gaussian_filter, keep_probe_states_orthogonal, loss_weight, phase_norm_weight, abs_norm_weight, probe_reg_constraint_weight, do_charge_flip, atv_weight, beta_wedge, tune_only_probe_phase, mixed_variance_weight,mixed_variance_sigma, phase_only_obj, tune_only_probe_abs, dynamically_resize_yx_object, update_beam_current, update_probe, update_obj, update_probe_pos, update_tilts, update_static_background, update_aberrations_array], epoch, default_float_cpu)
        if current_window_weight>0:
            if len(window)==2:
                this_window=get_window(probe.shape[0], window[0], window[1])
            else:
                this_window=xp.asarray(window)
        else:
            this_window=None
        if current_aberrations_array_step: beam_current=try_to_initialize_beam_current(beam_current,measured_data_shape, default_float, xp);
        
        this_chopped_sequence = this_chopped_sequence=np.sort(np.array(full_sequence))
        positions_correction,tilts_correction, aberrations_array, beam_current,static_background, current_loss, current_sse  =  bfgs_update(algorithm, slice_distances, current_probe_step, current_obj_step, current_probe_pos_step,current_tilts_step, dataset, wavelength, masks, pixel_size_x_A, pixel_size_y_A, current_phase_norm_weight, current_abs_norm_weight, min_step, current_probe_reg_constraint_weight,aperture_mask, recon_type, Cs, alpha_near_field, damping_cutoff_multislice, smooth_rolloff, update_extra_cut,  current_keep_probe_states_orthogonal, current_do_charge_flip,cf_delta_phase, cf_delta_abs, cf_beta_phase, cf_beta_abs, current_phase_only_obj, current_beta_wedge, current_wolfe_c1_constant, current_wolfe_c2_constant, current_atv_weight, atv_q, atv_p, current_tune_only_probe_phase, propmethod, this_chopped_sequence, load_one_by_one, data_multiplier,data_pad, phase_plate_in_h5, print_flag, current_loss_weight, max_count, reduce_factor, optimism, current_mixed_variance_weight, current_mixed_variance_sigma, data_bin, data_shift_vector, this_smart_memory, default_float, default_complex, default_int, upsample_pattern, static_background, current_static_background_step, tilt_mode, fancy_sigma, current_tune_only_probe_abs, aberration_marker, current_aberrations_array_step, probe_marker, aberrations_array, compute_batch, this_window, current_window_weight, current_dynamically_resize_yx_object, lazy_clean, current_apply_gaussian_filter, current_apply_gaussian_filter_amplitude, current_beam_current_step, xp, remove_fft_cache, is_first_epoch, current_hist_length, current_update_step_bfgs, current_fast_axis_reg_weight_positions, current_fast_axis_reg_weight_tilts, scan_size, current_slow_axis_reg_weight_positions, current_slow_axis_reg_coeff_positions, current_slow_axis_reg_weight_tilts, current_slow_axis_reg_coeff_tilts)
        is_first_epoch=False
        print_recon_state(t0, algorithm, epoch, current_loss, current_sse, current_obj_step, current_probe_step,current_probe_pos_step,current_tilts_step, current_static_background_step, current_aberrations_array_step, current_beam_current_step, current_hist_length, print_flag)
        if save_inter_checkpoints!=0:
            if epoch%save_inter_checkpoints==0: save_current_checkpoint_obj_probe(output_folder, obj, probe, tilts_correction, positions_correction,positions, tilts, static_background, current_probe_step, current_obj_step, current_probe_pos_step, current_tilts_step, current_static_background_step, current_aberrations_array_step, aberrations_array, beam_current, current_beam_current_step, xp);
        save_updated_arrays(output_folder, epoch,current_probe_step, current_probe_pos_step, current_tilts_step,current_obj_step, obj, probe, tilts_correction, positions_correction, positions, tilts,static_background, current_aberrations_array_step, current_static_background_step, count, current_loss, current_sse, aberrations_array, beam_current, current_beam_current_step, save_flag, save_loss_log, xp) # <-------------- save the results --------------
    save_current_checkpoint_obj_probe(output_folder, obj, probe, tilts_correction, positions_correction,positions, tilts, static_background, 1,1,1,1,1, 1, aberrations_array,beam_current, 1, xp)
    obj, probe, positions, positions_correction, tilts, tilts_correction, defocus_array,  beam_current=None, None, None, None, None,None, None, None
    try:
        cp.fft.config.clear_plan_cache()
        pool.free_all_blocks()
        pinned_pool.free_all_blocks()
    except:
        pass
    if print_flag!=0:
        sys.stdout.write("\nDone :)")
        sys.stdout.flush()



def bfgs_update(algorithm_type, this_slice_distances, this_step_probe, this_step_obj, this_step_pos_correction, this_step_tilts, measured_array, this_wavelength, masks, pixel_size_x_A, pixel_size_y_A, phase_norm_weight, abs_norm_weight, stepsize_threshold_low, probe_reg_weight, aperture_mask, recon_type, Cs, alpha_near_field, damping_cutoff_multislice, smooth_rolloff, update_extra_cut, keep_probe_states_orthogonal, do_charge_flip, cf_delta_phase, cf_delta_abs, cf_beta_phase, cf_beta_abs, phase_only_obj, beta_wedge, wolfe_c1_constant, wolfe_c2_constant, atv_weight, atv_q, atv_p, tune_only_probe_phase, propmethod, this_chopped_sequence, load_one_by_one, data_multiplier,data_pad, phase_plate_in_h5, print_flag, this_loss_weight, max_count, reduce_factor, optimism, mixed_variance_weight, mixed_variance_sigma, data_bin, data_shift_vector, smart_memory, default_float, default_complex, default_int, upsample_pattern, static_background, this_step_static_background, tilt_mode, fancy_sigma, tune_only_probe_abs, aberration_marker, this_step_aberrations_array, probe_marker, aberrations_array, compute_batch, current_window, current_window_weight, dynamically_resize_yx_object, lazy_clean, current_gaussian_filter, current_apply_gaussian_filter_amplitude, this_beam_current_step, xp, remove_fft_cache, is_first_epoch, hist_length, actual_step, fast_axis_reg_weight_positions, fast_axis_reg_weight_tilts, scan_size, current_slow_axis_reg_weight_positions, current_slow_axis_reg_coeff_positions, current_slow_axis_reg_weight_tilts, current_slow_axis_reg_coeff_tilts):
    global obj, probe, pool, pinned_pool, positions, positions_correction, tilts, tilts_correction, defocus_array, beam_current,  history_bfgs
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
    if probe_reg_weight==xp.inf:
        probe_reg_weight=0
        if type(aperture_mask)==xp.ndarray:
            probe=fourier_clean(probe, mask=aperture_mask, default_float=default_float)
        else:
            probe=fourier_clean(probe, cutoff=aperture_mask, default_float=default_float)
    if update_probe:
        if is_mixed_state and keep_probe_states_orthogonal:
            probe=make_states_orthogonal(probe)
        probe = fourier_clean(probe, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float)
    
    if update_obj:
        if do_charge_flip:
            obj=charge_flip(obj, cf_delta_phase, cf_delta_abs, cf_beta_phase, cf_beta_abs, fancy_sigma);
            reset_bfgs_history()
            empty_hist=True
        if not lazy_clean:
            if smart_memory:
                for i in range(obj.shape[2]):
                    for j in range(obj.shape[3]):
                        what=obj[:,:,i,j]
                        obj[:,:,i,j]=fourier_clean(what, cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
            else:
                obj=fourier_clean(obj,   cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
    if smart_memory:
        try:
            pool.free_all_blocks()
            pinned_pool.free_all_blocks()
        except:
            pass
    if empty_hist:
        total_loss, this_sse, this_object_grad, this_probe_grad, this_pos_grad, this_tilts_grad, static_background_grad, this_grad_aberrations_array, this_beam_current_grad = loss_and_direction(obj, probe, positions, positions_correction, tilts, tilts_correction, this_slice_distances,  measured_array,  algorithm_type, this_wavelength, update_probe, update_obj, update_pos_correction, update_tilts, masks, pixel_size_x_A, pixel_size_y_A, recon_type, Cs, defocus_array, alpha_near_field, damping_cutoff_multislice, smooth_rolloff_loss, propmethod, this_chopped_sequence, load_one_by_one, data_multiplier, data_pad, phase_plate_in_h5, this_loss_weight, data_bin, data_shift_vector, upsample_pattern, static_background, update_static_background, tilt_mode, aberration_marker, probe_marker, aberrations_array, compute_batch, phase_only_obj, beam_current, update_beam_current, update_aberrations_array, default_float, default_complex, xp, is_first_epoch, scan_size,fast_axis_reg_weight_positions, current_slow_axis_reg_weight_positions,current_slow_axis_reg_coeff_positions, current_slow_axis_reg_weight_tilts, current_slow_axis_reg_coeff_tilts, fast_axis_reg_weight_tilts, aperture_mask, probe_reg_weight, current_window_weight, current_window, phase_norm_weight, abs_norm_weight, atv_weight, atv_q, atv_p, mixed_variance_weight, mixed_variance_sigma) #get the loss and derivatives
        if smart_memory:
            try:
                pool.free_all_blocks()
                pinned_pool.free_all_blocks()
            except:
                pass
        if not lazy_clean and update_obj:
            if smart_memory:
                for i in range(obj.shape[2]):
                    for j in range(obj.shape[3]):
                        what=this_object_grad[:,:,i,j]
                        this_object_grad[:,:,i,j]=fourier_clean(what, cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
            else:
                this_object_grad=fourier_clean(this_object_grad,   cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
        if phase_only_obj:
            this_object_grad=complex_grad_to_phase_grad(this_object_grad, obj)
        if tune_only_probe_abs:
            this_probe_grad_fourier=ifft2(this_probe_grad)*this_probe_grad.shape[0]*this_probe_grad.shape[1]
            this_probe_fourier=fft2(probe)
            probe_fourier_mag,probe_fourier_phase=xp.sqrt(xp.abs(this_probe_fourier)), xp.angle(this_probe_fourier)
            this_probe_grad=complex_grad_to_mag_grad(this_probe_grad_fourier, probe_fourier_mag,probe_fourier_phase)
            tune_only_probe_phase=False
        if tune_only_probe_phase:
            this_probe_grad_fourier=ifft2_ishift(this_probe_grad) * this_probe_grad.shape[0] * this_probe_grad.shape[1]
            this_probe_fourier=shift_fft2(probe)
            probe_fourier_abs, probe_fourier_phase = xp.abs(this_probe_fourier), xp.angle(this_probe_fourier)
            this_probe_grad=complex_grad_to_phase_grad(this_probe_grad_fourier, this_probe_fourier)
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
        total_loss, this_sse, this_object_grad, this_probe_grad, this_pos_grad, this_tilts_grad, static_background_grad, this_grad_aberrations_array, this_beam_current_grad=history_bfgs["prev_loss"], history_bfgs["prev_sse"], history_bfgs["obj_grad"], history_bfgs["probe_grad"], history_bfgs["pos_grad"], history_bfgs["tilt_grad"], history_bfgs["static_background_grad"], history_bfgs["aberrations_grad"], history_bfgs["beam_current_grad"]
        
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
                        this_obj_update[:,:,i,j]=fourier_clean(what, cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
            else:
                this_obj_update=fourier_clean(this_obj_update,   cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
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
                                new_obj[:,:,i,j]=fourier_clean(what, cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
                    else:
                        new_obj=fourier_clean(new_obj,   cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
            else:
                new_obj=obj+actual_step*this_obj_update
        else:
            new_obj=obj
        if update_probe:
            if tune_only_probe_phase:
                probe_fourier=shift_fft2(probe)
                probe_fourier_abs, probe_fourier_angle=xp.abs(probe_fourier), xp.angle(probe_fourier)
                new_probe=ifft2_ishift(probe_fourier_abs * xp.exp(1j*(probe_fourier_angle + actual_step*this_probe_update)))
            else:
                if tune_only_probe_abs:
                    probe_fourier=shift_fft2(probe)
                    probe_fourier_abs, probe_fourier_angle = xp.abs(probe_fourier), xp.angle(probe_fourier)
                    new_probe=ifft2_ishift((probe_fourier_abs+ actual_step*this_probe_update) * xp.exp(1j*(probe_fourier_angle)))
                else:
                    new_probe=probe + actual_step*this_probe_update
        else:
            new_probe=probe
        new_positions_correction=positions_correction+actual_step*this_pos_update if update_pos_correction else positions_correction
        new_tilts_correction=tilts_correction+actual_step*this_tilts_update if update_tilts else tilts_correction
        new_static_background=static_background+actual_step*this_static_background_update if update_static_background else static_background
        new_aberrations_array=aberrations_array+actual_step*this_aberrations_array_update if update_aberrations_array else aberrations_array
        new_beam_current=beam_current+actual_step*this_beam_current_update if update_beam_current else beam_current
        new_total_loss, new_sse, new_object_grad, new_probe_grad, new_pos_grad, new_tilts_grad, new_static_background_grad, new_grad_aberrations_array, new_beam_current_grad = loss_and_direction(new_obj, new_probe, positions, new_positions_correction, tilts, new_tilts_correction, this_slice_distances,  measured_array,  algorithm_type, this_wavelength, update_probe, update_obj, update_pos_correction, update_tilts, masks, pixel_size_x_A, pixel_size_y_A, recon_type, Cs, defocus_array, alpha_near_field, damping_cutoff_multislice, smooth_rolloff_loss, propmethod, this_chopped_sequence, load_one_by_one, data_multiplier, data_pad, phase_plate_in_h5, this_loss_weight, data_bin, data_shift_vector, upsample_pattern, new_static_background, update_static_background, tilt_mode, aberration_marker, probe_marker, new_aberrations_array, compute_batch, phase_only_obj, new_beam_current, update_beam_current, update_aberrations_array, default_float, default_complex, xp, is_first_epoch, scan_size,fast_axis_reg_weight_positions, current_slow_axis_reg_weight_positions,current_slow_axis_reg_coeff_positions, current_slow_axis_reg_weight_tilts,current_slow_axis_reg_coeff_tilts, fast_axis_reg_weight_tilts, aperture_mask, probe_reg_weight, current_window_weight, current_window, phase_norm_weight, abs_norm_weight, atv_weight, atv_q, atv_p, mixed_variance_weight, mixed_variance_sigma)
        if smart_memory:
            try:
                pool.free_all_blocks()
                pinned_pool.free_all_blocks()
            except:
                pass
        if not(lazy_clean) and update_obj:
            if smart_memory:
                for i in range(new_obj.shape[2]):
                    for j in range(new_obj.shape[3]):
                        what=new_object_grad[:,:,i,j]
                        new_object_grad[:,:,i,j]=fourier_clean(what, cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
            else:
                new_object_grad=fourier_clean(new_object_grad,   cutoff=damping_cutoff_multislice-update_extra_cut, rolloff=smooth_rolloff, default_float=default_float)
        if phase_only_obj:
            new_object_grad=complex_grad_to_phase_grad(new_object_grad, new_obj)
        if tune_only_probe_abs:
            this_probe_grad_fourier=ifft2(new_probe_grad)*new_probe_grad.shape[0]*new_probe_grad.shape[1]
            new_probe_fourier=fft2(new_probe)
            probe_fourier_mag,probe_fourier_phase=xp.sqrt(xp.abs(new_probe_fourier)), xp.angle(new_probe_fourier)
            new_probe_grad=complex_grad_to_mag_grad(new_probe_fourier, probe_fourier_mag,probe_fourier_phase)
        if tune_only_probe_phase:
            new_probe_grad_fourier=ifft2_ishift(new_probe_grad) * new_probe_grad.shape[0] * new_probe_grad.shape[1]
            new_probe_fourier=shift_fft2(new_probe)
            probe_fourier_abs, probe_fourier_phase = xp.abs(new_probe_fourier), xp.angle(new_probe_fourier)
            new_probe_grad=complex_grad_to_phase_grad(new_probe_grad_fourier, new_probe_fourier)
        new_d_value=0
        if update_obj: new_d_value+=(2-phase_only_obj)*cp.sum(cp.real(cp.conjugate(this_obj_update)*new_object_grad));
        if update_probe: new_d_value+=(2-tune_only_probe_phase)*cp.sum(cp.real(cp.conjugate(this_probe_update)*new_probe_grad));
        if update_pos_correction: new_d_value+=cp.sum(this_pos_update*new_pos_grad);
        if update_tilts: new_d_value+=cp.sum(this_tilts_update*new_tilts_grad);
        if update_static_background: new_d_value+=cp.sum(this_static_background_update*new_static_background_grad);
        if update_aberrations_array: new_d_value+=cp.sum(this_aberrations_array_update*new_grad_aberrations_array);
        if update_beam_current: new_d_value+=cp.sum(this_beam_current_update*new_beam_current_grad);
        this_wolfe_1=wolfe_1(total_loss, new_total_loss, d_value, actual_step, wolfe_c1_constant)
        this_wolfe_2=wolfe_2(d_value, new_d_value, wolfe_c2_constant)
        if print_flag>=3:
            sys.stdout.write("\nUpdate %d. This loss is %.3e. Loss change is %.3e. Dir-derivative is %.3e. New dir-derivative is %.3e, This step is %.3e."%(count, total_loss, total_loss-new_total_loss, d_value, new_d_value, actual_step))
            sys.stdout.flush()
        if this_wolfe_1 and this_wolfe_2:
            break
        else:
            if not(this_wolfe_1) and this_wolfe_2:
                actual_step*=reduce_factor # backtracking
            if not(this_wolfe_2) and this_wolfe_1:
                actual_step*=optimism # boosting the step!
            if not(this_wolfe_2 or this_wolfe_1):
                actual_step=1e-10
            if not(max_count is None):
                if count>max_count:
                    break
            if actual_step<stepsize_threshold_low:
                break
        ##############################################################################################################################
    if new_total_loss>=total_loss + actual_step*wolfe_c1_constant*d_value: ## checking Wolfe_1 again
        sys.stdout.write('\nWARNING! The sufficient loss descrese is not achieved, the update is regected, keeping the the same step! Resetting the history! Please terminate if this message appears during the next epoch!\n')
        sys.stdout.flush()
        reset_bfgs_history()
        return positions_correction, tilts_correction, aberrations_array, beam_current,static_background, total_loss, this_sse
    if print_flag>=2:
        sys.stdout.write("\n-->Update done with %d steps! This loss is %.3e. Loss change is %.3e. Dir-derivative is %.3e. New dir-derivative is %.3e."%(count,total_loss, total_loss-new_total_loss, d_value, new_d_value))
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
    this_step_obj=actual_step if update_obj else 0
    this_step_probe=actual_step if update_probe else 0
    this_step_pos_correction=actual_step if update_pos_correction else 0
    this_step_tilts=actual_step if update_tilts else 0
    this_step_static_background=actual_step if update_static_background else 0
    this_step_aberrations_array=actual_step if update_aberrations_array else 0
    this_beam_current_step=actual_step if update_beam_current else 0
    
    if this_step_obj!=0 and (current_gaussian_filter or current_apply_gaussian_filter_amplitude):
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
                prev_grad=xp.pad(history_bfgs["obj_grad"], pad_width, mode="constant", constant_values=0)
                history_bfgs["obj_grad"]=prev_grad
                for itemind, item in enumerate(history_bfgs["obj_hist_s"]):
                    history_bfgs["obj_hist_s"][itemind]=xp.pad(item, pad_width, mode="constant", constant_values=0)
                for itemind, item in enumerate(history_bfgs["obj_hist_y"]):
                    history_bfgs["obj_hist_y"][itemind]=xp.pad(item, pad_width, mode="constant", constant_values=0)
    return new_positions_correction, new_tilts_correction, new_aberrations_array, new_beam_current,new_static_background, total_loss, this_sse



#### ------ initialize.py ---------
def create_h5_file_from_numpy(path_numpy, path_h5, swap_axes=False,flip_ky=False,flip_kx=False, flip_y=False,flip_x=False,comcalc_len=1000, comx=None, comy=None, bin=1, crop_left=None, crop_right=None, crop_top=None, crop_bottom=None, normalize=True, cutoff_ratio=None, pad_k=0, data_dtype=np.float32, rescale=1, exist_ok=True):
    sys.stdout.write("\n******************************************************************************\n************************ Creating an .h5 File ********************************\n******************************************************************************\n")
    sys.stdout.flush()
    if os.path.isfile(path_h5):
        print("\n.h5 File exists!")
        if exist_ok:
            return None
        else:
            print("\nDeleting the exitsting one!")
            if path_h5[-3:]==".h5" and path_numpy!=path_h5:
                try:
                    os.remove(path_h5)
                except:
                    pass
    if path_numpy[-4:]==".npy":
        data=np.load(path_numpy)
    else:
        f0=h5py.File(path_numpy, "r")
        data=np.array(f0["data"])
        f0.close()
    if flip_y:
        data=data[::-1,:,:,:]
    if flip_x:
        data=data[:,::-1,:,:]
    if flip_ky:
        data=data[:,:,::-1,:]
    if flip_kx:
        data=data[:,:,:,::-1]
    if swap_axes:
        data=np.swapaxes(data, 2,3)
    data=data.reshape(data.shape[0]*data.shape[1],data.shape[2], data.shape[3])
    data[data<0]=0
    x,y=np.arange(data.shape[2]), np.arange(data.shape[1])
    x,y=np.meshgrid(x-np.mean(x), y-np.mean(y), indexing="xy")
    r=(x**2+y**2)**0.5
    if comcalc_len is None:
        comcalc_len=data.shape[0]
    ssum=np.sum(data[:comcalc_len], axis=(1,2))
    if comx is None:
        comx=int(np.round(np.mean(np.sum(data[:comcalc_len]*x[None, :,:], axis=(1,2))/ssum)))
    if comy is None:
        comy=int(np.round(np.mean(np.sum(data[:comcalc_len]*y[None, :,:], axis=(1,2))/ssum)))
    sys.stdout.write("\nCOM x&y before correction: %d, %d"%( comx,comy))
    data=np.roll(data, (-comy, -comx), axis=(1,2))
    if comy>0:
        data[:,-comy:,:]=0
    elif comy<0:
        data[:, :-comy, :]=0
    if comx>0:
        data[:,:,-comx:]=0
    elif comx<0:
        data[:,:,:-comx]=0
    if not(cutoff_ratio is None):
        r= r>= np.max(x)*cutoff_ratio
        data[:, r]=0
    comx=int(np.round(np.mean(np.sum(data[:comcalc_len]*x[None, :,:], axis=(1,2))/ssum)))
    comy=int(np.round(np.mean(np.sum(data[:comcalc_len]*y[None, :,:], axis=(1,2))/ssum)))
    sys.stdout.write("\nCOM x&y after correction: %d, %d"%( comx,comy))
    if normalize:
        ssum=np.sum(data[:comcalc_len], axis=(1,2))
        data/=np.mean(ssum)
    if rescale!=1:
        data/=rescale
    if not(crop_bottom is None):
        data=data[:,:-crop_bottom, :]
    if not(crop_top is None):
        data=data[:,crop_top:, :]
    if not(crop_left is None):
        data=data[:,:, crop_left:]
    if not(crop_right is None):
        data=data[:,:, :-crop_right]
    if bin!=1:
        data2=np.zeros((data.shape[0], data.shape[1]//bin, data.shape[2]//bin))
        for i in range(bin):
            for j in range(bin):
                data2+=data[:,i:bin*(data.shape[1]//bin):bin, j:bin*(data.shape[1]//bin):bin]
        data=data2
        del(data2)
    if pad_k!=0:
        data=np.pad(data, [[0,0],[pad_k, pad_k],[pad_k,pad_k]])
    if not(data_dtype is None):
        data=data.astype(data_dtype)
    f=h5py.File(path_h5, "a")
    f.create_dataset("data", data=data)
    f.close()

def get_curl(angle, dpcx, dpcy):
    rotx, roty = dpcx * np.cos(angle) - dpcy * np.sin(angle), dpcx * np.sin(angle) + dpcy * np.cos(angle)
    gXY, gXX = np.gradient(rotx); gYY, gYX = np.gradient(roty)
    curl=np.std(gXY - gYX)
    return curl
def get_curl_derivative(angle, dpcx, dpcy):
    rotx, roty = dpcx * np.cos(angle) - dpcy * np.sin(angle), dpcx * np.sin(angle) + dpcy * np.cos(angle)
    gXY, gXX = np.gradient(rotx); gYY, gYX = np.gradient(roty)
    std_derivative = 1 / np.sqrt(len(gXY) - 1)
    curl_derivative = np.sum((gXX + gYY) * (-dpcx * np.sin(angle) - dpcy * np.cos(angle)))
    return std_derivative * curl_derivative
def GetPLRotation(dpcx, dpcy):
    sys.stdout.write("\nStarting the DPC rotation angle calculation!")
    sys.stdout.flush()
    result = minimize(get_curl, x0=0, method="Powell", args=(dpcx, dpcy), bounds=[(-np.pi, np.pi)], tol=1e-4, options={"maxiter":100})
    #print(result)
    R = result.x[0]
    return R
    
def get_offset(x_range, y_range, scan_step_A, detector_pixel_size_rezA, patternshape, rot_angle_deg=0):
    px_size=1/(detector_pixel_size_rezA*patternshape[-1])
    positions=np.empty((x_range*y_range,2))
    i=0
    for ind1 in range(0,y_range,1):
        for ind2 in range(0, x_range,1):
            positions[i,0]=ind1
            positions[i,1]=ind2
            i+=1
    if rot_angle_deg!=0:
        rot_ang=rot_angle_deg*np.pi/180
        positions_prime_x,positions_prime_y=positions[:,1] * np.cos(rot_ang) + positions[:,0] * np.sin(rot_ang), -1*positions[:,1] * np.sin(rot_ang) + positions[:,0] * np.cos(rot_ang)
        positions[:,0]=positions_prime_y
        positions[:,1]=positions_prime_x
    offy=-np.min(positions[:,0])*scan_step_A
    offx=-np.min(positions[:,1])*scan_step_A
    return offy, offx
    
def get_positions_px_size(x_range, y_range,scan_step_A, detector_pixel_size_rezA, patternshape, rot_angle_deg=0, flip_x=False,flip_y=False, print_flag=False):
    px_size=1/(detector_pixel_size_rezA*patternshape[-1])
    if print_flag:
        sys.stdout.write("\npixel size in A: %.3e"%px_size)
    positions=np.empty((x_range*y_range,2))
    i=0
    for ind1 in range(0,y_range,1):
        for ind2 in range(0, x_range,1):
            positions[i,0]=ind1
            positions[i,1]=ind2
            i+=1
    if rot_angle_deg!=0:
        rot_ang=rot_angle_deg*np.pi/180
        positions_prime_x,positions_prime_y=positions[:,1] * np.cos(rot_ang) + positions[:,0] * np.sin(rot_ang), -1*positions[:,1] * np.sin(rot_ang) + positions[:,0] * np.cos(rot_ang)
        positions[:,0]=positions_prime_y
        positions[:,1]=positions_prime_x
    if flip_x:
        positions[:,1]*=-1
    if flip_y:
        positions[:,0]*=-1
    positions[:,0]=positions[:,0]-np.min(positions[:,0])
    positions[:,1]=positions[:,1]-np.min(positions[:,1])
    positions*=scan_step_A/px_size
    return positions, px_size

    
    
    
def append_exp_params(experimental_params, pypty_params=None):
    sys.stdout.write("\n******************************************************************************\n******** Attaching the experimental parameters to your PyPty preset. *********\n******************************************************************************\n")
    sys.stdout.flush()
    if type(pypty_params)==str:
        pypty_params=load_params(pypty_params)
    
    path_data_h5=experimental_params.get("data_path", "")
    output_folder=experimental_params.get("output_folder", "")
    path_json=experimental_params.get("path_json", "")

    acc_voltage=experimental_params.get("acc_voltage", None)
    rez_pixel_size_A=experimental_params.get("rez_pixel_size_A", None)
    rez_pixel_size_mrad=experimental_params.get("rez_pixel_size_mrad", None)
    conv_semiangle_mrad=experimental_params.get("conv_semiangle_mrad", None)

    aperture=experimental_params.get("aperture", None)
    data_pad=experimental_params.get("data_pad", None)
    upsample_pattern=experimental_params.get("upsample_pattern",1)

    scan_size=experimental_params.get("scan_size", None)
    scan_step_A=experimental_params.get("scan_step_A", None)
    fov_nm=experimental_params.get("fov_nm", None)
    flip_x_positions = experimental_params.get("flip_x_positions", False)
    flip_y_positions = experimental_params.get("flip_y_positions", False)
    defocus=experimental_params.get("defocus", 0)
    PLRotation_deg=experimental_params.get("PLRotation_deg", 0)
    total_thickness=experimental_params.get("total_thickness", 1)
    num_slices=experimental_params.get("num_slices", 1)
    bright_threshold=experimental_params.get("bright_threshold", 0.1)
    plot=experimental_params.get("plot", True)
    print_flag=experimental_params.get("print_flag", True)
    comx=None
    comy=None
    try:
        os.makedirs(output_folder, exist_ok=True)
    except:
        pass
    try:
        with open(path_json, 'r') as file:
            jsondata = json.load(file)
    except:
        if print_flag:
            sys.stdout.write("\njson is not provided!")
    if path_data_h5[-3:]==".h5":
        h5file=h5py.File(path_data_h5, "r")
        h5data=h5file["data"]
    elif path_data_h5[-4:]==".npy":
        h5data=np.load(path_data_h5)
        if len(h5data.shape)==4:
            scan_size=[h5data.shape[0], h5data.shape[1]]
            h5data=h5file.reshape(h5data.shape[0]* h5data.shape[1], h5data.shape[2],h5data.shape[3])
    if scan_size is None:
        scan_size=jsondata['metadata']['scan']['scan_size'];
    if acc_voltage is None:
        try:
            acc_voltage=jsondata['metadata']['hardware_source']['high_tension_v']*1e-3; ##kV
        except:
            print("\nAssuming 60kV of acc. voltage!")
            acc_voltage=60
    if defocus is None:
        try:
            defocus=-jsondata['metadata']['hardware_source']['defocus']; ## unknown units
        except:
            print("\nAssuming zero defocus!")
            defocus=0
    if fov_nm is None:
        try:
            fov_nm=jsondata['metadata']['scan']['fov_nm'];
        except:
            fov_nm=0
            print("\nNo scan-FOV provided, specify scan step!")
    if scan_step_A is None: scan_step_A=fov_nm*10/scan_size[0]
    
    # conv_semiangle_mrad
    # rez_pixel_size_A
    # PLRotation_deg
    wavelength=12.4/np.sqrt(acc_voltage*(acc_voltage+2*511))
    y_range,x_range=scan_size
    if PLRotation_deg is None:
        x,y=np.arange(h5data.shape[-1]), np.arange(h5data.shape[-2])
        x,y=np.meshgrid(x-np.mean(x), y-np.mean(y), indexing="xy")
        ssum=np.empty(scan_size)
        if (comx is None) or (comy is None):
            comx=np.empty(scan_size)
            comy=np.empty(scan_size)
            for index_data_y in range(scan_size[0]):
                for index_data_x in range(scan_size[1]):
                    dataindex=index_data_x+index_data_y*scan_size[1]
                    ssum[index_data_y, index_data_x]=np.sum(h5data[dataindex])
                    comx[index_data_y, index_data_x]=np.sum(h5data[dataindex]*x)
                    comy[index_data_y, index_data_x]=np.sum(h5data[dataindex]*y)
            comx=comx/ssum.astype(np.float32)
            comy=comy/ssum.astype(np.float32)
            comx-=np.mean(comx)
            comy-=np.mean(comy)
        PLRotation=GetPLRotation(comx,comy)
        PLRotation_deg=PLRotation*180/np.pi ## we need a negative angle for the scan grid rotation
        if print_flag:
            sys.stdout.write("\niDPC rotation angle is %.2f deg. (Rotation of the reciprocal space with respect to real space.)"%PLRotation_deg)
    mean_pattern_as_it_is=np.mean(h5data, axis=0)
    mean_pattern=upsample_something(mean_pattern_as_it_is, upsample_pattern, True, np)
    if aperture is None:
        aperture=mean_pattern>bright_threshold*np.max(mean_pattern)
    if plot:
        try:
            fig, ax=plt.subplots(1,2)
            ax[0].imshow(mean_pattern)
            ax[0].set_title("mean pattern")
            ax[1].imshow(aperture)
            ax[1].set_title("bright field disk")
            plt.show()
        except:
            pass
    if rez_pixel_size_mrad is None:
        if rez_pixel_size_A is None:
            r=np.sqrt(np.sum(aperture)/np.pi)
            if print_flag:
                sys.stdout.write("\nRadius of bright field is %.2f px"%r)
            rez_pixel_size_A=conv_semiangle_mrad*1e-3/(r*wavelength) ## A^-1
        else:
            rez_pixel_size_A/=upsample_pattern
    else:
        rez_pixel_size_A=rez_pixel_size_mrad*1e-3/(upsample_pattern*wavelength)
    positions, old_px_size=get_positions_px_size(x_range, y_range, scan_step_A, detector_pixel_size_rezA=rez_pixel_size_A, patternshape=[mean_pattern.shape[0],mean_pattern.shape[1]], rot_angle_deg=-1*PLRotation_deg, flip_x=flip_x_positions,flip_y=flip_y_positions, print_flag=print_flag)
    old_shape=mean_pattern.shape[1]
    if data_pad is None: data_pad=int(np.ceil(old_shape/4));
    new_shape=old_shape+2*data_pad
    new_px_size=old_px_size*old_shape/new_shape
    positions *= new_shape/old_shape
    if print_flag:
        sys.stdout.write("\nPixel size after padding: %.2e Å"%new_px_size)
    aperture=np.pad(aperture, data_pad, mode="constant", constant_values=0)
    if pypty_params is None:
        pypty_params={
        'data_path': path_data_h5,
        'output_folder': output_folder,
        'positions': positions,
        'obj': np.ones((1,1,num_slices,1), dtype=np.complex128),
        'acc_voltage': acc_voltage,
        'slice_distances': np.array([total_thickness/num_slices]),
        'pixel_size_x_A': new_px_size,
        'pixel_size_y_A': new_px_size,
        'aperture_mask': aperture,
        'extra_probe_defocus': defocus,
        'data_pad': data_pad,
        'probe': None,
        'epoch_max': 1000,
        'epoch_prev': 0,
        'print_flag': 3,
        }
    else:
        pypty_params['data_path']=path_data_h5
        pypty_params['output_folder']=output_folder
        pypty_params['positions']=positions
        pypty_params['acc_voltage']=acc_voltage
        pypty_params['slice_distances']=np.array([total_thickness/num_slices])
        pypty_params['pixel_size_x_A']=new_px_size
        pypty_params['pixel_size_y_A']=new_px_size
        pypty_params['aperture_mask']=aperture
        pypty_params['extra_probe_defocus']=defocus
        pypty_params['data_pad']=data_pad
        pypty_params['probe']=None
        pypty_params['obj']=np.ones((1,1,num_slices,1), dtype=np.complex128)
    pypty_params["mean_pattern"]=mean_pattern_as_it_is
    pypty_params["upsample_pattern"]=upsample_pattern
    pypty_params["rez_pixel_size_A"]=rez_pixel_size_A
    pypty_params["conv_semiangle_mrad"]=conv_semiangle_mrad
    pypty_params["scan_size"]=scan_size
    pypty_params["scan_step_A"]=scan_step_A
    pypty_params["fov_nm"]=fov_nm
    pypty_params["PLRotation_deg"]=PLRotation_deg
    pypty_params["bright_threshold"]=bright_threshold
    pypty_params["plot"]=plot
    pypty_params["comx"]=comx
    pypty_params["comy"]=comy
    pypty_params["print_flag"]=print_flag
    pypty_params["num_slices"] = num_slices
    pypty_params["total_thickness"] = total_thickness
    try:
        h5file.close()
    except:
        pass
    return pypty_params

def getdpcpot(pypty_params, hpass=0, lpass=0, save=False, comx=None, comy=None):
    dataset_h5=pypty_params.get("data_path", "")
    scan_size=pypty_params.get("scan_size", None)
    angle=pypty_params.get("PLRotation_deg", None)
    plot=pypty_params.get("plot", False)
    if dataset_h5[-3:]==".h5":
        dataset_h5=h5py.File(dataset_h5, "r")
        dataset_h5=dataset_h5["data"]
    elif dataset_h5[-4:]==".npy":
        dataset_h5=np.load(dataset_h5)
        if len(dataset_h5.shape)==4:
            scan_size=[dataset_h5.shape[0], dataset_h5.shape[1]]
            dataset_h5=dataset_h5.reshape(dataset_h5.shape[0]* dataset_h5.shape[1], dataset_h5.shape[2],dataset_h5.shape[3])
    if (comx is None) or (comy is None):
        comx=np.zeros(dataset_h5.shape[0])
        comy=np.zeros(dataset_h5.shape[0])
        x=np.arange(dataset_h5.shape[2])
        y=np.arange(dataset_h5.shape[1])
        x=x-np.mean(x)
        y=y-np.mean(y)
        x,y=np.meshgrid(x,y, indexing="xy")
        for i in range(dataset_h5.shape[0]):
            data=dataset_h5[i]
            comx[i]=np.sum(x*data)
            comy[i]=np.sum(y*data)
        comx-=np.mean(comx)
        comy-=np.mean(comy)
        comx=comx.reshape(scan_size[0], scan_size[1])
        comy=comy.reshape(scan_size[0], scan_size[1])
    if angle is None:
        angle=GetPLRotation(comx,comy)
    else:
        angle=angle*np.pi/180
    rcomx = comx * np.cos(angle) + comy * np.sin(angle)
    rcomy = -comx * np.sin(angle) + comy * np.cos(angle)
    fCX = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(rcomx)))
    fCY = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(rcomy)))
    KX = np.fft.fftshift(np.fft.fftfreq(rcomx.shape[1]))
    KY = np.fft.fftshift(np.fft.fftfreq(rcomx.shape[0]))
    kx, ky = np.meshgrid(KX, KY, indexing="xy")
    fCKX = fCX * kx
    fCKY = fCY * ky
    fnum = (fCKX + fCKY)
    fdenom = np.pi * 2j * (hpass + (kx ** 2 + ky ** 2) + lpass * (kx ** 2 + ky ** 2) ** 2)
    fK = np.divide(fnum, fdenom)
    pot=np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(fK))))
    if plot:
        fig,ax=plt.subplots(1,1)
        im=ax.imshow(pot, cmap="gray")
        this_x=np.arange(scan_size[1])
        this_y=np.arange(scan_size[0])
        this_x,this_y=np.meshgrid(this_x, this_y, indexing="xy")
        #ax.quiver(this_x, this_y, -comx, -comy, units="xy", color="red", capstyle="round")
        #fig.colorbar(im, ax=ax)
        ax.set_title("DPC potential")
        ax.axis("off")
        ax.axis("off")
        plt.show()
    pypty_params["comx"]=comx
    pypty_params["comy"]=comy
    if save:
        np.save(pypty_params["output_folder"]+"/idpc.npy", pot)
    pypty_params["PLRotation_deg"]=angle*180/np.pi
    return pot, pypty_params, rcomx, rcomy

def getvirtualhaadf(pypty_params, save=True):
    dataset_h5=pypty_params.get("data_path", "")
    scan_size=pypty_params.get("scan_size", None)
    plot=pypty_params.get("plot", False)
    if dataset_h5[-3:]==".h5":
        dataset_h5=h5py.File(dataset_h5, "r")
        dataset_h5=dataset_h5["data"]
    elif dataset_h5[-4:]==".npy":
        dataset_h5=np.load(dataset_h5)
        if len(dataset_h5.shape)==4:
            scan_size=[dataset_h5.shape[0], dataset_h5.shape[1]]
            dataset_h5=dataset_h5.reshape(dataset_h5.shape[0]* dataset_h5.shape[1], dataset_h5.shape[2],dataset_h5.shape[3])
    haadf=-1*np.sum(dataset_h5, (-1,-2)).reshape(scan_size)
    if plot:
        fig,ax=plt.subplots(1,1)
        im=ax.imshow(haadf, cmap="gray")
        ax.set_title("Virtual HAADF")
        ax.axis("off")
        ax.axis("off")
        plt.show()
    if save:
        np.save(pypty_params["output_folder"]+"/virtual_haadf.npy",haadf)
    return haadf

def plot_modes(ttt):
    if len(ttt.shape)==4:
        for i in range(ttt.shape[-1]):
            for j in range(ttt.shape[-2]):
                fig, ax=plt.subplots(1,4, figsize=(10,40))
                im0=ax[0].imshow(np.abs(ttt[:,:,j,i]), cmap="gray", vmax=np.max(np.abs(ttt)))
                im1=ax[1].imshow(np.angle(ttt[:,:,j,i]), cmap="gray")
                im2=ax[2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ttt[:,:,j,i])))), cmap="gray")
                im3=ax[3].imshow(np.angle(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ttt[:,:,j,i])))), cmap="gray")
                fig.colorbar(im0, ax=ax[0], pad=0, fraction=0.047, location='bottom')
                fig.colorbar(im1, ax=ax[1], pad=0, fraction=0.047, location='bottom')
                fig.colorbar(im2, ax=ax[2], pad=0, fraction=0.047, location='bottom')
                fig.colorbar(im3, ax=ax[3], pad=0, fraction=0.047, location='bottom')
                ax[0].axis("off")
                ax[0].set_title("R-Space mag")
                ax[1].axis("off")
                ax[1].set_title("R-Space phase")
                ax[2].axis("off")
                ax[2].set_title("Q-Space mag")
                ax[3].axis("off")
                ax[3].set_title("Q-Space phase")
                plt.tight_layout()
                plt.show()
    else:
        for i in range(ttt.shape[-1]):
            fig, ax=plt.subplots(1,4, figsize=(10,40))
            im0=ax[0].imshow(np.abs(ttt[:,:,i]), cmap="gray", vmax=np.max(np.abs(ttt)))
            im1=ax[1].imshow(np.angle(ttt[:,:,i]), cmap="gray")
            im2=ax[2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ttt[:,:,i])))), cmap="gray")
            im3=ax[3].imshow(np.angle(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ttt[:,:,i])))), cmap="gray")
            fig.colorbar(im0, ax=ax[0], pad=0, fraction=0.047, location='bottom')
            fig.colorbar(im1, ax=ax[1], pad=0, fraction=0.047, location='bottom')
            fig.colorbar(im2, ax=ax[2], pad=0, fraction=0.047, location='bottom')
            fig.colorbar(im3, ax=ax[3], pad=0, fraction=0.047, location='bottom')
            ax[0].axis("off")
            ax[0].set_title("R-Space mag")
            ax[1].axis("off")
            ax[1].set_title("R-Space phase")
            ax[2].axis("off")
            ax[2].set_title("Q-Space mag")
            ax[3].axis("off")
            ax[3].set_title("Q-Space phase")
            plt.tight_layout()
            plt.show()

def get_ptycho_obj_from_scan(params, num_slices=None, array_phase=None,array_abs=None, scale_phase=1, scale_abs=1, cutoff=None, scan_array_A=None, return_array=False, fill_value_type="edge", xp=np):
    data_path=params.get("data_path", "")
    try:
        if data_path[-2:]=="h5":
            this_file=h5py.File(data_path, "r")
            dataset=this_file['data']
        else:
            dataset=np.load(data_path, mmap_mode="r")
        if len(dataset.shape)==2:
            masks=params.get("masks", None)
            psx,psy=masks.shape[-2], masks.shape[-1]
            psx, psy=psx+2*data_pad, psy+2*data_pad
        else:
            psx,psy=dataset.shape[-2], dataset.shape[-1]
            psx, psy=psx+2*data_pad, psy+2*data_pad
        try:
            this_file.close()
        except:
            pass
    except:
        try:
            aperture=params.get("aperture_mask", None)
            psx,psy=aperture.shape[1], aperture.shape[0]
        except:
            probe=params.get("probe", None)
            psx,psy=probe.shape[1], probe.shape[0]
    data_pad=params.get("data_pad", 0)
    if num_slices is None:
        num_slices=params.get("num_slices", 1)
    if num_slices == "auto":
        num_slices= int(np.ceil((np.max(array_phase)-np.min(array_phase))/(0.75*np.pi)))
    positions=np.copy(np.array(params.get("positions", [[0,0]])))
    sequence=params.get("sequence", None)
    pixel_size_x_A=params.get("pixel_size_x_A", 1)
    scan_step_A=params.get("scan_step_A", 1)
    total_thickness=params.get("total_thickness", 1)
    if cutoff is None:
        cutoff=pixel_size_x_A/scan_step_A
    use_full_FOV=params.get("use_full_FOV", False)
    positions=np.round(positions).astype(int)
    if scan_array_A is None:
        scan_y, scan_x=positions[:,0], positions[:,1]
    else:
        scan_y, scan_x=scan_array_A[:,0]/pixel_size_x_A, scan_array_A[:,1]/pixel_size_x_A
    if not(use_full_FOV):
        scan_y, scan_x=scan_y-np.min(scan_y), scan_x-np.min(scan_x)
    scan_y, scan_x=scan_y+psy//2, scan_x+psx//2
    scan_y, scan_x = scan_y.astype(int), scan_x.astype(int)
    max_y_px,max_x_px=np.max(scan_y)+psy-psy//2,np.max(scan_x)+psx-psx//2
    max_y_px,max_x_px=np.max(positions[:,0])+psy, np.max(positions[:,1])+psx
    image_gird_y, image_gird_x=np.arange(max_y_px), np.arange(max_x_px)
    im_X,im_Y=np.meshgrid(image_gird_x, image_gird_y, indexing="xy")
    if xp==np:
        interpolator=CloughTocher2DInterpolator
    #else:
     #   interpolator=cupyx.scipy.interpolate.CloughTocher2DInterpolator
    #cupyx.scipy.interpolate.CloughTocher2DInterpolator
    if array_phase is None:
        phase_ptycho=0
    else:
        if fill_value_type=="median":
            fill_value=np.median(array_phase)
        else:
            if fill_value_type=="edge":
                fill_value=np.max([np.min(array_phase[:3,:]), np.min(array_phase[-3:,:]), np.min(array_phase[:,:3]), np.min(array_phase[:,-3:])])
            else:
                fill_value=0
        f=interpolator(list(zip(scan_x, scan_y)), array_phase.flatten(), fill_value=fill_value)
        phase_ptycho=f(im_X,im_Y)
       
    if array_abs is None:
        abs_ptycho=1
    else:
        if fill_value_type=="median":
            fill_value=np.median(array_abs)
        else:
            if fill_value_type=="edge":
                fill_value=np.mean([np.mean(array_abs[:1,:]), np.mean(array_abs[-1:,:]), np.mean(array_abs[:,:1]), np.mean(array_abs[:,-1:])])
            else:
                fill_value=0
        f_abs=CloughTocher2DInterpolator(list(zip(scan_x, scan_y)), array_abs.flatten(), fill_value=fill_value)
        abs_ptycho=f_abs(im_X,im_Y)
    ptycho=scale_abs*(abs_ptycho**(1/num_slices))*xp.exp(1j*phase_ptycho*scale_phase/num_slices)
    if not(sequence is None) and not(use_full_FOV):
        minx, miny, maxx, maxy=np.min(positions[sequence,1]), np.min(positions[sequence,0]), np.max(positions[sequence,1]), np.max(positions[sequence,0])
        ptycho=ptycho[miny:maxy+psy, minx: maxx+ psx]
    
    ptycho=np.expand_dims(np.tile(np.expand_dims(ptycho,-1), (num_slices)),-1)
    if xp!=np:
        ptycho=ptycho.get()
    params["slice_distances"] =  np.array([total_thickness / num_slices])
    params["obj"] = ptycho
    if not return_array:
        del ptycho
        ptycho=None
    return params, ptycho

def get_aperture(params):
    data_path=params.get("data_path", "")
    data_pad=params.get("data_pad", 0)
    plot=params.get("plot", False)
    threshold=params.get("bright_threshold", 0.4)
    if data_path[-2:]=="h5":
        this_file=h5py.File(data_path, "r")
        dataset=this_file['data']
    else:
        dataset=np.load(data_path, mmap_mode="r")
    if len(dataset.shape)==3:
        meanpat=np.mean(dataset[:], 0)
    if len(dataset.shape)==4:
        meanpat=np.mean(dataset[:,:], (0,1))
    meanpat/=np.max(meanpat)
    meanpat=meanpat>threshold
    meanpat=np.pad(meanpat, data_pad)
    if plot:
        plt.imshow(meanpat)
        plt.show()
    try:
        this_file.close()
    except:
        pass
    params["aperture_mask"]=meanpat
    return params



def run_tcbf_alignment(params, aberrations=None, binning_for_fit=[8], save=True, n_aberrations_to_fit=12, plot_CTF_shifts=True, plot_inter_image=True, save_inter_imags=False,refine_box_dim=10, upsample=3, reference_type="bf", optimize_angle=False, scan_pad=None, cancel_large_shifts=None, compensate_lowfreq_drift=False, aperture=None, append_lowfreq_shifts_to_params=True, subscan_region=None, interpolate_scan_factor=1, cross_corr_type="phase", save_iteration_results=False, binning_cross_corr=1, phase_cross_corr_formula=False, f_scale_lsq=1,x_scale_lsq=1, loss_lsq="linear", tol_ctf=1e-8):
    """
    This function fit the beam CTF to the shifts between the individual pixel images of the 4d-stem dataset. The shift estimation is done via phase-cross correaltion. The shift of the CTF can be done either on an aberration basis or on a full discretized grid.
    inputs:
        pypty_params - dictionary with experimental parameters and reconsturciton settings. For more please see functions append_exp_params() and run_ptychography()
        aberrations- list with an initial guess for aberrations.
        binning_for_fit- list of binning values at which the fit will be performed. To do 4 iterations at binning value of 8 it should be [8,8,8,8]
        save - boolean flag. If true, the intermidate tcBF images on the intial scan grid will be saved as .png
        tol_ctf - tolerance for CTF fit
        n_aberrations_to_fit - integer. If the inital guess for aberrations is not provided, the code with try to initialize them as a list of zeros with this length
        plot_CTF_shifts - boolean flag
        plot_inter_image - boolean flag
        save_inter_imags - boolean flag
        refine_box_dim - radius of a box in which the shifts between the images will be refined. The cross-correlation can estimate the shift only on the initial grid. To get a more precise value, we have to interpolate. This is done here via a cubic spline.
        upsample - the upsampling of the cross-correlation for precise maximum estimation.
        reference_type - by default is set to "bf". In this case all pixel images will be correlated to the tcbf estimate. The other option is "zero". In this case the images will be correlated to the image of the pixel closest to the optical axis
        optimize_angle - experimental feature. Should allow to fit the PL rotation angle
        scan_pad - amount of scan positions to add to the edges in order to prevent wrap around artifacts. If None the code will figure it out automatically
        cancel_large_shifts - None or float strictly between 0 and 1. If not None, the abnoramally large shifts will be ignored in the CTF fit.
        compensate_lowfreq_drift - boolean flag. If True the code will try to cancel the aperture drift.
        aperture - boolean mask for aperture. If None the code will try to get it from the pypty_params. Note that the function append_exp_params generates the aperture automatically
        append_lowfreq_shifts_to_params - boolean flag. If true, the lowfreq drift correction will be stored in pypty_params. This should accelerate the later preparation of the data.
        subscan_region,  None or list of subscan boundaries (left, top, right, bottom) on which one should do the fit.
        interpolate_scan_factor, integer default 1. If larger than 1 the scan will be upsampled via interpolation. This is an experimental feature!!!
        cross_corr_type - type of cross correlation. Default "phase" for phase cross correlation that should perform better with noisy data. Anything but "phase" will result in a classical Foruier-correlation.
    outputs:
        pypty_params - updated dictionary with parameters.
    """
    global cpu_mode
    pypty_params=params.copy()
    ## load parameters
    dataset_h5=pypty_params.get("data_path", "")
    acc_voltage=pypty_params.get("acc_voltage", 60)
    scan_size=np.copy(np.array(pypty_params.get("scan_size", None)))
    scan_step_A=pypty_params.get("scan_step_A", 1)
    if aperture is None:
        aperture=pypty_params.get("aperture_mask", None)
    if type(aperture)==str:
        if aperture=="none" or aperture=="None":
            aperture=None
    pixel_size_x_A=pypty_params.get("pixel_size_x_A", 1)
    rot_deg=pypty_params.get("PLRotation_deg", 0)
    rez_pixel_size_A=pypty_params.get("rez_pixel_size_A", 1)
    data_pad=pypty_params.get("data_pad",0)
    upsample_pattern=pypty_params.get("upsample_pattern",1)
    smart_memory=pypty_params.get("smart_memory", True)
    try:
        smart_memory=smart_memory(0)
    except:
        smart_memory=smart_memory
    sequence=pypty_params.get("sequence", None)
    if cross_corr_type!="phase": phase_cross_corr_formula=False;
    if not(sequence is None):
        mask_sequence=np.ones(scan_size[0]*scan_size[1])
        mask_sequence[sequence]=0
        mask_sequence=mask_sequence.reshape(scan_size[0], scan_size[1])
    
    if upsample_pattern!=1:
        if not(aperture is None):
            aperture=downsample_something(aperture, upsample_pattern, np)
        data_pad=data_pad//upsample_pattern
        rez_pixel_size_A*=upsample_pattern
    if data_pad!=0 and not(aperture is None):
        aperture=aperture[data_pad:-data_pad,data_pad:-data_pad]
    wavelength=12.4/np.sqrt(acc_voltage*(acc_voltage+2*511))
    mrad_per_px=1000*rez_pixel_size_A*wavelength
    if aberrations is None:
        aberrations=list(np.zeros(n_aberrations_to_fit))
        aberrations[0]=-1*pypty_params.get("extra_probe_defocus", 0)
    
    plot=pypty_params.get("plot", False)
    print_flag=pypty_params.get("print_flag", False)
    num_abs=len(aberrations)
    possible_n, possible_m, possible_ab=convert_num_to_nmab(num_abs)
    aber_print, s=nmab_to_strings(possible_n, possible_m, possible_ab), ""
    for i in range(len(aberrations)): s+=aber_print[i]+" %.2e Å, "%aberrations[i];
    if print_flag:
        sys.stdout.write("\n******************************************************************************\n************************ Running the tcBF alignment **************************\n******************************************************************************\n")
        sys.stdout.write("\nInitial aberrations: %s"% s[:-2]);
        sys.stdout.flush()
    angle_offset=-1*rot_deg*3.141592654/180
    rot_rad=0
    
    if dataset_h5[-3:]==".h5":
        f=h5py.File(dataset_h5, "r")
        dataset_h5=f["data"]
    elif dataset_h5[-4:]==".npy":
        dataset_h5=np.load(dataset_h5)
        if len(dataset_h5.shape)==4:
            scan_size=[dataset_h5.shape[0], dataset_h5.shape[1]]
            dataset_h5=dataset_h5.reshape(dataset_h5.shape[0]* dataset_h5.shape[1], dataset_h5.shape[2],dataset_h5.shape[3])
    
   
    ## if bf disc is wobbling, try to compensate it, also we can save this shifts for ptycho reconsturction coming after this alignment!
    if compensate_lowfreq_drift:
        aperture_shifts_x=pypty_params.get("aperture_shifts_x", None)
        aperture_shifts_y=pypty_params.get("aperture_shifts_y", None)
        aperture = pypty_params.get("lowfreq_compensated_aperture", None)
        if (aperture_shifts_x is None) or (aperture_shifts_y is None) or (aperture is None):
            aperture_shifts_x, aperture_shifts_y=np.zeros(dataset_h5.shape[0], dtype=int), np.zeros(dataset_h5.shape[0], dtype=int)
            apx,apy=np.meshgrid(np.arange(-dataset_h5.shape[2]//2,dataset_h5.shape[2]-dataset_h5.shape[2]//2,1),np.arange(-dataset_h5.shape[1]//2,dataset_h5.shape[1]-dataset_h5.shape[1]//2,1), indexing="xy")
            structure = np.ones((5, 5), dtype=bool)
            if aperture is None:
                aperture_0=np.zeros((dataset_h5.shape[1], dataset_h5.shape[2]))
            for ind1 in range(dataset_h5.shape[0]):
                testim=dataset_h5[ind1]
                testim=testim>compensate_lowfreq_drift*np.mean(testim)
                testim=binary_closing(testim, structure=structure)
                aperture_shifts_x[ind1]=np.average(apx, weights=testim)
                aperture_shifts_y[ind1]=np.average(apy, weights=testim)
                if ind1==0:
                    if plot:
                        plt.imshow(testim)
                        plt.title("binary pattern for shift estimation")
                        plt.axis("off")
                        plt.show()
                if aperture is None:
                    rolled_im=np.roll(testim, (-int(np.round(aperture_shifts_y[ind1])), -int(np.round(aperture_shifts_x[ind1]))), axis=(0,1))
                    aperture_0+=rolled_im
            if append_lowfreq_shifts_to_params:
                pypty_params["aperture_shifts_x"]=aperture_shifts_x.reshape(scan_size[0], scan_size[1])
                pypty_params["aperture_shifts_y"]=aperture_shifts_y.reshape(scan_size[0], scan_size[1])
            if aperture is None:
                pypty_params["lowfreq_compensated_aperture"]=aperture_0
                aperture=aperture_0>0.5*np.max(aperture_0)
                if plot:
                    plt.imshow(aperture_0)
                    plt.title("estimated aperture")
                    plt.axis("off")
                    plt.show()
        else:
            aperture_shifts_x=aperture_shifts_x.flatten()
            aperture_shifts_y=aperture_shifts_y.flatten()
            aperture=aperture>0.5*np.max(aperture)
    if not(xp==np or smart_memory):
        dataset_h5=cp.asarray(dataset_h5)
    if not(subscan_region is None):
        left_border, top_border, right_border, bottom_border=subscan_region
        if print_flag:
            sys.stdout.write("\n Warning: you will do tcBF on a subscan!")
        this_sequence_tcbf = []
        for dummyi0 in range(top_border, bottom_border, 1):
            for dummyi1 in range(left_border, right_border,1):
                this_sequence_tcbf.append(dummyi1+dummyi0*scan_size[0])
        scan_size[1]=right_border-left_border
        scan_size[0]=bottom_border-top_border
        dataset_h5=cp.asarray(dataset_h5[this_sequence_tcbf])
        if compensate_lowfreq_drift:
            aperture_shifts_x, aperture_shifts_y=aperture_shifts_x[this_sequence_tcbf], aperture_shifts_y[this_sequence_tcbf]
        if not(sequence is None):
            mask_sequence=mask_sequence[top_border:bottom_border, left_border:right_border]
    if not(sequence is None):
        mask_sequence=mask_sequence.flatten()
        dataset_h5=cp.asarray(dataset_h5)
        dataset_h5[mask_sequence.astype(bool), :,:]=cp.mean(dataset_h5[:,aperture])
       # *mask_sequence[:,None,None]
    if interpolate_scan_factor!=1:
        scan_step_A/=interpolate_scan_factor
        upsampled_y, upsampled_x= np.arange(0, scan_size[0], 1/interpolate_scan_factor), np.arange(0, scan_size[1], 1/interpolate_scan_factor)
        new_data=np.zeros((len(upsampled_y)*len(upsampled_x), dataset_h5.shape[1], dataset_h5.shape[2]))
        if print_flag:
            sys.stdout.write("\nUpsampling the scan")
        for dummyi2 in tqdm(range(dataset_h5.shape[1])):
            for dummyi3 in range(dataset_h5.shape[2]):
                this_data=(dataset_h5[:, dummyi2,dummyi3]).reshape(scan_size)
                this_data=RectBivariateSpline(np.arange(scan_size[0]), np.arange(scan_size[1]), this_data, ky=3, kx=3)(upsampled_y, upsampled_x)
                new_data[:, dummyi2,dummyi3]=this_data.flatten()
        dataset_h5=cp.asarray(new_data)
        del new_data
        if compensate_lowfreq_drift:
            aperture_shifts_x = RectBivariateSpline(np.arange(scan_size[0]), np.arange(scan_size[1]), aperture_shifts_x.reshape(scan_size[0], scan_size[1]) , ky=1, kx=1)(upsampled_y, upsampled_x).flatten()
            aperture_shifts_y = RectBivariateSpline(np.arange(scan_size[0]), np.arange(scan_size[1]), aperture_shifts_y.reshape(scan_size[0], scan_size[1]) , ky=1, kx=1)(upsampled_y, upsampled_x).flatten()
        scan_size[0]*=interpolate_scan_factor
        scan_size[1]*=interpolate_scan_factor
        
    if scan_pad is None:
        scan_pad=1+int(np.ceil(np.ceil(pixel_size_x_A*(dataset_h5.shape[2]+2*data_pad)/(scan_step_A))/2))
    padded_scan_size=[scan_size[0]+2*scan_pad, scan_size[1]+2*scan_pad]
    ## create aperture
    if aperture is None:
        aperture=np.ones((dataset_h5.shape[2], dataset_h5.shape[1]))
    else:
        aperture=np.asarray(aperture)
    if print_flag:
        print("shape of data: ", dataset_h5.shape, " scan size: ",scan_size )
    skx, sky=dataset_h5.shape[2], dataset_h5.shape[1]
    skx2, sky2=skx//2, sky//2
    ## create the grids of spatial frequencies
    x_freq_scan_grid, y_freq_scan_grid=np.meshgrid(np.fft.fftfreq(padded_scan_size[1]), np.fft.fftfreq(padded_scan_size[0]), indexing="xy")
    x_freq_scan_grid, y_freq_scan_grid=cp.asarray(x_freq_scan_grid), cp.asarray(y_freq_scan_grid)
    kx_detector_full,ky_detector_full=np.meshgrid(np.arange(-skx2,skx-skx2, 1)*mrad_per_px*1e-3,np.arange(-sky2,sky-sky2, 1)*mrad_per_px*1e-3 , indexing="xy")
    kx_detector_full, ky_detector_full=np.cos(rot_rad)*kx_detector_full+np.sin(rot_rad)*ky_detector_full, -np.sin(rot_rad)*kx_detector_full+np.cos(rot_rad)*ky_detector_full
    kx_detector, ky_detector=kx_detector_full[aperture], ky_detector_full[aperture]
    kx_full_run=np.arange(-skx2,skx-skx2, 1)*mrad_per_px*1e-3
    ky_full_run=np.arange(-sky2,sky-sky2, 1)*mrad_per_px*1e-3
    kx_full_run, ky_full_run=np.cos(rot_rad)*kx_full_run+np.sin(rot_rad)*ky_full_run, -np.sin(rot_rad)*kx_full_run+np.cos(rot_rad)*ky_full_run
    ## create the folders etc
    try:
        os.makedirs(pypty_params["output_folder"], exist_ok=True)
        os.makedirs(pypty_params["output_folder"]+"/tcbf/", exist_ok=True)
    except:
        sys.stdout.write("output folder was not created!")
    ## now we have two options for the CTF fit: either do it on an aberration function aka Zernike-basis (conventional option) or fit a full 2D discretized phase of the beam. The later option IS experimental and I have to finish it!
    
    ## prepare some arrays
    bin_prev=0
    fit_abberations_array=np.zeros((len(binning_for_fit)+1, len(aberrations)))
    fit_abberations_array[0,:]=aberrations
    if optimize_angle:
        fit_angle_array=np.zeros(len(binning_for_fit)+1)
        fit_angle_array[0]=-1*angle_offset
    if print_flag:
        sys.stdout.write("Initializing the abberation fit!")
        sys.stdout.flush()
    ## now we will iterate through provided binning values (binning will happen in diffraction space)
    for index_bin, bin_fac in enumerate(binning_for_fit):
        try:
            cp.fft.config.clear_plan_cache() ## free the memory
            pool.free_all_blocks()
            pinned_pool.free_all_blocks()
        except:
            pass
        if print_flag:
            sys.stdout.write("\n---> Starting iteration %d/%d of the CTF fit, this binning is %d"%(index_bin+1,len(binning_for_fit), bin_fac))
            sys.stdout.flush()
        zeroindex_x, zeroindex_y=skx//2, sky//2
        difference_x_left,  difference_y_left  = (zeroindex_x-int(np.floor(bin_fac/2)))%bin_fac, (zeroindex_y-int(np.floor(bin_fac/2)))%bin_fac
        difference_x_right, difference_y_right =  skx- (skx-zeroindex_x-int(np.ceil(bin_fac/2)))%bin_fac, sky- (sky-zeroindex_y-int(np.ceil(bin_fac/2)))%bin_fac
        new_skx, new_sky=(difference_x_right-difference_x_left)//bin_fac, (difference_y_right-difference_y_left)//bin_fac
        if bin_prev!=bin_fac: ## if we have not yet prepared the data for the binning value do following:
            if print_flag:
                sys.stdout.write("\nBinning the data by %d"%(bin_fac))
                sys.stdout.flush()
            if cpu_mode or not(smart_memory):
                binned_data_bright_field=dataset_h5[:, difference_y_left:difference_y_right, difference_x_left:difference_x_right] ## trim the data
            else:
                binned_data_bright_field=cp.asarray(dataset_h5[:, difference_y_left:difference_y_right, difference_x_left:difference_x_right])
            if compensate_lowfreq_drift: ## if we want to compensate the aperture wobbling, then we roll the patterns!
                for ind111 in range(dataset_h5.shape[0]):
                    shifty, shiftx= int(np.round(aperture_shifts_y[ind111])), int(np.round(aperture_shifts_x[ind111]))
                    pattern_cropped=cp.copy(binned_data_bright_field[ind111])
                    binned_data_bright_field[ind111]=cp.roll(pattern_cropped, (-shifty, -shiftx), axis=(0,1))
            aperture_binned=aperture[difference_y_left:difference_y_right, difference_x_left:difference_x_right] ## trim the aperture
            binned_data_bright_field=cp.sum(binned_data_bright_field.reshape(dataset_h5.shape[0], new_sky,bin_fac, new_skx,bin_fac), axis=(-3, -1)) ## bin the data
            aperture_binned=np.sum(aperture_binned.reshape(new_sky,bin_fac, new_skx,bin_fac), axis=(-3, -1)) ## bin the aperture
            aperture_binned=aperture_binned.astype(bool)
            binned_kx_detector_full=kx_detector_full[difference_y_left:difference_y_right, difference_x_left:difference_x_right] # trim the x-coordinate
            binned_ky_detector_full=ky_detector_full[difference_y_left:difference_y_right, difference_x_left:difference_x_right] # trim the y-coordinate
            binned_kx_detector_full=cp.mean(binned_kx_detector_full.reshape(new_sky,bin_fac, new_skx,bin_fac), axis=(-3, -1)) # bin the x-coordinate
            binned_ky_detector_full=cp.mean(binned_ky_detector_full.reshape(new_sky,bin_fac, new_skx,bin_fac), axis=(-3, -1)) # bin the y-coordinate
            binned_kx_detector, binned_ky_detector=binned_kx_detector_full[aperture_binned], binned_ky_detector_full[aperture_binned]
            binned_data_bright_field=cp.array([d[aperture_binned] for d in binned_data_bright_field]) ## select the pixels in the bright field
            binned_data_bright_field=binned_data_bright_field.reshape(scan_size[0], scan_size[1], binned_data_bright_field.shape[1]) ## reshape
            binned_data_bright_field=cp.pad(binned_data_bright_field, [[scan_pad, scan_pad], [scan_pad, scan_pad], [0,0]]) ## pad with zeros on the sides
            for dummyind in range(binned_data_bright_field.shape[-1]):  # fill the edge values with a "mean" count
                mean=cp.mean(binned_data_bright_field[scan_pad:-scan_pad,scan_pad:-scan_pad,dummyind])
                binned_data_bright_field[:scan_pad,:,dummyind]=mean
                binned_data_bright_field[-scan_pad:,:,dummyind]=mean
                binned_data_bright_field[:,:scan_pad,dummyind]=mean
                binned_data_bright_field[:,-scan_pad:,dummyind]=mean
            zero_freq=np.argmin(binned_kx_detector**2+binned_ky_detector**2) ## find where is your lowest spatial frequency after binning
            binned_data_bright_field_fourier=fft2(binned_data_bright_field, axes=(0,1)) ### fourier transform
            Matrix_shifts_x=np.zeros((len(binned_ky_detector), len(aberrations)))  ## this thing will be needed for aberation fit later
            Matrix_shifts_y=np.zeros((len(binned_ky_detector), len(aberrations)))
            for indmat2 in range(len(aberrations)): ## we prepare a Jacobian for the CTF fit
                thisaberations_delta=np.zeros_like(aberrations)
                thisaberations_delta[indmat2]=1
                D_ctf_grad_x_dab, D_ctf_grad_y_dab=get_ctf_derivatives(thisaberations_delta, binned_kx_detector ,binned_ky_detector, wavelength, angle_offset)
                Matrix_shifts_x[:,indmat2]=D_ctf_grad_x_dab*wavelength/(6.283185307179586*scan_step_A)
                Matrix_shifts_y[:,indmat2]=D_ctf_grad_y_dab*wavelength/(6.283185307179586*scan_step_A)
            if print_flag:
                sys.stdout.write("\nFFT of binned data is done!")
                sys.stdout.flush()
            bin_prev=bin_fac
        else: ## if the preparation was done at the previous iteration, reuse the results!
            if print_flag:
                sys.stdout.write("\nUsing results of previous binning")
                sys.stdout.flush()
        ctf_grad_x, ctf_grad_y=get_ctf_derivatives(aberrations,binned_kx_detector, binned_ky_detector, wavelength, angle_offset)
            
        reference_x, reference_y=ctf_grad_x*wavelength/(6.283185307179586*scan_step_A),ctf_grad_y*wavelength/(6.283185307179586*scan_step_A) ## this are our reference shifts
        reference_shifts=np.zeros((2, reference_x.shape[0]))
        reference_shifts[0]=reference_x
        reference_shifts[1]=reference_y
        if not(cpu_mode):
            reference_x, reference_y = cp.asarray(reference_x), cp.asarray(reference_y)
        kernel=cp.exp(-6.283185307179586j*(reference_x[None, None,:] * x_freq_scan_grid[:,:,None]+ reference_y[None, None,:] * y_freq_scan_grid[:,:,None])) ## here we create a shift kernel to generate a tcBF image
        image_bf_binned_fourier=(cp.sum(binned_data_bright_field_fourier*kernel, -1)) ## align the pixel images
        # now we have to decide with what will the reference shifts be compared:
        if reference_type=="bf": # option 1: cross-corelate the individual pixel images with a tcBF estimate
            refence=image_bf_binned_fourier
        else: # option 2: cross-corelate the individual pixel images with an image corresponding to the lowest spatial frequency
            refence=binned_data_bright_field_fourier[:,:, zero_freq]
        if cross_corr_type=="phase":
            full_cross_corr=cp.fft.fftshift(ifft2(cp.exp(-1j*cp.angle(binned_data_bright_field_fourier*cp.conjugate(refence)[:,:,None])), axes=(0,1)), axes=(0,1)) ## phase cross correlation
        else:
            full_cross_corr=cp.fft.fftshift(ifft2( cp.conjugate(binned_data_bright_field_fourier)*refence[:,:,None]  , axes=(0,1)), axes=(0,1)) ## phase cross correlation
        if not cpu_mode:
            reference_x=reference_x.get()
            reference_y=reference_y.get()
        if (plot and plot_inter_image) or save_inter_imags: ## plot the tcBF estimate
            image_bf_binned_plot=cp.real(ifft2(image_bf_binned_fourier))
            if not(cpu_mode):
                image_bf_binned_plot=image_bf_binned_plot.get()
            plt.imshow(image_bf_binned_plot, cmap="gray")
            plt.title("tcBF image at bin %d. Iteration %d"%(bin_fac, index_bin))
            plt.axis("off")
            if save_inter_imags:
                plt.savefig(pypty_params["output_folder"]+"/tcbf/"+str(index_bin)+"png", dpi=200)
                np.save(pypty_params["output_folder"]+"/tcbf/tcbf_"+str(index_bin)+".npy", image_bf_binned_plot)
            if not(plot and plot_inter_image):
                plt.close()
            else:
                plt.show()
        estimated_shifts=np.zeros((2, binned_data_bright_field_fourier.shape[-1])) ## now we have to find the peaks in the correlations
        total=binned_data_bright_field_fourier.shape[-1]
        success=np.zeros(total, dtype=bool)
        for dummyind in range(total):
            this_cross_corr=full_cross_corr[:,:,dummyind] ## get a correlation between the reference image and a pixel image "dummyind"
            this_cross_corr_abs=cp.real(this_cross_corr)
            if binning_cross_corr==1:
                indy, indx=cp.unravel_index(this_cross_corr_abs.argmax(), this_cross_corr_abs.shape) ## find maximum
            else:
                sh0_cc, sh1_cc=(this_cross_corr_abs.shape[0])//binning_cross_corr, (this_cross_corr_abs.shape[1])//binning_cross_corr
                this_cross_corr_abs_binned=this_cross_corr_abs[:binning_cross_corr*sh0_cc, :binning_cross_corr*sh1_cc]
                this_cross_corr_abs_binned=cp.sum(this_cross_corr_abs_binned.reshape(sh0_cc, binning_cross_corr, sh1_cc, binning_cross_corr),(1,3))
                indy, indx=cp.unravel_index(this_cross_corr_abs_binned.argmax(), this_cross_corr_abs_binned.shape)
                indy=indy*binning_cross_corr+binning_cross_corr//2
                indx=indx*binning_cross_corr+binning_cross_corr//2
               # this_cross_corr_abs_cropped=this_cross_corr_abs[indy*binning_cross_corr-binning_cross_corr:indy*binning_cross_corr+binning_cross_corr, indx*binning_cross_corr-binning_cross_corr:indx*binning_cross_corr+binning_cross_corr]
               # new_indy,new_indx=cp.unravel_index(this_cross_corr_abs_cropped.argmax(), this_cross_corr_abs_cropped.shape)
               # indy=new_indy+indy*binning_cross_corr-binning_cross_corr
                #indx=new_indx+indx*binning_cross_corr-binning_cross_corr
            ## now we have to remember that the our scan grid is relatively sparce, i.e. the argmax index is not really exact, so we have to refine it
            chopped_cross_corr=this_cross_corr[indy-refine_box_dim:indy+refine_box_dim+1, indx-refine_box_dim:indx+refine_box_dim+1] ## crop a small "box" around the maximum of the correlation and try to upsample it via interpolation
            if cross_corr_type=="phase" and phase_cross_corr_formula: ## for phase cross corr, the output is a 2D sinc function. We can find its maximum analyticaly, for more info see H. Foroosh et al. "Extension of Phase Correlation to Subpixel Registration"
                peak_center=this_cross_corr_abs[indy, indx]
                test_peak_left=this_cross_corr_abs[indy, indx-1]
                test_peak_right=this_cross_corr_abs[indy, indx+1]
                test_peak_top=this_cross_corr_abs[indy-1, indx]
                test_peak_bottom=this_cross_corr_abs[indy+1, indx]
                if test_peak_right>test_peak_left:
                    shift_x=test_peak_right /(test_peak_right+ peak_center)
                    if np.abs(shift_x)>1: shift_x=test_peak_right/(test_peak_right-peak_center);
                else:
                    if test_peak_right<test_peak_left:
                        shift_x=test_peak_left/(test_peak_left + peak_center)
                        if np.abs(shift_x)>1: shift_x=test_peak_left/(test_peak_left - peak_center);
                    else:
                        shift_x=0
                if test_peak_bottom>test_peak_top:
                    shift_y=test_peak_bottom/(test_peak_bottom + peak_center)
                    if np.abs(shift_y)>1: shift_y=test_peak_bottom/(test_peak_bottom - peak_center);
                else:
                    if test_peak_bottom<test_peak_top:
                        shift_y=test_peak_top/(peak_center+ test_peak_top)
                        if np.abs(shift_y)>1: shift_y=test_peak_top/(-peak_center+ test_peak_top);
                    else:
                        shift_y=0
                shift_x=indx-shift_x-padded_scan_size[1]//2
                shift_y=indy-shift_y-padded_scan_size[0]//2
                success[dummyind]=True ## if everything is okay, make a note about success!
            else:
                if not cpu_mode:
                    chopped_cross_corr=chopped_cross_corr.get()
                x_old, y_old=np.meshgrid(np.arange(-refine_box_dim,refine_box_dim+1,1), np.arange(-refine_box_dim,refine_box_dim+1,1), indexing="xy")
                x_new, y_new=np.meshgrid(np.arange(-refine_box_dim, refine_box_dim+0.1/upsample,1/upsample), np.arange(-refine_box_dim,refine_box_dim+0.1/upsample,1/upsample), indexing="xy")
                try:
                    interp_cross_corr=np.abs(griddata((x_old.flatten(), y_old.flatten()), chopped_cross_corr.flatten(), (x_new, y_new), method='cubic', fill_value=0)) ## i know that this should be real, but somehow abs is more stable on the gpu--> bug to be solved!
                    refined_indy, refined_indx=np.unravel_index(interp_cross_corr.argmax(), interp_cross_corr.shape) ## this argmax is much more precise!
                    new_indy=indy+y_new[refined_indy, refined_indx]
                    new_indx=indx+x_new[refined_indy, refined_indx]
                    shift_y=new_indy-padded_scan_size[0]//2 ## now we compute the shift between the reference and pixel image
                    shift_x=new_indx-padded_scan_size[1]//2
                    success[dummyind]=True ## if everything is okay, make a note about success!
                except:
                    shift_y, shift_x=0,0
                    success[dummyind]=False
            estimated_shifts[0,dummyind]=shift_x
            estimated_shifts[1,dummyind]=shift_y
            if print_flag:
                if print_flag==2:
                    sys.stdout.write("\rFitting the shifts: %d/%d. shift y: %.2f, shift x: %.2f...."%(dummyind+1, total, shift_y, shift_x))
                if print_flag>2:
                    sys.stdout.write("\nFitting the shifts: %d/%d. shift y: %.2f, shift x: %.2f...."%(dummyind+1, total, shift_y, shift_x))
                sys.stdout.flush()
        if not(cancel_large_shifts is None): ## now it might be that for a particular pixel the reference (grad of the CTF) and the cross correlation shifts differ way to much. It might ruin the fit. Thus, we can ignore this pixel at this particular itaretion and come back later!
            denom=np.sum((reference_shifts)**2, axis=0)
            nom=np.sum((estimated_shifts-reference_shifts)**2, axis=0)
            nom[denom==0]=0
            denom[denom==0]=1
            radial_shifts_difference= nom /denom
            threshold=np.percentile(radial_shifts_difference, q=cancel_large_shifts*100)
            above_threshold= radial_shifts_difference>= threshold
            success[above_threshold]=False
        if print_flag:
            sys.stdout.write("\nFound matching shifts for %d/%d pixels.\n"%(np.sum(success), total)) ## success is the number of binned bright field pixels for which we successfully interpolated the crosscorr, found maximum and the resulting shift is not too far away from what we have expected!
        binned_kx_detector_suc,binned_ky_detector_suc=binned_kx_detector[success],binned_ky_detector[success]
        estimated_shifts=estimated_shifts[:,success]
        #estimated_shifts[0,:]-=np.mean(estimated_shifts[0,:])
        #estimated_shifts[1,:]-=np.mean(estimated_shifts[1,:])
       # estimated_shifts=np.round(estimated_shifts, 2)
        
        def ctf_residuals(this_guess): # define the residuals
            nonlocal binned_kx_detector_suc,binned_ky_detector_suc, estimated_shifts, wavelength, optimize_angle, upsample, phase_cross_corr_formula
            if optimize_angle: ## experimental
                aberrations, angle_offset=this_guess[:-1], this_guess[-1]
            else:
                aberrations, angle_offset=this_guess, 0
            ctf_grad_x, ctf_grad_y=get_ctf_derivatives(aberrations, binned_kx_detector_suc, binned_ky_detector_suc, wavelength, angle_offset)
            this_shifts_x=ctf_grad_x*wavelength/(6.283185307179586*scan_step_A)
            this_shifts_y=ctf_grad_y*wavelength/(6.283185307179586*scan_step_A)
            if not(phase_cross_corr_formula): ### this rounds the residuals, so jacobian is not true anymore, but it also prevents fitting super high values for higher aberrations. It is what it is..
                this_shifts_x=(np.round(this_shifts_x*upsample,0))/upsample
                this_shifts_y=(np.round(this_shifts_y*upsample,0))/upsample
            dif_x=this_shifts_x-estimated_shifts[0,:]
            dif_y=this_shifts_y-estimated_shifts[1,:]
            return np.asarray([[dif_x], [dif_y]]).ravel()
        shape=(estimated_shifts.shape[1])
        if not(optimize_angle):
            final_mat=np.zeros((shape*2,Matrix_shifts_x.shape[1]))
            final_mat[:shape,:]=Matrix_shifts_x[success,:]
            final_mat[shape:,:]=Matrix_shifts_y[success,:]
        else:
            final_mat=None
        def loss_ctf_residuals(z): ## this function is not used currently, but i may change it in the future
            nonlocal upsample, phase_cross_corr_formula
            z_1=z**0.5
            if not(phase_cross_corr_formula):
                z_2=(np.round(z*upsample,0))/upsample
            z_3=z_2**2
            l0=z_3 ## loss, actually false -> to be updated
            l1=z_3 ## first derivative, actually false -> to be updated
            l2=z_3 ## second derivative, actually false -> to be updated
            return np.vstack(((l0,l1),l2))
        
        def jacobian_residuals(x): ## Jacobian
            nonlocal final_mat, binned_kx_detector_suc, binned_ky_detector_suc, wavelength
            if final_mat is None:
                aberrations=x[:-1]
                angle_offset=x[-1]
                Matrix_shifts_x=np.zeros((len(binned_kx_detector_suc), len(aberrations)))
                Matrix_shifts_y=np.zeros((len(binned_kx_detector_suc), len(aberrations)))
                for indmat2 in range(len(aberrations)):
                    thisaberations_delta=np.zeros_like(aberrations)
                    thisaberations_delta[indmat2]=1
                    D_ctf_grad_x_dab, D_ctf_grad_y_dab=get_ctf_derivatives(thisaberations_delta, binned_kx_detector_suc ,binned_ky_detector_suc, wavelength, angle_offset)
                    Matrix_shifts_x[:,indmat2]=D_ctf_grad_x_dab*wavelength/(6.283185307179586*scan_step_A)
                    Matrix_shifts_y[:,indmat2]=D_ctf_grad_y_dab*wavelength/(6.283185307179586*scan_step_A)
                angle_gradient_x, angle_gradient_y=get_ctf_gradient_rotation_angle(aberrations, binned_kx_detector_suc, binned_ky_detector_suc, wavelength, angle_offset)
                shape=len(binned_ky_detector_suc)
                final_mat=np.zeros((shape*2, len(x)))
                final_mat[:shape,:-1]=Matrix_shifts_x
                final_mat[shape:,:-1]=Matrix_shifts_y
                final_mat[:shape,-1]=angle_gradient_x
                final_mat[shape:,-1]=angle_gradient_y
            return final_mat
            
        if optimize_angle:
            start_x=np.hstack((aberrations, angle_offset))
        else:
            start_x=aberrations
        result=least_squares(ctf_residuals,start_x, jac=jacobian_residuals, x_scale=x_scale_lsq, loss=loss_lsq, f_scale=f_scale_lsq, ftol=tol_ctf) ## do least squares!
        aberrations= np.asarray(result.x)
        if save_iteration_results:
            np.save(pypty_params["output_folder"]+"/tcbf/estimated_shifts_%d.npy"%index_bin, estimated_shifts)
            np.save(pypty_params["output_folder"]+"/tcbf/aberrations_%d.npy"%index_bin, aberrations)
        if optimize_angle:
            angle_offset=aberrations[-1]
            aberrations=aberrations[:-1]
            fit_angle_array[index_bin+1]=-1*angle_offset
        fit_abberations_array[index_bin+1,:]=aberrations
        if print_flag:
            sys.stdout.write("\nCTF fitted successfully: %s."%(result.success))
            sys.stdout.flush()
        if plot and plot_CTF_shifts: ## plot the results
            ctf_grad_x, ctf_grad_y=get_ctf_derivatives(aberrations, binned_kx_detector_suc,binned_ky_detector_suc,  wavelength, angle_offset)
            fig, ax=plt.subplots(1,2, figsize=(10,5))
            ap_show=rotate(aperture_binned, angle=0, axes=(1, 0), reshape=False, order=0)
            ax[0].imshow(ap_show, cmap="gray", extent=[np.min(binned_kx_detector_full),np.max(binned_kx_detector_full), np.min(binned_ky_detector_full), np.max(binned_ky_detector_full)])
            ax[0].quiver(binned_kx_detector_suc,binned_ky_detector_suc, estimated_shifts[0,:], estimated_shifts[1,:],  color="red", capstyle="round")
            ax[1].imshow(ap_show, cmap="gray", extent=[np.min(binned_kx_detector_full), np.max(binned_kx_detector_full), np.min(binned_ky_detector_full), np.max(binned_ky_detector_full)])
            ax[1].quiver(binned_kx_detector_suc,binned_ky_detector_suc, ctf_grad_x, ctf_grad_y,  color="red", capstyle="round")
            ax[0].set_title("Fitted shifts")
            ax[1].set_title("Fitted CTF grad")
            plt.show()
        if print_flag:
            num_abs=len(aberrations)
            possible_n, possible_m, possible_ab=convert_num_to_nmab(num_abs)
            aber_print, s=nmab_to_strings(possible_n, possible_m, possible_ab), ""
            for i in range(len(aberrations)): s+=aber_print[i]+" %.2e A, "%aberrations[i];
            sys.stdout.write("\nFitted aberrations: %s"%s[:-2])
            if optimize_angle:
                sys.stdout.write("\nFitted PL rot angle: %.2f deg"%(-1*(angle_offset)*180/np.pi))
        if not(cpu_mode):
            cp.fft.config.clear_plan_cache()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    
    if print_flag:
        sys.stdout.write("\nFinal CTF Fit done!")
    
    if save_iteration_results:
        np.save(pypty_params["output_folder"]+"/tcbf/aberrations_A.npy", fit_abberations_array)
        if optimize_angle:
            np.save(pypty_params["output_folder"]+"/tcbf/PL_angle_deg.npy", (fit_angle_array)*180/np.pi)
    
    if plot:
        num_abs=len(aberrations)
        possible_n, possible_m, possible_ab=convert_num_to_nmab(num_abs)
        leg=nmab_to_strings(possible_n, possible_m, possible_ab)
        fig, ax=plt.subplots(len(aberrations),1,figsize=(10, 2*len(aberrations)))
        if len(aberrations)==1:
            ax=[ax]
        for index_aberr in range(len(aberrations)):
            ax[index_aberr].plot(fit_abberations_array[:,index_aberr],"-.", linewidth=2, label=leg[index_aberr])
            ax[index_aberr].legend(loc=1)
            ax[index_aberr].set_xlabel("iteration")
            ax[index_aberr].set_ylabel("value")
        if save_iteration_results:
            fig.savefig(pypty_params["output_folder"]+"/tcbf/aberrations_fit.png")
        plt.show()
        if optimize_angle:
            fig, ax=plt.subplots(figsize=(10, 2))
            ax.plot((fit_angle_array)*180/np.pi, "-.", linewidth=2, label="angle offset")
            ax.set_xlabel("iteration")
            ax.set_ylabel("angle (deg)")
            if save_iteration_results:
                fig.savefig(pypty_params["output_folder"]+"/tcbf/angle_fit.png")
            plt.show()
    del binned_data_bright_field
    try:
        cp.fft.config.clear_plan_cache()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except:
        pass
    pypty_params["extra_probe_defocus"]=0
    
    if optimize_angle:
        old_pl_rot   = pypty_params["PLRotation_deg"]
        new_pl_rot   = -1*(angle_offset)*180/np.pi
        old_postions = pypty_params["positions"]
        
        opy, opx=old_postions[:,0], old_postions[:,1]
        rot_ang=-1*(new_pl_rot-old_pl_rot) * np.pi/180
        opx_prime, opy_prime=opx * np.cos(rot_ang) + opy * np.sin(rot_ang), -1*opx * np.sin(rot_ang) + opy * np.cos(rot_ang)
        opx_prime-=np.min(opx_prime)
        opy_prime-=np.min(opy_prime)
        
        old_postions[:,1]=opx_prime
        old_postions[:,0]=opy_prime
        pypty_params["PLRotation_deg"]=new_pl_rot
        pypty_params["positions"]=old_postions
   
    pypty_params["aberrations"]=aberrations
    pypty_params["beam_ctf"]=None
    pypty_params["probe"]=None
    try:
        f.close()
    except:
        pass
    return pypty_params

def simple_wdd(pypty_params, mean_pattern=None, eps_wiener=1e-3, thresh=None):
    global cpu_mode
    if not(cpu_mode):
        cp.fft.config.clear_plan_cache()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    data_path = pypty_params.get('data_path', "")
    scan_size=np.copy(np.array(pypty_params.get('scan_size', [0,0])))
    if data_path[-3:]==".h5":
        data=h5py.File(data_path, "r")["data"]
        data=np.asarray(data)
    elif data_path[-4:]==".npy":
        data=np.load(data_path)
        if len(data.shape)==4:
            scan_size=[data.shape[0], data.shape[1]]
            data=data.reshape(data.shape[0]* data.shape[1], data.shape[2],data.shape[3])
    
    data_pad = pypty_params.get('data_pad', 0)
    acc_voltage=pypty_params.get('acc_voltage', 60)
    probe=pypty_params.get('probe', None)
    aberrations = pypty_params.get('aberrations', [0])
    pixel_size_x_A = pypty_params.get('pixel_size_x_A', 1)
    pixel_size_y_A = pypty_params.get('pixel_size_y_A', 1)
    
    pixel_size_x_A*=(data.shape[2]+2*data_pad)/data.shape[2]
    pixel_size_y_A*=(data.shape[1]+2*data_pad)/data.shape[1]
    scan_step_A= pypty_params.get('scan_step_A', 1)
    PLRotation_deg=pypty_params.get('PLRotation_deg', 0)
    rez_pixel_size_A=pypty_params.get("rez_pixel_size_A", 1)
    rot_scan_grid_rad=-np.pi*PLRotation_deg/180 ## real-space coordinate rotation!
    data=cp.asarray(np.asarray(data).reshape(scan_size[0], scan_size[1], data.shape[1], data.shape[2]).astype(np.complex64))
    #if scan_pad is None:
     #   scan_pad=1+int(np.ceil(np.ceil(pixel_size_x_A*(data.shape[2]+2*data_pad)/(scan_step_A))/2))
    window =  pypty_params.get('window', None)
    extra_probe_defocus=pypty_params.get('extra_probe_defocus', 0)
    wavelength=12.4 /((2*511.0+acc_voltage)*acc_voltage)**0.5
    if mean_pattern is None:
        mean_pattern=cp.asarray(np.mean(data[:100,:100], axis=(0,1)))
    if probe is None:
        probe=cp.expand_dims(cp.fft.fftshift(ifft2_ishift(cp.sqrt(mean_pattern))),-1)
    if extra_probe_defocus!=0: probe=apply_defocus_probe(probe, extra_probe_defocus,acc_voltage, pixel_size_x_A, pixel_size_y_A,cp.complex64, cp.float32, cp);
    probe=probe[:,:,0]
    if not(aberrations is None):
        num_abs=len(aberrations)
        possible_n, possible_m, possible_ab=convert_num_to_nmab(num_abs)
        aber_print, s=nmab_to_strings(possible_n, possible_m, possible_ab), ""
        for i in range(len(aberrations)): s+=aber_print[i]+": %.2e Å; "%aberrations[i];
        sys.stdout.write("\nProvided aberrations: %s"%s[:-1])
        kx=cp.fft.fftshift(cp.fft.fftfreq(data.shape[3], pixel_size_x_A))*wavelength
        ky=cp.fft.fftshift(cp.fft.fftfreq(data.shape[2], pixel_size_y_A))*wavelength
        kx, ky=cp.meshgrid(kx,ky, indexing="xy")
        ctf= cp.asarray(get_ctf(aberrations, kx, ky, wavelength))
        probe=ifft2_ishift(shift_fft2(probe)*cp.exp(-1j*ctf))
    def flatten_edge(x):
        smooth=np.ones_like(x)
        maxval=0.6*cp.max(x)
        edge_len=len(x[x>maxval])
        term=0.5*(1+cp.cos(cp.pi*cp.arange(edge_len)/(edge_len-1)))
        smooth[-edge_len:]*=term
        edge_len=len(x[-x>maxval])
        term=0.5*(1+cp.cos(cp.pi*cp.arange(edge_len)/(edge_len-1)))[::-1]
        smooth[:edge_len]*=term
        return smooth
    if window is None:
        window=get_window(mean_pattern.shape[1], mean_pattern.shape[1]*0.4, mean_pattern.shape[1]*0.45)
    kysh=data.shape[2]
    kxsh=data.shape[3]
    kx,ky=cp.meshgrid(cp.fft.fftshift(cp.fft.fftfreq(data.shape[3], pixel_size_x_A)),cp.fft.fftshift(cp.fft.fftfreq(data.shape[2], pixel_size_y_A)) , indexing="xy")
    qx, qy=cp.meshgrid(cp.fft.fftshift(cp.fft.fftfreq(data.shape[1], scan_step_A)),cp.fft.fftshift(cp.fft.fftfreq(data.shape[0], scan_step_A)) , indexing="xy")
    qx_rot, qy_rot=cp.cos(rot_scan_grid_rad)*qx+cp.sin(rot_scan_grid_rad)*qy, -cp.sin(rot_scan_grid_rad)*qx+cp.cos(rot_scan_grid_rad)*qy
    rho_x, rho_y=cp.fft.fftshift(cp.fft.fftfreq(kxsh)), cp.fft.fftshift(cp.fft.fftfreq(kysh))
    smopth_rho_x, smopth_rho_y=flatten_edge(rho_x), flatten_edge(rho_y)
    rho_x*=smopth_rho_x
    rho_y*=smopth_rho_y
    smooth_rho=cp.prod(cp.array(cp.meshgrid(smopth_rho_x, smopth_rho_y)), axis=0)
    rho_x, rho_y=cp.meshgrid(rho_x, rho_y, indexing="xy")
    ap_conj=cp.conjugate(shift_fft2(probe))
    if not(thresh is None):
        thresh=thresh*np.max(cp.abs(ap_conj))**2*(1/np.prod(ap_conj.shape))
    def shift_aperuture(probe, shx, shy, rho_x, rho_y):
        nonlocal smooth_rho
        kernel=smooth_rho*cp.exp(6.283185307179586j*(rho_x*shx + rho_y*shy))
        ap_shift=shift_fft2(probe*kernel)
        pix_shift_x=int(np.round(shx))
        pix_shift_y=int(np.round(shy))
        if pix_shift_y<0:
            ap_shift[pix_shift_y:,:]=0
        if pix_shift_y>0:
            ap_shift[:pix_shift_y,:]=0
        if pix_shift_x<0:
            ap_shift[:,pix_shift_x:]=0
        if pix_shift_x>0:
            ap_shift[:,:pix_shift_x]=0
        return ap_shift
    def get_probe_wd_ij(i,j):
        nonlocal ap_conj,probe, qx_rot, qy_rot, rho_x, rho_y, rez_pixel_size_A
        ap_shift=shift_aperuture(probe, qx_rot[i,j]/rez_pixel_size_A, qy_rot[i,j]/rez_pixel_size_A, rho_x, rho_y)
        return ifft2_ishift(ap_conj*ap_shift)
    if not(cpu_mode):
        cp.fft.config.clear_plan_cache()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    data=data.astype(cp.complex64)   ## y,x,k_y,k_x
    for i in tqdm(range(data.shape[2])):
        for j in range(data.shape[3]):
            dd=data[:,:,i,j]
            dd=shift_fft2(dd) # qy, qx,ky,kx, dataset G
            data[:,:,i,j]=dd
    if not(cpu_mode):
        cp.fft.config.clear_plan_cache()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    sys.stdout.write("\n-->1st FFT done")
    for i in tqdm(range(data.shape[0])):
        for j in range(data.shape[1]):
            dd=data[i,j,:,:]
            dd=ifft2_ishift(dd) #qy, qx,r_y,r_x, dataset H
            wdij=cp.asarray(get_probe_wd_ij(i,j))
            if thresh is None:
                dd=dd*cp.conjugate(wdij)/(eps_wiener+cp.abs(wdij)**2)
            else:
                where=cp.abs(wdij)>=thresh
                dd[where]=dd[where]/wdij[where]
                dd[(1-where).astype(bool)]=0
            data[i,j,:,:]=dd # Chi_o, qy,qx, rho_y, rho_x
    if not(cpu_mode):
        cp.fft.config.clear_plan_cache()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    sys.stdout.write("\n-->2nd FFT done")
    min_k_ind=cp.unravel_index(cp.argmin(kx**2+ky**2),[data.shape[2],data.shape[3]])
    min_q_ind=cp.unravel_index(cp.argmin(qx_rot**2+qy_rot**2),[data.shape[0],data.shape[1]])
    o=cp.sum(data, axis=(2,3)) ## summing is equivalent to taking the zero freq component of the FFT with respect to ky,kx
    prefactor=cp.sqrt(cp.abs(o[min_q_ind[0],min_q_ind[1]]))
    o=o/prefactor
    o=ifft2_ishift(o)
    return o, probe



def phase_cross_corr_align(im_ref_fft, im_2_fft, refine_box_dim, upsample, x_real, y_real, shift_y=None, shift_x=None):
    if shift_y is None or shift_x is None:
        ## find the phase cross-correlation
        phase_cross_corr=np.exp(-1j*np.angle(im_2_fft*np.conjugate(im_ref_fft)))
        phase_cross_corr=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift((phase_cross_corr))))
        phase_cross_corr_abs=np.abs(phase_cross_corr)
        ##----> interpolate for sub-pixel shift
        indy, indx=np.unravel_index(phase_cross_corr_abs.argmax(), phase_cross_corr_abs.shape)
        chopped_cross_corr=phase_cross_corr[indy-refine_box_dim:indy+refine_box_dim+1, indx-refine_box_dim:indx+refine_box_dim+1]
        x_old, y_old=np.meshgrid(np.arange(-refine_box_dim,refine_box_dim+1,1), np.arange(-refine_box_dim,refine_box_dim+1,1), indexing="xy")
        x_new, y_new=np.meshgrid(np.arange(-refine_box_dim, refine_box_dim+0.1/upsample,1/upsample), np.arange(-refine_box_dim,refine_box_dim+0.1/upsample,1/upsample), indexing="xy")
        interp_cross_corr=np.abs(griddata((x_old.flatten(), y_old.flatten()), chopped_cross_corr.flatten(), (x_new, y_new), method='cubic', fill_value=0))
        refined_indy, refined_indx=np.unravel_index(interp_cross_corr.argmax(), interp_cross_corr.shape)
        new_indy, new_indx=indy+y_new[refined_indy, refined_indx], indx+x_new[refined_indy, refined_indx]
        shift_y, shift_x=new_indy-im_2_fft.shape[0]//2, new_indx-im_2_fft.shape[1]//2
    im_2_fft_shifted=im_2_fft*np.exp(-2j*np.pi*(x_real*shift_x+y_real*shift_y))
    return im_2_fft_shifted
  
  
def get_focussed_probe_from_vacscan(pypty_params, mean_pat_vaccum):
    data_path=pypty_params["data_path"]
    upsample_pattern= pypty_params["upsample_pattern"]
    if data_path[-3:]==".h5":
        f=h5py.File(data_path, "r")["data"]
        meanpat_f=np.mean(f, axis=0)
    elif data_path[-4:]==".npy":
        meanpat_f=np.load(data_path)
        if len(meanpat_f.shape)==4:
            meanpat_f=np.mean(meanpat_f, (0,1))
        else:
            meanpat_f=np.mean(f, axis=0)
    x,y=np.fft.fftshift(np.fft.fftfreq(meanpat_f.shape[1])), np.fft.fftshift(np.fft.fftfreq(meanpat_f.shape[0]))
    x,y=np.meshgrid(x,y, indexing="xy")
    mean_pat_vaccum_fft=np.fft.fftshift(np.fft.fft2(mean_pat_vaccum))
    mean_pat_data_fft  =np.fft.fftshift(np.fft.fft2(meanpat_f))
    mean_pat_vaccum_fft=phase_cross_corr_align(mean_pat_data_fft, mean_pat_vaccum_fft, refine_box_dim=1, upsample=1, x_real=x, y_real=y)
    mean_pattern_vacuum=np.fft.ifft2(np.fft.ifftshift(mean_pat_vaccum_fft))
    mean_pattern_vacuum=np.abs(mean_pattern_vacuum)
    mean_pattern_vacuum=upsample_something(mean_pattern_vacuum, upsample_pattern, True, np)
    focussed_probe_aligned=np.expand_dims(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift((mean_pattern_vacuum**0.5)))),-1)
    return focussed_probe_aligned

def get_approx_beam_tilt(pypty_params, power=3, make_binary=False, com_mask=None, percentile_filter_value=None,percentile_filter_size=10 ):
    dataset_h5=pypty_params.get("data_path", "")
    pixel_size_x_A=pypty_params.get("pixel_size_x_A", 1)
    rez_pixel_size_A=pypty_params.get("rez_pixel_size_A", 1)
    acc_voltage=pypty_params.get("acc_voltage", 60)
    scan_size=pypty_params.get("scan_size", None)
    if dataset_h5[-3:]==".h5":
        dataset_h5=h5py.File(dataset_h5,  "r")["data"]
    elif dataset_h5[-4:]==".npy":
        dataset_h5=np.load(dataset_h5)
        if len(dataset_h5.shape)==4:
            scan_size=[dataset_h5.shape[0],dataset_h5.shape[1]]
            dataset_h5=dataset_h5.reshape(dataset_h5.shape[0]*dataset_h5.shape[1],dataset_h5.shape[2], dataset_h5.shape[3] )
    plot=pypty_params.get("plot", False)
    print_flag=pypty_params.get("print_flag", False)
    sequence=pypty_params.get("sequence", None)
    if print_flag:
        sys.stdout.write("\n******************************************************************************\n*********************** Estimating the beam tilt. ****************************\n******************************************************************************\n")
    if make_binary:
        mean_val=np.mean(dataset_h5)
        dataset_h5=np.array(dataset_h5)> mean_val*make_binary
        if plot:
            plt.imshow(np.mean(dataset_h5, 0))
            plt.axis("off")
            plt.show()
    if com_mask is None:
        com_mask=pypty_params.get("com_mask", None)
    comx=pypty_params.get("aperture_shifts_x", pypty_params.get("comx", None))
    comy=pypty_params.get("aperture_shifts_y", pypty_params.get("comy", None))
    
    if comx is None or comy is None:
        x, y=np.arange(0, dataset_h5.shape[2]), np.arange(0, dataset_h5.shape[1])
        x, y=(x-np.mean(x)), (y-np.mean(y))
        x,y=np.meshgrid(x,y, indexing="xy")
        if sequence is None:
            ssum=np.sum(dataset_h5, axis=(1,2))
            comx=np.sum(dataset_h5*x[None, :,:], axis=(1,2))/ssum
            comy=np.sum(dataset_h5*y[None, :,:], axis=(1,2))/ssum
            pypty_params["comx"]=comx
            pypty_params["comy"]=comy
        else:
            comx, comy=np.zeros(dataset_h5.shape[0]), np.zeros(dataset_h5.shape[0])
            sequence=np.sort(np.array(sequence))
            ssum=np.sum(dataset_h5[sequence], axis=(1,2))
            comx[sequence]=np.sum(dataset_h5[sequence]*x[None, :,:], axis=(1,2))/ssum
            comy[sequence]=np.sum(dataset_h5[sequence]*y[None, :,:], axis=(1,2))/ssum
    else:
        comx=comx.flatten()
        comy=comy.flatten()
    comx=comx.astype(np.float32)
    comy=comy.astype(np.float32)
    
    positions=pypty_params["positions"]*pixel_size_x_A
    x=positions[:,1]
    y=positions[:,0]
    if power!=np.inf and power!="inf":
        X=[]
        for i in range(power+1):
            for j in range(power+1):
                if i+j<power:
                    X.append((x**i) * (y**j))
        X=np.swapaxes(np.array(X),0,1)
        X_full=np.copy(X)
        if not(com_mask is None):
            com_mask=com_mask.flatten()
            X=X[com_mask,:]
            cropped_comx=comx[com_mask]
            cropped_comy=comy[com_mask]
        else:
            cropped_comx=comx
            cropped_comy=comy
        coefficients_comx, residuals, rank, s = np.linalg.lstsq(X, cropped_comx, rcond=None)
        coefficients_comy, residuals, rank, s = np.linalg.lstsq(X, cropped_comy, rcond=None)
        fitted_comx=np.zeros_like(x)
        fitted_comy=np.zeros_like(y)
        for i in range(len(coefficients_comy)):
            fitted_comx+=coefficients_comx[i]*X_full[:,i]
            fitted_comy+=coefficients_comy[i]*X_full[:,i]
    else:
        sys.stdout.write("WARNING: COMs were taken directly from data without fit!")
        fitted_comx=np.copy(comx)
        fitted_comy=np.copy(comy)
    if plot:
        plt.imshow(comx.reshape(scan_size[0], scan_size[1]), cmap="Spectral")
        plt.colorbar()
        plt.title("True COMx")
        plt.show()
        plt.imshow(comy.reshape(scan_size[0], scan_size[1]),cmap="Spectral")
        plt.colorbar()
        plt.title("True COMy")
        plt.show()
        plt.imshow(fitted_comx.reshape(scan_size[0], scan_size[1]),cmap="Spectral")
        plt.colorbar()
        plt.title("Fitted COMx")
        plt.show()
        plt.imshow(fitted_comy.reshape(scan_size[0], scan_size[1]),cmap="Spectral")
        plt.colorbar()
        plt.title("Fitted COMy")
        plt.show()
        plt.imshow(np.abs(fitted_comx-comx).reshape(scan_size[0], scan_size[1]),cmap="Spectral")
        plt.colorbar()
        plt.title("Difference COMx (abs)")
        plt.show()
        plt.imshow(np.abs(fitted_comy-comy).reshape(scan_size[0], scan_size[1]),cmap="Spectral")
        plt.colorbar()
        plt.title("Difference COMy (abs)")
        plt.show()
    ### convert to rads
    l=12.4/np.sqrt(acc_voltage*(acc_voltage+2*511))
    rad_per_px=rez_pixel_size_A*l
    fitted_comx*=rad_per_px
    fitted_comy*=rad_per_px
    tilts=np.zeros((positions.shape[0], 6))
    if not percentile_filter_value is None:
        fitted_comy=cpu_percentile(fitted_comy, percentile_filter_value, percentile_filter_size)
        fitted_comx=cpu_percentile(fitted_comx, percentile_filter_value, percentile_filter_size)
    tilts[:,4]=fitted_comy
    tilts[:,5]=fitted_comx
    pypty_params["tilt_mode"]=1
    pypty_params["tilts"]=tilts
    return pypty_params


def compensate_pattern_drift(aperture, patterns):
    aperture=np.fft.fftshift(np.fft.fft2(aperture))
    shy,shx=aperture.shape
    for i in range(patterns.shape[0]):
        for j in range(patterns.shape[1]):
            im2=np.copy(patterns[i,j])
            im_2_fft=np.fft.fftshift(np.fft.fft2(im2))
            phase_cross_corr=np.exp(-1j*np.angle(im_2_fft*np.conjugate(aperture)))
            phase_cross_corr=np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(phase_cross_corr))))
            indy, indx=np.unravel_index(phase_cross_corr.argmax(), phase_cross_corr.shape)
            im2=np.roll(im2, (indy-shy//2, indx-shx//2), axis=(0,1))
            patterns[i,j]=im2
    return patterns

def create_binned_dataset(path_orig, path_new, bin):
    f_old=h5py.File(path_orig, "r")
    data_old=f_old["data"]
    data_new=np.zeros((data_old.shape[0], data_old.shape[1]//bin, data_old.shape[2]//bin))
    for i in range(bin):
        for j in range(bin):
            data_new+=data_old[:,i:bin*(data_old.shape[1]//bin):bin, j:bin*(data_old.shape[1]//bin):bin]
    f_old.close()
    f=h5py.File(path_new, "a")
    try:
        data=f["data"]
        data=data_new
    except:
        f.create_dataset("data", data=data_new)
    f.close()
    


def iterative_dpc(COMx, COMy, phase=None, select=None,px_size=1,print_flag=False, hpass=0, lpass=0, step_size=0.1, num_iterations=100, beta=0.5, bin_fac=1, use_backtracking=True, pad_width=1):
    if select is None:
        Ny, Nx = COMx.shape
    else:
        Ny, Nx = select.shape
    if phase is None:
        padded_phase = np.random.rand(Ny+pad_width, Nx+pad_width)*1e-3
    else:
        padded_phase = np.pad(phase, [[0,pad_width],[0,pad_width]])
    kx, ky=np.meshgrid(np.fft.fftshift(np.fft.fftfreq(padded_phase.shape[1], px_size)), np.fft.fftshift(np.fft.fftfreq(padded_phase.shape[0], px_size)))
    k2=kx**2+ky**2
    k4=lpass*k2**2
    where=k2==0
    k2[where]=1
    mask=np.ones((Ny, Nx))
    if not(select is None):
        mask[(1-select).astype(bool)]=0
    mask=np.pad(mask, [[0,pad_width],[0,pad_width]])
    for iteration in range(num_iterations):
        error_y, error_x=np.gradient(padded_phase, px_size, edge_order=2)
        if select is None:
            error_x[:Ny, :Nx]-=COMx
            error_y[:Ny, :Nx]-=COMy
        else:
            error_x[:Ny, :Nx][select]-=COMx
            error_y[:Ny, :Nx][select]-=COMy
        error_y*=mask
        error_x*=mask
        error= np.sum(error_x**2) +np.sum(error_y**2)
        phase_update = (kx*np.fft.fftshift(np.fft.fft2(error_x)) + ky*np.fft.fftshift(np.fft.fft2(error_y)))/(2j*np.pi*(k4 + k2 + hpass))
        phase_update[where]=0
        derivative=np.real(np.fft.ifft2(np.fft.ifftshift(phase_update)))
        phase_update=-step_size*derivative
        if use_backtracking:
            count=0
            new_error=error+1
            while new_error > error and (count<99):
                new_phase = padded_phase + phase_update
                new_error_y, new_error_x = np.gradient(new_phase, px_size, edge_order=2)
                if select is None:
                    new_error_x[:Ny, :Nx]-=COMx
                    new_error_y[:Ny, :Nx]-=COMy
                else:
                    new_error_x[:Ny, :Nx][select]-=COMx
                    new_error_y[:Ny, :Nx][select]-=COMy
                new_error_y*=mask
                new_error_x*=mask
                new_error= np.sum(new_error_x**2) + np.sum(new_error_y**2)
                phase_update*=beta
                count+=1
            if new_error < error and (count<99):
                padded_phase = new_phase
        else:
            padded_phase+=phase_update
            count=2
        if count<1:
            step_size*=1/beta
        if count>=3:
            step_size*=beta
            if count>99:
                break
        if (print_flag>2) or iteration==(num_iterations-1):
            sys.stdout.write(f"\rIteration {iteration}, Total Error: {error:.3e}, count: {count}, step: {step_size:.2e}")
            sys.stdout.flush()
    return padded_phase[:Ny, :Nx]



def upsampled_tcbf(pypty_params, upsample=5, pad=10,compensate_lowfreq_drift=False, default_float=64,round_shifts=False,xp=np,save=0,max_parallel_fft=100, bin_fac=1):
    """
    Run a tcBF reconstruction on an upsampled grid. Note that usually before doing so you need to execute run_tcbf_alignment fucntion to adjust pypty_params.
    inputs:
        pypty_params - dictionary with experimetal paramers and other settings. For more please see run_ptychography() and append_exp_params()
        upsample - integer upsampling factor
        pad - amount of scan positions to add to the sides to eliminate wrap-around artifacts
        compensate_lowfreq_drift - boolean flag. If true, the code will try to compensate drifts of an aperture. Requieres to run_tcbf_alignemnt beforehand!!!
        default_float- 64 or 32 for better memory
        round_shifts - boolean. If true, shifts will be rounded.
        xp - backend. numpy or Cupy
        save - boolean flag. Default false.
        max_parallel_fft - amount of FFTs to do in a vectorized fashion.
        bin_fac - binning for the data in reciprocal space. Default 1 (no binning).
    outputs:
        O_r - real valued tcBF image
    """
    bright_field_pixels=None
    conv_angle_rad = pypty_params.get("conv_semiangle_mrad", 1)*1e-3
    acc_voltage_kV= pypty_params.get("acc_voltage", 60)
    scan_step= pypty_params.get("scan_step_A", 1)
    aperture= pypty_params.get("aperture_mask", 1)
    aberrations=pypty_params.get("aberrations", None)
    PL_rot=pypty_params.get("PLRotation_deg", 0)
    data_path= pypty_params.get("data_path", "")
    data_pad= pypty_params.get("data_pad", 1)
    rez_pixel_size_A= pypty_params.get("rez_pixel_size_A", 1)
    upsample_pattern= pypty_params.get("upsample_pattern", 1)
    xp=cp  #pypty_params.get("backend", cp)
    if data_pad!=0:
        aperture=aperture[data_pad:-data_pad,data_pad:-data_pad]
    if upsample_pattern!=1:
        aperture=downsample_something(aperture, upsample_pattern, np)
        
    scan_size= np.copy(pypty_params.get("scan_size", None))
    if data_path[-3:]==".h5":
        f=h5py.File(data_path, "r")
        patterns=f["data"]
    elif data_path[-4:]==".npy":
        patterns=np.load(data_path)
        if len(patterns.shape)==4:
            scan_size=[patterns.shape[0], patterns.shape[1]]
            patterns=patterns.reshape(patterns.shape[0]* patterns.shape[1], patterns.shape[2],patterns.shape[3])
            
    comx= pypty_params.get("aperture_shifts_x", pypty_params.get("comx", None)) # pypty_params.get("comx", None)
    comy=pypty_params.get("aperture_shifts_y", pypty_params.get("comy", None)) #pypty_params.get("comy", None)
    print_flag=pypty_params.get("print_flag", 1)
    if print_flag:
        sys.stdout.write("\n******************************************************************************\n************************ Creating upsampled tcBF Image ***********************\n******************************************************************************\n")
        sys.stdout.flush()
    try:
        os.makedirs(pypty_params["output_folder"], exist_ok=True)
        os.makedirs(pypty_params["output_folder"]+"/tcbf/", exist_ok=True)
    except:
        sys.stdout.write("output folder was not created!")
    wavelength=12.4/np.sqrt(acc_voltage_kV*(acc_voltage_kV+2*511))
    radperpixel=rez_pixel_size_A*wavelength*bin_fac
    if default_float==64:
        default_float=xp.float64
        default_complex=xp.complex128
    else:
        default_float=xp.float32
        default_complex=xp.complex64
    N_steps_y, N_steps_x= scan_size
    aperture=aperture>0.5*np.max(aperture)
    if bin_fac!=1:
        sys.stdout.write("\nBinning the data")
        sky, skx=patterns.shape[1], patterns.shape[2]
        zeroindex_x, zeroindex_y=skx//2, sky//2
        difference_x_left,  difference_y_left  = (zeroindex_x-int(np.floor(bin_fac/2)))%bin_fac, (zeroindex_y-int(np.floor(bin_fac/2)))%bin_fac
        difference_x_right, difference_y_right =  skx- (skx-zeroindex_x-int(np.ceil(bin_fac/2)))%bin_fac, sky- (sky-zeroindex_y-int(np.ceil(bin_fac/2)))%bin_fac
        new_skx, new_sky=(difference_x_right-difference_x_left)//bin_fac, (difference_y_right-difference_y_left)//bin_fac
        if not(compensate_lowfreq_drift):
            aperture=aperture[difference_y_left:difference_y_right, difference_x_left:difference_x_right]
            aperture=np.mean(aperture.reshape(new_sky,bin_fac, new_skx,bin_fac), axis=(-3, -1)).astype(bool)
            bright_field_pixels=[]
            for ind111 in tqdm(range(patterns.shape[0])):
                pattern_binned=np.sum((np.copy(patterns[ind111])[difference_y_left:difference_y_right, difference_x_left:difference_x_right]).reshape(new_sky,bin_fac, new_skx,bin_fac), axis=(-3, -1))
                bright_field_pixels.append(pattern_binned[aperture])
            bright_field_pixels=np.array(bright_field_pixels)
    if compensate_lowfreq_drift: ## if we want to compensate the aperture wobbling, then we roll the patterns!
        bright_field_pixels=[]
        aperture=pypty_params.get("lowfreq_compensated_aperture", aperture)
        aperture=aperture>0.5*np.max(aperture)
        if bin_fac!=1:
            aperture=np.mean(aperture[difference_y_left:difference_y_right, difference_x_left:difference_x_right].reshape(new_sky,bin_fac, new_skx,bin_fac), axis=(-3, -1)).astype(bool)
        sys.stdout.write("\nAligning data!")
        comx=comx.flatten()
        comy=comy.flatten()
        maskx, masky=np.meshgrid(np.fft.fftfreq(patterns.shape[2]), np.fft.fftfreq(patterns.shape[1]), indexing="xy")
        for ind111 in tqdm(range(patterns.shape[0])):
            shifty, shiftx= int(comy[ind111]), int(comx[ind111])
            #pattern_shifted=np.real(np.fft.ifft2(np.exp(6.283185307179586j*(maskx*shiftx+masky*shifty))*np.fft.fft2(np.sqrt(np.copy(patterns[ind111])))))**2
            pattern_shifted=np.roll(np.copy(patterns[ind111]), (-shifty, -shiftx), axis=(0,1))
            if bin_fac!=1:
                pattern_shifted=np.sum(pattern_shifted[difference_y_left:difference_y_right, difference_x_left:difference_x_right].reshape(new_sky,bin_fac, new_skx,bin_fac), axis=(-3, -1))
            bright_field_pixels.append(pattern_shifted[aperture.astype(bool)])
        bright_field_pixels=np.array(bright_field_pixels)
    if bright_field_pixels is None:
        bright_field_pixels=xp.array([d[aperture] for d in patterns])
    sys.stdout.write("\n%.2e rad per pixel"%radperpixel)
    px_size_final=scan_step/upsample
    if max_parallel_fft is None:
        max_parallel_fft=bright_field_pixels.shape[-1]
    mask_scan=xp.zeros((N_steps_y*upsample+2*pad*upsample, N_steps_x*upsample+2*pad*upsample))
    mask_scan[pad*upsample:-pad*upsample:upsample, pad*upsample:-pad*upsample:upsample]=1
    mask_scan=mask_scan.astype(bool)
    qx,qy=xp.meshgrid(xp.fft.fftfreq(mask_scan.shape[1], 1), xp.fft.fftfreq(mask_scan.shape[0], 1))
    qx,qy=1j*qx, 1j*qy
    apshy, apshx=aperture.shape
    dqx, dqy=np.meshgrid(np.arange( -(apshx//2), apshx-(apshx//2), 1)*radperpixel, np.arange(-(apshy//2), apshy-(apshy//2), 1)*radperpixel)
    dqx, dqy=dqx[aperture], dqy[aperture]
    dqx, dqy= np.cos(PL_rot*np.pi/180)*dqx+ np.sin(PL_rot*np.pi/180)*dqy,  np.cos(PL_rot*np.pi/180)*dqy- np.sin(PL_rot*np.pi/180)*dqx
    aberrations=np.asarray(aberrations)
    if print_flag:
        sys.stdout.write("\nyour final pixel size will be %.2f Å"%px_size_final)
        sys.stdout.write("\nfinal shape of image will be: (%d, %d) "%(mask_scan.shape[0], mask_scan.shape[1]))
        sys.stdout.flush()
   
    bright_field_pixels=xp.asarray(bright_field_pixels).astype(default_float)
    weights=xp.zeros_like(mask_scan).astype(default_complex)
    O_r=xp.zeros_like(mask_scan).astype(default_complex)
    mask_for_weights=xp.fft.fft2(mask_scan)
    drx, dry=get_ctf_derivatives(aberrations, dqx, dqy, wavelength, 0)
    drx, dry=xp.array(drx), xp.array(dry)
    drx*=wavelength/px_size_final
    dry*=wavelength/px_size_final
    
    if round_shifts:
        drx=(xp.round(drx/(2*xp.pi),0).astype(int))
        dry=(xp.round(dry/(2*xp.pi),0).astype(int))
        if xp!=np:
            drx=drx.get()
            dry=dry.get()
        for i in tqdm(range(int(np.ceil(bright_field_pixels.shape[-1])))):
            ddry=dry[i]
            ddrx=drx[i]
            aligned_batch=cp.zeros((O_r.shape[0],O_r.shape[1]), dtype=default_float)
            aligned_batch[mask_scan.astype(bool)]=bright_field_pixels[:, i]
            aligned_batch=xp.roll(aligned_batch, (ddry,ddrx), axis=(0,1))
            w1=xp.roll(mask_scan, (ddry, ddrx), axis=(0,1))
            O_r+=aligned_batch
            weights+=w1
    else:
        for i in tqdm(range(int(np.ceil(bright_field_pixels.shape[-1]/max_parallel_fft)))):
            this_kern=xp.exp(-qx[:,:,None]*drx[None, None, i*max_parallel_fft:max_parallel_fft*(i+1)]-qy[:,:,None]*dry[None, None, i*max_parallel_fft:max_parallel_fft*(i+1)])
            aligned_batch=xp.zeros((O_r.shape[0],O_r.shape[1], this_kern.shape[-1]), dtype=default_float)
            aligned_batch[mask_scan.astype(bool),:]=bright_field_pixels[:, i*max_parallel_fft:max_parallel_fft*(i+1)]
            aligned_batch=xp.fft.fft2(aligned_batch, axes=(0,1)).astype(default_complex)
            aligned_batch=aligned_batch*this_kern
            O_r+=xp.sum(aligned_batch, -1)
            weights+=xp.sum(xp.copy(mask_for_weights)[:,:,None]*this_kern,-1)
        O_r=xp.fft.ifft2(O_r, axes=(0,1))
        weights=xp.fft.ifft2(weights, axes=(0,1))
    O_r_before=xp.copy(O_r)
    O_r=xp.conjugate(weights)*(O_r/(1e-3+xp.abs(weights)**2))
    O_r=xp.real(O_r)
    try:
        f.close()
    except:
        pass
    try:
        O_r=O_r.get()
        xp.fft.config.clear_plan_cache()
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()
    except:
        O_r=np.array(O_r)
    if save: np.save(pypty_params["output_folder"]+"/tcbf/tcbf_image_upsampling_%d.npy"%(upsample), O_r);
    return  O_r, px_size_final



def iterative_poisson_solver(laplace, phase=None, select=None,px_size=1,print_flag=False, hpass=0, lpass=0, step_size=0.1, num_iterations=100, beta=0.5, bin_fac=1, use_backtracking=True, pad_width=1, xp=np):
    if print_flag:
        sys.stdout.write("\n******************************************************************************\n*************************** Solving Poisson Equation *************************\n******************************************************************************\n")
        sys.stdout.flush()
    if select is None:
        Ny, Nx = laplace.shape
    else:
        Ny, Nx = select.shape
    if phase is None:
        padded_phase = xp.random.rand(Ny+pad_width, Nx+pad_width)*1e-5
    else:
        padded_phase = xp.pad(xp.asarray(phase), [[0,pad_width],[0,pad_width]])
    kx, ky=xp.meshgrid(xp.fft.fftshift(xp.fft.fftfreq(padded_phase.shape[1], px_size)), xp.fft.fftshift(xp.fft.fftfreq(padded_phase.shape[0], px_size)))
    k2=kx**2+ky**2
    k4=lpass*k2**2
    where=k2==0
    k2[where]=1
    mask=xp.ones((Ny, Nx))
    if not(select is None):
        mask[(1-select).astype(bool)]=0
    mask=xp.pad(mask, [[0,pad_width],[0,pad_width]])
    laplace=xp.asarray(laplace)
    
    
    for iteration in range(num_iterations):
        dely, delx=xp.gradient(padded_phase, px_size, edge_order=1)
        delyy, delyx= xp.gradient(dely, px_size, edge_order=1)
        delxy, delxx= xp.gradient(delx, px_size, edge_order=1)
        error= delyy+ delxx
        if select is None:
            error[:Ny, :Nx]-=laplace
        else:
            error[:Ny, :Nx][select]-=laplace
        error*=mask
        phase_update = xp.real(xp.fft.ifft2(xp.fft.ifftshift(xp.fft.fftshift(xp.fft.fft2(error))/(-4* xp.pi**2 *(k4 + k2 + hpass)))))
        error=xp.sum(error**2)
        phase_update*=-step_size
        if use_backtracking:
            count=0
            new_error=error+1
            while new_error > error and (count<99):
                new_phase = padded_phase + phase_update
                dely, delx=xp.gradient(new_phase, px_size, edge_order=1)
                delyy, delyx= xp.gradient(dely, px_size, edge_order=1)
                delxy, delxx= xp.gradient(delx, px_size, edge_order=1)
                new_error= delyy+ delxx
                if select is None:
                    new_error[:Ny, :Nx]-=laplace
                else:
                    new_error[:Ny, :Nx][select]-=laplace
                new_error*=mask
                new_error= xp.sum(new_error**2)
                phase_update*=beta
                count+=1
            if new_error < error and (count<99):
                padded_phase = xp.copy(new_phase)
        else:
            padded_phase+=phase_update
            count=2
        if count<=1:
            step_size*=1/beta
        if count>=3:
            step_size*=beta
            if count>99:
                break
        if (print_flag>2) or iteration==(num_iterations-1):
            if print_flag==2:
                sys.stdout.write(f"\rIteration {iteration}, Total Error: {error:.3e}, count: {count}, step: {step_size:.2e}")
            else:
                sys.stdout.write(f"\nIteration {iteration}, Total Error: {error:.3e}, count: {count}, step: {step_size:.2e}")
            sys.stdout.flush()
    try:
        padded_phase=padded_phase.get()
    except:
        pass
    return padded_phase[:Ny, :Nx]

def create_aberrations_chunks(pypty_params,chop_size, n_abs):
    scan_size=pypty_params.get("scan_size", None)
    sh0,sh1=scan_size
    aberration_marker=np.zeros((sh0,sh1))
    n_chops_0=int(np.ceil(sh0/chop_size))
    n_chops_1=int(np.ceil(sh1/chop_size))
    for i in range(n_chops_0):
        for j in range(n_chops_1):
            aberration_marker[i*chop_size:(i+1)*chop_size,j*chop_size:(j+1)*chop_size]=i*n_chops_0+j
    pypty_params['aberrations_array']  = np.zeros((n_chops_0*n_chops_1, n_abs), dtype=np.float32)
    pypty_params['aberration_marker'] = (aberration_marker.flatten()).astype(int)
    return pypty_params


def create_probe_marker_chunks(pypty_params,chop_size):
    scan_size=pypty_params.get("scan_size", None)
    sh0,sh1=scan_size
    probe_marker=np.zeros((sh0,sh1))
    n_chops_0=int(np.ceil(sh0/chop_size))
    n_chops_1=int(np.ceil(sh1/chop_size))
    for i in range(n_chops_0):
        for j in range(n_chops_1):
            probe_marker[i*chop_size:(i+1)*chop_size,j*chop_size:(j+1)*chop_size]=i*n_chops_0+j
    pypty_params['probe_marker'] = (probe_marker.flatten()).astype(int)
    return pypty_params


def create_sequence_box(pypty_params, left, top, width, height):
    seq=[]
    scan_size=pypty_params["scan_size"]
    for ii1 in range(top, top+height, 1):
        for jj1 in range(left, left+width, 1):
            seq.append(scan_size[1]*ii1+jj1)
    pypty_params["sequence"]=seq
    return pypty_params
    
    
def create_sub_sequence(pypty_params, left, top, width, height, sub):
    seq=[]
    scan_size=pypty_params["scan_size"]
    ii1_list=np.arange(top, top+height, sub)
    jj1_list=np.arange(left, left+width, sub)
    for ii1 in ii1_list:
        for jj1 in jj1_list:
            seq.append(int(scan_size[1]*ii1+jj1))
    seq=list(np.unique(seq))
    pypty_params["sequence"]=seq
    return pypty_params
    
    
def create_sequence_from_points(pypty_params, im, yf,xf, width_roi=20):
    scan_size=pypty_params["scan_size"]
    seq=[]
    for i in range(len(yf)):
        ty,tx=yf[i], xf[i]
        for w in range(-width_roi//2,width_roi-width_roi//2,1):
            for h in range(-width_roi//2,width_roi-width_roi//2,1):
                seq.append((ty+w)*scan_size[1]+h+tx)
    seq=np.array(seq)
    seq=seq[(seq>=0)*(seq<(scan_size[0]*scan_size[1]))]
    seq=list(np.unique(seq))
    return seq


def rotate_scan_grid(pypty_params, angle_deg):
    old_pl_rot   = pypty_params["PLRotation_deg"]
    new_pl_rot   = old_pl_rot + angle_deg
    old_postions = pypty_params["positions"]
    opy, opx=old_postions[:,0], old_postions[:,1]
    rot_ang=np.pi
    opx_prime, opy_prime=opx * np.cos(rot_ang) + opy * np.sin(rot_ang), -1*opx * np.sin(rot_ang) + opy * np.cos(rot_ang)
    opx_prime-=np.min(opx_prime)
    opy_prime-=np.min(opy_prime)
    old_postions[:,1]=opx_prime
    old_postions[:,0]=opy_prime
    pypty_params["PLRotation_deg"]=new_pl_rot
    pypty_params["positions"]=old_postions
    return pypty_params

def tiltbeamtodata(pypty_params, align_type="com"):
    probe=pypty_params["probe"]
    data_path=pypty_params["data_path"]
    data_pad=pypty_params["data_pad"]
    if data_path[-3:]==".h5":
        h5file=h5py.File(data_path, "r")
        h5data=h5file["data"]
    elif data_path[-4:]==".npy":
        h5data=np.load(data_path)
        if len(h5data.shape)==4:
            scan_size=[h5data.shape[0], h5data.shape[1]]
            h5data=h5file.reshape(h5data.shape[0]* h5data.shape[1], h5data.shape[2],h5data.shape[3])
    pacbed=np.sum(h5data, 0)
    beam_fft=np.sum(np.abs(np.fft.fftshift(np.fft.fft2(probe, axes=(0,1)), axes=(0,1)))**2, -1)[data_pad:-data_pad,data_pad:-data_pad]
    if align_type=="com":
        x=np.arange(pacbed.shape[0])
        x=x-np.mean(x)
        x,y=np.meshgrid(x,x)
        comxpac=np.average(x, weights=pacbed)
        comypac=np.average(y, weights=pacbed)
        comxbeam=np.average(x, weights=beam_fft)
        comybeam=np.average(y, weights=beam_fft)
        shift_x=comxpac-comxbeam
        shift_y=comypac-comybeam
    else:
        beam_fft=np.fft.fft2(beam_fft)
        pacbed=np.fft.fft2(pacbed)
        
        cross=np.conjugate(beam_fft)*pacbed
        cross=np.angle(cross)
        cross=np.exp(1j*cross)
        cross=np.real(np.fft.fftshift(np.fft.ifft2(cross)))
        indy, indx=np.unravel_index(cross.argmax(), cross.shape)
        peak_center=cross[indy, indx]
        test_peak_left=cross[indy, indx-1]
        test_peak_right=cross[indy, indx+1]
        test_peak_top=cross[indy-1, indx]
        test_peak_bottom=cross[indy+1, indx]
        if test_peak_right>test_peak_left:
            shift_x=test_peak_right /(test_peak_right+ peak_center)
            if np.abs(shift_x)>1: shift_x=test_peak_right/(test_peak_right-peak_center);
        else:
            if test_peak_right<test_peak_left:
                shift_x=test_peak_left/(test_peak_left + peak_center)
                if np.abs(shift_x)>1: shift_x=test_peak_left/(test_peak_left - peak_center);
            else:
                shift_x=0
        if test_peak_bottom>test_peak_top:
            shift_y=test_peak_bottom/(test_peak_bottom + peak_center)
            if np.abs(shift_y)>1: shift_y=test_peak_bottom/(test_peak_bottom - peak_center);
        else:
            if test_peak_bottom<test_peak_top:
                shift_y=test_peak_top/(peak_center+ test_peak_top)
                if np.abs(shift_y)>1: shift_y=test_peak_top/(-peak_center+ test_peak_top);
            else:
                shift_y=0
                
        shift_x=indx-shift_x-cross.shape[1]//2
        shift_y=indy-shift_y-cross.shape[1]//2
    print("\nShifting beam by %.2e px along x and by %.2e px along y."%(shift_x, shift_y))

    kx,ky=np.meshgrid(np.fft.fftshift(np.fft.fftfreq(probe.shape[1])), np.fft.fftshift(np.fft.fftfreq(probe.shape[0])))
    kernel=np.exp(2j*np.pi*(kx*shift_x+ky*shift_y))[:,:,None]
    probe*=kernel
    pypty_params["probe"]=probe
    return pypty_params



def fit_aberrations_to_wave(wave, px_size_A, acc_voltage, thresh=0,
                            aberrations_guess=[0,0,0,0,0,0,0,0,0,0,0,0],
                            plot=True, ftol=1e-20, xtol=1e-20, loss="linear", max_mrad=np.inf):
    try:
        from skimage.restoration import unwrap_phase
    except:
        pass
    wave=wave.copy()
    
    x=np.arange(wave.shape[0])-wave.shape[0]//2
    x,y=np.meshgrid(x,x)
    kx=np.fft.fftshift(np.fft.fftfreq(wave.shape[0]))
    kx,ky=np.meshgrid(kx,kx, indexing="xy")
    comx, comy=np.average(x, weights=np.abs(wave)**2), np.average(y, weights=np.abs(wave)**2)

    fourier_wave=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wave)))
    mag=np.abs(fourier_wave)
    mag=mag>=thresh*np.max(mag)
    phase=-1*np.angle(fourier_wave)-2*np.pi*(kx*comx+ky*comy)
    phase*=mag
    phase= unwrap_phase(phase, wrap_around=(False, False))
    wavelength=12.4/np.sqrt(acc_voltage*(acc_voltage+2*511))
    kx,ky=kx*wavelength/px_size_A, ky*wavelength/px_size_A
    kr=(kx**2+ky**2)**0.5
    mag*=kr<=(max_mrad*1e-3)
    phase-=phase[kr==0]
    phase*=mag
    
    ctf_matrix=get_ctf_matrix(kx, ky, len(aberrations_guess), wavelength, np)[:, mag]
    ctf_matrix=np.swapaxes(ctf_matrix, 0,1)
    def jac_ctf_fit(x):
        nonlocal ctf_matrix
        return ctf_matrix
    phase_crop=phase[mag]
    def objective(aberrations):
        nonlocal phase_crop, ctf_matrix
        ctf=cp.sum(aberrations[None,:]*ctf_matrix, axis=1)
        return ctf-phase_crop
    result=least_squares(objective,aberrations_guess, jac=jac_ctf_fit, ftol=ftol, loss=loss, xtol=xtol)
    aberrations=result["x"]
    num_abs=len(aberrations)
    possible_n, possible_m, possible_ab=convert_num_to_nmab(num_abs)
    aber_print, s=nmab_to_strings(possible_n, possible_m, possible_ab), ""
    for i in range(len(aberrations)): s+=aber_print[i]+": %.2e Å; "%aberrations[i];
    sys.stdout.write("\nFitted aberrations: %s"%s[:-1])
    fitted_ctf=get_ctf(aberrations, kx, ky, wavelength, angle_offset=0)*mag
    if plot:
        fig, ax=plt.subplots(1,3, figsize=(9,3))
        im0=ax[0].imshow(fitted_ctf)
        fig.colorbar(im0, ax=ax[0], orientation="horizontal", fraction=0.0475, pad=0)
        ax[0].set_title("Fitted phase")
        ax[0].axis("off")
        im1=ax[1].imshow(phase)
        fig.colorbar(im1, ax=ax[1], orientation="horizontal", fraction=0.0475, pad=0)
        ax[1].set_title("Target phase")
        ax[1].axis("off")

        im2=ax[2].imshow(phase-fitted_ctf)
        fig.colorbar(im2, ax=ax[2], orientation="horizontal", fraction=0.0475, pad=0)
        ax[2].set_title("Difference")
        ax[2].axis("off")
        plt.show()
    return aberrations
    
    
def mesh_model_positions(step_size, angle_rad, x, y):
    x_model = step_size * np.cos(angle_rad) * x - step_size * np.sin(angle_rad) * y
    y_model = step_size * np.sin(angle_rad) * x + step_size * np.cos(angle_rad) * y
    return x_model, y_model
    
def mesh_objective_positions(ini_guess, x, y, mesh_x, mesh_y):
    step, angle=ini_guess
    x_model, y_model = mesh_model_positions(step, angle, x, y)
    return np.sum((x_model - mesh_x)**2 + (y_model - mesh_y)**2)

def get_step_angle_scan_grid(positions, scan_size):
    pos=positions.copy()
    posy, posx=pos[:,0], pos[:,1]
    x, y=np.meshgrid(np.arange(scan_size[1]),np.arange(scan_size[0]))
    x= x.flatten()-np.mean(x)
    y= y.flatten()-np.mean(y)
    posy-=np.mean(posy)
    posx-=np.mean(posx)
    result = minimize(mesh_objective_positions,  x0=[10,0],
                      args=(x, y, posx, posy), method="Powell",
                      bounds=[(0.001,10000),(-np.pi, np.pi)],
                      tol=1e-10, options={"maxiter":1000})
    angle = (result["x"][1])*180/np.pi
    step= (result["x"][0])
    final_mesh_x, final_mesh_y=mesh_model_positions(step, angle*np.pi/180, x, y)
    difference=np.stack((posy-final_mesh_y,posx-final_mesh_x))
    print("std: ", np.std(difference), " px.")
    return step, angle

def add_scalebar_ax(ax, x,y, width, height, x_t, y_t, px_size, unit):
    try:
        from matplotlib.patches import Rectangle
        import matplotlib.patheffects as PathEffects
    except:
        pass
    rect=Rectangle([x,y], width/px_size, height, color="white", alpha=0.9)
    text2=ax.text(x_t, y_t, str(width)+" "+unit, c="w", fontsize=20)
    text2.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    ax.add_patch(rect)
    rect.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])



