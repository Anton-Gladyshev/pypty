try:
    import cupyx
    import cupy as cp
except:
    import numpy as cp
from pypty import fft as pyptyfft
from pypty import utils as pyptyutils

def multislice(full_probe, this_obj_chopped, num_slices, n_obj_modes, n_probe_modes, this_distances, this_wavelength, q2, qx, qy, exclude_mask, is_single_dist, this_tan_x,this_tan_y, damping_cutoff_multislice, smooth_rolloff, master_propagator_phase_space,  half_master_propagator_phase_space, mask_clean, waves_multislice, wave, default_float, default_complex):
    """
    Simulate multislice wave propagation using a classic split-step integrator (2nd order precision with respect to slice thickness if beam is optimized).

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
    """
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
            wave=pyptyfft.fft2(wave, axes=(1,2), overwrite_x=True)
            waves_multislice[:, :,:,ind_multislice, :, :, 1]=wave ### FFT of \psi^{out}_{ind\ multislice}, x2 cutoff, but it is needed only for tilts grads,  we will set the x1 cutoff later!
            wave*=propagator_phase_space
            wave=pyptyfft.ifft2(wave, axes=(1,2), overwrite_x=True) # x1 cutoff
    cp.conjugate(waves_multislice, out=waves_multislice)
    return waves_multislice, wave


def multislice_grads(dLoss_dP_out, waves_multislice, this_obj_chopped, object_grad, tilts_grad, is_single_dist,this_distances, exclude_mask, this_wavelength, q2, qx, this_tan_x, qy, this_tan_y, num_slices, n_obj_modes,tiltind, master_propagator_phase_space, this_step_tilts, damping_cutoff_multislice, smooth_rolloff, tilt_mode,  compute_batch, mask_clean, this_step_probe, this_step_obj, this_step_pos_correction, masked_pixels_y, masked_pixels_x, default_float, default_complex, helper_flag_4):
    """
    Compute gradients for classic multislice propagation model (object, probe, and tilts).


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
    """
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
            dLoss_dP_out=pyptyfft.fft2(dLoss_dP_out, axes=(1,2), overwrite_x=True) ## x2 cutoff
            dLoss_dP_out*=propagator_phase_space ### reuse it for tilts update!!!!
            if this_step_tilts>0 and (tilt_mode==0 or tilt_mode==3 or tilt_mode==4):
                sh=-12.566370614359172*prop_distance/(waves_multislice.shape[1]*waves_multislice.shape[2])
                dLoss_dPropagator=cp.sum(waves_multislice[:, :,:,i_update-1,:,:,1]*dLoss_dP_out, (3,4))
                tilts_grad[tiltind,3]+=sh*cp.sum(cp.imag(dLoss_dPropagator*qx), (1,2))
                tilts_grad[tiltind,2]+=sh*cp.sum(cp.imag(dLoss_dPropagator*qy), (1,2))
            dLoss_dP_out=pyptyfft.ifft2(dLoss_dP_out, axes=(1,2), overwrite_x=True) #  x1 cutoff
    if this_step_obj>0:
        if waves_multislice.shape[-3]==1:
            this_grad=waves_multislice[:,:,:,:,0,:,0] #just  one probe mode, x2 cutoff
        else:
            this_grad=cp.sum(waves_multislice[:,:,:,:,:,:,0], -2) #sum over all probe modes, x2 cutoff
        this_grad=pyptyfft.fft2(this_grad, (1,2), overwrite_x=True)
        this_grad*=mask_clean[:,:,:,None,None]
        this_grad=pyptyfft.ifft2(this_grad, (1,2), overwrite_x=True)
        scatteradd(object_grad, masked_pixels_y, masked_pixels_x, this_grad)
    if helper_flag_4:
        if waves_multislice.shape[-2]==1:
            interm_probe_grad=dLoss_dP_out[:,:,:,:,0]
        else:
            interm_probe_grad=cp.sum(dLoss_dP_out,-1) ## sum over obj modes
    else:
        interm_probe_grad=None
    return object_grad, interm_probe_grad, tilts_grad



def better_multislice(full_probe, this_obj_chopped, num_slices, n_obj_modes, n_probe_modes, this_distances, this_wavelength, q2, qx, qy, exclude_mask, is_single_dist, this_tan_x,this_tan_y, damping_cutoff_multislice, smooth_rolloff, master_propagator_phase_space,  half_master_propagator_phase_space, mask_clean, waves_multislice, wave, default_float, default_complex):
    """
    Simulate multislice wave propagation using an additive split-step method (5th order precision with respect to slice thickness).

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
    """
    if n_obj_modes==1:
        wave=full_probe[:,:,:,:,None]
    else:
        wave=cp.repeat(full_probe[:,:,:,:,None], n_obj_modes, axis=-1)
    for ind_multislice in range(0,num_slices,1):
        transmission_func=this_obj_chopped[:, :,:, ind_multislice:ind_multislice+1, :]
        half_transmission_func=transmission_func**0.5
        half_transmission_func=pyptyutils.fourier_clean_3d(half_transmission_func, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        if master_propagator_phase_space is None:
            half_propagator_phase_space=cp.exp(-1.570796326794j*this_distances[ind_multislice]*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*mask_clean
            half_propagator_phase_space=cp.expand_dims(half_propagator_phase_space,(-1,-2))
            propagator_phase_space=half_propagator_phase_space**2 # actually a full propagator over one slice
        else:
            half_propagator_phase_space=half_master_propagator_phase_space
            propagator_phase_space=master_propagator_phase_space
        fourier_wave=pyptyfft.fft2(wave,        axes=(1,2)) # Fourier
        
        help_psi_11=wave*half_transmission_func
        help_psi_11=pyptyfft.fft2(help_psi_11, axes=(1,2)) ## Fourier
        help_psi_12=pyptyfft.ifft2(help_psi_11*half_propagator_phase_space, axes=(1,2)) ## Real
        help_psi_13=help_psi_12*half_transmission_func
        help_psi_13=pyptyfft.fft2(help_psi_13, axes=(1,2)) ## Fourier
        help_psi_14=pyptyfft.ifft2(help_psi_13*half_propagator_phase_space, axes=(1,2)) ## Real
        
        help_psi_21=pyptyfft.ifft2(fourier_wave*half_propagator_phase_space, axes=(1,2)) ## Real
        help_psi_22=help_psi_21*half_transmission_func
        help_psi_22=pyptyfft.fft2(help_psi_22, axes=(1,2)) # Forier
        help_psi_23=pyptyfft.ifft2(help_psi_22*half_propagator_phase_space, axes=(1,2)) ## Real
        help_psi_24=help_psi_23*half_transmission_func # Real
        help_psi_24=pyptyutils.fourier_clean_3d(help_psi_24, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        
        help_psi_31=pyptyfft.ifft2(fourier_wave * propagator_phase_space, axes=(1,2)) # Real
        help_psi_32=help_psi_31 * transmission_func # Real
        help_psi_32=pyptyutils.fourier_clean_3d(help_psi_32, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp) # Real
        help_psi_41=wave * transmission_func
        help_psi_41=pyptyfft.fft2(help_psi_41, axes=(1,2)) ## Fourier
        help_psi_42=pyptyfft.ifft2(help_psi_41*propagator_phase_space, axes=(1,2)) ## Real
        
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
    cp.conjugate(waves_multislice, out=waves_multislice)
    return waves_multislice, wave ## all waves and just the exit wave
    
def better_multislice_grads(dLoss_dP_out, waves_multislice, this_obj_chopped, object_grad, tilts_grad, is_single_dist, this_distances, exclude_mask, this_wavelength, q2, qx, this_tan_x, qy, this_tan_y, num_slices, n_probe_modes, n_obj_modes,tiltind, this_step_tilts,  master_propagator_phase_space, half_master_propagator_phase_space, damping_cutoff_multislice, smooth_rolloff, tilt_mode,  compute_batch, mask_clean, masked_pixels_y, masked_pixels_x, default_float, default_complex):
    """
    Compute gradients of object, probe, and tilts for the "better_multislice" wave propagation model.

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
    """
    sh0=this_obj_chopped.shape[0]
    if is_single_dist:
        prop_distance=this_distances[0]
        half_propagator_phase_space=cp.conjugate(cp.copy(half_master_propagator_phase_space))
        propagator_phase_space=cp.conjugate(cp.copy(master_propagator_phase_space))
    backshifted_exclude=cp.expand_dims(mask_clean, (-1,-2))
    for i_update in range(num_slices-1,-1,-1):
        if not(is_single_dist):
            prop_distance=this_distances[i_update]
            half_propagator_phase_space=cp.exp(1.570796326794j*this_distances[ind_multislice]*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*pyptyfft.ifftshift(exclude_mask)
            propagator_phase_space=propagator_phase_space**2 # actually a full propagator over one slice
        ### get the derivatives of the four sub-exitwaves
        dLoss_dpsi_14=dLoss_dP_out*(2/3)
        dLoss_dpsi_24=dLoss_dP_out*(2/3)
        dLoss_dpsi_32=dLoss_dP_out*(-1/6)
        dLoss_dpsi_42=dLoss_dP_out*(-1/6)
        ## get the  object and half of the object (conjugated)
        transmission_func=cp.conjugate(this_obj_chopped[:,:,:, i_update:i_update+1, :])
        half_transmission_func=transmission_func**0.5
        half_transmission_func=pyptyutils.fourier_clean_3d(half_transmission_func, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
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
        dLoss_dpsi_14_fourier=pyptyfft.fft2(dLoss_dpsi_14, axes=(1,2))
        dLoss_dpsi_13=pyptyfft.ifft2(dLoss_dpsi_14_fourier*half_propagator_phase_space, axes=(1,2)) ### <<< fourir cleaned 2/3 cutoff
        dLoss_dpsi_12=dLoss_dpsi_13*half_transmission_func ###<<< aliasling!
        dLoss_dpsi_12_fourier=pyptyfft.fft2(dLoss_dpsi_12, axes=(1,2))*backshifted_exclude
        dLoss_dpsi_11=pyptyfft.ifft2(dLoss_dpsi_12_fourier*half_propagator_phase_space, axes=(1,2)) ###<< 2/3 cutoff
        dLoss_dpsi_23=dLoss_dpsi_24*half_transmission_func
        dLoss_dpsi_23_fourier=pyptyfft.fft2(dLoss_dpsi_23, axes=(1,2))*backshifted_exclude
        dLoss_dpsi_22=pyptyfft.ifft2(dLoss_dpsi_23_fourier*half_propagator_phase_space, axes=(1,2)) ##2/3 cutoff
        dLoss_dpsi_21=dLoss_dpsi_22*half_transmission_func
        dLoss_dpsi_21_fourier=pyptyfft.fft2(dLoss_dpsi_21, axes=(1,2))*backshifted_exclude
        dLoss_dpsi_31=dLoss_dpsi_32*transmission_func
        dLoss_dpsi_31_fourier=pyptyfft.fft2(dLoss_dpsi_31, axes=(1,2))*backshifted_exclude
        dLoss_dpsi_42_fourier=pyptyfft.fft2(dLoss_dpsi_42, axes=(1,2))*backshifted_exclude
        dLoss_dpsi_41=pyptyfft.ifft2(dLoss_dpsi_42_fourier*propagator_phase_space, axes=(1,2))
        dLoss_dP_out=pyptyfft.ifft2((pyptyfft.fft2(dLoss_dpsi_41 * transmission_func + dLoss_dpsi_11 * half_transmission_func, axes=(1,2))+ dLoss_dpsi_21_fourier*half_propagator_phase_space + dLoss_dpsi_31_fourier * propagator_phase_space)*backshifted_exclude, axes=(1,2)) ### << actually a gradient with respect to an input wave of this slice!
        #dLoss_dP_out=ifft2((fft2(dLoss_dpsi_41 * transmission_func, axes=(0,1))+ dLoss_dpsi_31_fourier * propagator_phase_space)*backshifted_exclude, axes=(0,1))
        dLoss_dObject=dLoss_dpsi_13*help_psi_12 + dLoss_dpsi_11*input_wave0 + dLoss_dpsi_24*help_psi_23 + dLoss_dpsi_22*help_psi_21 # << object half
        dLoss_dObject=(dLoss_dObject*0.5)/half_transmission_func
        dLoss_dObject=cp.sum(dLoss_dObject+dLoss_dpsi_32*help_psi_31+dLoss_dpsi_41*input_wave0, -2) #  i<<full object, summing over the probe modes!!!
        dLoss_dObject=pyptyutils.fourier_clean_3d(dLoss_dObject, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
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
    """
    Simulate multislice wave propagation using an yoshida integrator (5th order precision with respect to slice thickness).

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
    """
    #sh0=this_obj_chopped.shape[0]
    sigma_yoshida=(2+2**(-1/3)+2**(1/3))/3
    if n_obj_modes==1:
        wave=full_probe[:,:,:,:,None]
    else:
        wave=cp.repeat(full_probe[:,:,:,:,None], n_obj_modes, axis=-1)
    for ind_multislice in range(0,num_slices,1):
        transmission_func_1=this_obj_chopped[:, :,:, ind_multislice:ind_multislice+1, :]
        transmission_func_2=transmission_func_1**(0.5-sigma_yoshida/2)
        transmission_func_1=transmission_func_1**(sigma_yoshida/2)
        transmission_func_1=pyptyutils.fourier_clean_3d(transmission_func_1, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        transmission_func_2=pyptyutils.fourier_clean_3d(transmission_func_2, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        if master_propagator_phase_space is None:
            propagator_phase_space_1=cp.expand_dims(cp.exp(-3.141592653j*sigma_yoshida*this_distances[ind_multislice]*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*mask_clean,(-1,-2))
            propagator_phase_space_2=cp.expand_dims(cp.exp(-3.141592653j*(1-2*sigma_yoshida)*this_distances[ind_multislice]*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*mask_clean,(-1,-2))
        else:
            propagator_phase_space_1=half_master_propagator_phase_space
            propagator_phase_space_2=master_propagator_phase_space
        waves_multislice[:, :,:,ind_multislice, :,:,0]=wave # psi_0 (input), cutoff
        wave=wave*transmission_func_1 #psi1 x2 cutoff
        wave=pyptyfft.fft2(wave,        axes=(1,2)) ##psi1_fft x2 cutoff
        waves_multislice[:, :,:,ind_multislice, :,:,1]=wave #x2 cutoff
        wave=pyptyfft.ifft2(wave * propagator_phase_space_1, axes=(1,2)) #psi2, cutoff
        waves_multislice[:, :,:,ind_multislice, :,:,2]=wave #cutoff
        wave=wave*transmission_func_2 #psi3 # x2 cutoff
        wave=pyptyfft.fft2(wave,        axes=(1,2)) ##psi3_fft # x2 cutoff
        waves_multislice[:, :,:,ind_multislice, :,:,3]=wave # psi3_fft x2 cutoff
        wave=pyptyfft.ifft2(wave * propagator_phase_space_2, axes=(1,2)) #psi4 # cutoff
        waves_multislice[:, :,:,ind_multislice, :,:,4]=wave # cutoff
        wave=wave*transmission_func_2 #psi5 # x2 cutoff
        wave=pyptyfft.fft2(wave,        axes=(1,2)) ##psi5_fft # x2 cutoff
        waves_multislice[:, :,:,ind_multislice, :,:,5]=wave #  psi5_fft x2 cutoff
        wave=pyptyfft.ifft2(wave * propagator_phase_space_1, axes=(1,2)) #psi6 # cutoff
        waves_multislice[:, :,:,ind_multislice, :,:,6]=wave # psi6, cutoff
        wave=wave*transmission_func_1  # exit wave, aka next input wave # x2 cutoff
        wave=pyptyutils.fourier_clean_3d(wave, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)# cutoff
    cp.conjugate(waves_multislice, out=waves_multislice)
    return waves_multislice, wave
    
def yoshida_multislice_grads(dLoss_dP_out, waves_multislice, this_obj_chopped, object_grad, tilts_grad, is_single_dist, this_distances, exclude_mask, this_wavelength, q2, qx, this_tan_x, qy, this_tan_y, num_slices, n_probe_modes, n_obj_modes,tiltind, this_step_tilts,  master_propagator_phase_space, half_master_propagator_phase_space, damping_cutoff_multislice, smooth_rolloff, tilt_mode,  compute_batch, mask_clean, masked_pixels_y, masked_pixels_x, default_float, default_complex):
    """
    Compute gradients for object, probe, and tilt parameters using Yoshida multislice propagation.


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
    """
    sigma_yoshida=(2+2**(-1/3)+2**(1/3))/3
    sh0=this_obj_chopped.shape[0]
    if is_single_dist:
        prop_distance=this_distances[0]
        propagator_phase_space_1=cp.conjugate(half_master_propagator_phase_space) #(conjugated)
        propagator_phase_space_2=cp.conjugate(master_propagator_phase_space)# (conjugated)
    for i_update in range(num_slices-1,-1,-1):
        if not(is_single_dist):
            prop_distance=this_distances[i_update]
            propagator_phase_space_1=cp.expand_dims(cp.exp(3.141592653j*sigma_yoshida*this_distances[i_update]*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*mask_clean,(-1,-2))
            propagator_phase_space_2=cp.expand_dims(cp.exp(3.141592653j*(1-2*sigma_yoshida)*this_distances[i_update]*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*mask_clean,(-1,-2))
        ## get the  object and half of the object (conjugated)
        transmission_func_0=cp.conjugate(this_obj_chopped[:, :,:, i_update:i_update+1, :])
        transmission_func_2=transmission_func_0**(0.5-sigma_yoshida/2)
        transmission_func_1=transmission_func_0**(sigma_yoshida/2)
        transmission_func_3=transmission_func_0**(0.5*sigma_yoshida-1)
        transmission_func_4=transmission_func_0**(0.5+0.5*sigma_yoshida)
        transmission_func_1=pyptyutils.fourier_clean_3d(transmission_func_1, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        transmission_func_2=pyptyutils.fourier_clean_3d(transmission_func_2, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        transmission_func_3=pyptyutils.fourier_clean_3d(transmission_func_3, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        transmission_func_4=pyptyutils.fourier_clean_3d(transmission_func_4, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
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
        dLoss_dpsi6_fourier=pyptyfft.fft2(dLoss_dpsi6, axes=(1,2))      # x2 cutoff
        dLoss_dpsi5=pyptyfft.ifft2(dLoss_dpsi6_fourier*propagator_phase_space_1, axes=(1,2)) #cutoff
        dLoss_dpsi4=dLoss_dpsi5*transmission_func_2              # x2 cutoff
        dLoss_dO2s=dLoss_dpsi5*psi4                             # x2 cutoff
        dLoss_dpsi4_fourier=pyptyfft.fft2(dLoss_dpsi4, axes=(1,2))       # x2 cutoff
        dLoss_dpsi3=pyptyfft.ifft2(dLoss_dpsi4_fourier*propagator_phase_space_2, axes=(1,2))   ## cutoff
        dLoss_dpsi2=dLoss_dpsi3*transmission_func_2                      # x2 cutoff
        dLoss_dO2s+=dLoss_dpsi3*psi2                                    # x2 cutoff
        dLoss_dpsi2_fourier=pyptyfft.fft2(dLoss_dpsi2, axes=(1,2))              # x2 cutoff
        dLoss_dpsi1=pyptyfft.ifft2(dLoss_dpsi2_fourier*propagator_phase_space_1, axes=(1,2))   # cutoff
        dLoss_dOs+=dLoss_dpsi1*psi0                                     # x2 cutoff
        dLoss_dP_out=dLoss_dpsi1*transmission_func_1                   # x2 cutoff
        dLoss_dP_out=pyptyutils.fourier_clean_3d(dLoss_dP_out, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)  ## cutoff
        dLoss_dOs=pyptyutils.fourier_clean_3d(dLoss_dOs, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)  ## cutoff
        dLoss_dO2s=pyptyutils.fourier_clean_3d(dLoss_dO2s, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp) ## cutoff
        dLoss_dOs=(0.5*sigma_yoshida)*dLoss_dOs*transmission_func_3+(0.5-0.5*sigma_yoshida)*dLoss_dO2s*transmission_func_4
        dLoss_dObject=cp.sum(dLoss_dOs,-2)
        dLoss_dObject=pyptyutils.fourier_clean_3d(dLoss_dObject, cutoff=damping_cutoff_multislice, rolloff=smooth_rolloff, default_float=default_float, xp=cp)
        scatteradd(object_grad[:,:,i_update, :], masked_pixels_y, masked_pixels_x, dLoss_dObject)
        if this_step_tilts>0 and (tilt_mode==0 or tilt_mode==3 or tilt_mode==4):
            sh=psi0.shape[2]*psi0.shape[1]
            dLoss_dPropagator=(dLoss_dpsi4_fourier*psi3*propagator_phase_space_2*(1-2*sigma_yoshida))+(dLoss_dpsi6_fourier*psi5+dLoss_dpsi2_fourier*psi1)*propagator_phase_space_1*sigma_yoshida
            dLoss_dPropagator=cp.sum(dLoss_dPropagator, (-1,-2)) ## summing over the modes
            tilts_grad[tiltind,3]+=-12.566370614359172*prop_distance*cp.imag(cp.sum(dLoss_dPropagator*qx))/sh ##prop_distance=-1*distance!
            tilts_grad[tiltind,2]+=-12.566370614359172*prop_distance*cp.imag(cp.sum(dLoss_dPropagator*qy))/sh
    interm_probe_grad=cp.sum(dLoss_dP_out,-1) ## sum over the object modes
    return object_grad,  interm_probe_grad, tilts_grad


def wide_beam_multislice(full_probe, this_obj_chopped, num_slices, n_obj_modes, n_probe_modes, this_distances, this_wavelength, q2, qx, qy, exclude_mask, is_single_dist, this_tan_x,this_tan_y, damping_cutoff_multislice, smooth_rolloff, master_propagator_phase_space,  half_master_propagator_phase_space, mask_clean, waves_multislice, wave, default_float, default_complex, wide_beam_coeffs):
    if n_obj_modes==1:
        wave=full_probe[:,:,:,:,None]
    else:
        wave=cp.repeat(full_probe[:,:,:,:,None], n_obj_modes, axis=-1)
    for ind_multislice in range(0,num_slices,1):
        waves_multislice[:,:,:, ind_multislice, :, :,0]=wave ## save ind_wb wave
        if is_single_dist:
            propagator_phase_space=master_propagator_phase_space
        else:
            propagator_phase_space=cp.expand_dims((-3.141592654*this_distances[ind_multislice]*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y))*mask_clean),(-1,-2))
            wide_beam_coeffs=cp.ones(9, dtype=default_complex)
            pizl=this_wavelength/(3.141592654*this_distances[ind_multislice])
            wide_beam_coeffs[1]=1j
            wide_beam_coeffs[2]=-0.25j*pizl-0.5
            wide_beam_coeffs[3]=0.125j*pizl**2-1j/6
            wide_beam_coeffs[4]=(-5j/64)*pizl**3 -(1/32)*pizl**2+1/24
            wide_beam_coeffs[5]=(7j/128)*pizl**4 + 1j/120
            wide_beam_coeffs[6]=(-21/512)*pizl**5-(1/1024)*pizl**4 + (1j/768)*pizl**3 - 1/720
            wide_beam_coeffs[7]=(33j/1024)*pizl**6-1j/5040
            wide_beam_coeffs[8]=(-429j/16384)*pizl**7-(25/8192)*pizl**6 + (1/6144)*pizl**4 + 1/(40320)
            
        for ind_wb in range(8):
            wave_0=wave*this_obj_chopped[:, :,:, ind_multislice:ind_multislice+1, :] # the slice dimension became a singular dimension of the probe modes, x2 cutoff
            wave_1=(pyptyfft.fft2(wave, axes=(1,2))*propagator_phase_space + pyptyfft.fft2(wave_0, axes=(1,2)))*cp.expand_dims(mask_clean, (-1,-2))
            wave=pyptyfft.ifft2(wave_1,axes=(1,2))
            waves_multislice[:,:,:, ind_multislice, :, :,ind_wb+1]=1*wave ## save last wave
        wave=cp.sum(wide_beam_coeffs[None,None,None, None, None,:]*waves_multislice[:,:,:, ind_multislice, :, :,:], axis=-1)
    cp.conjugate(waves_multislice, out=waves_multislice)
    return waves_multislice, wave

def wide_beam_multislice_grads(dLoss_dP_out, waves_multislice, this_obj_chopped, object_grad, tilts_grad, is_single_dist,this_distances, exclude_mask, this_wavelength, q2, qx, this_tan_x, qy, this_tan_y, num_slices, n_obj_modes,tiltind, master_propagator_phase_space, this_step_tilts, damping_cutoff_multislice, smooth_rolloff, tilt_mode,  compute_batch, mask_clean, this_step_probe, this_step_obj, this_step_pos_correction, masked_pixels_y, masked_pixels_x, default_float, default_complex, helper_flag_4, wide_beam_coeffs):
    this_obj_chopped=cp.conjugate(this_obj_chopped)
    sub_grads=cp.zeros_like(waves_multislice[:,:,:, 0, :, :,:])
    for i_update in range(num_slices-1,-1,-1): #backward propagation
        
        sub_grads[:,:,:, :, :, 0]=dLoss_dP_out
       
        if is_single_dist:
            prop_distance=this_distances[0]
            propagator_phase_space=cp.conjugate(master_propagator_phase_space)
        else:
            prop_distance=this_distances[i_update-1]
            propagator_phase_space=(3.141592654*prop_distance*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*mask_clean
            propagator_phase_space=cp.expand_dims(propagator_phase_space,(-1,-2)) ## expanding
            wide_beam_coeffs=cp.ones(9, dtype=default_complex)
            pizl=this_wavelength/(3.141592654*this_distances[i_update])
            wide_beam_coeffs[1]=1j
            wide_beam_coeffs[2]=-0.25j*pizl-0.5
            wide_beam_coeffs[3]=0.125j*pizl**2-1j/6
            wide_beam_coeffs[4]=(-5j/64)*pizl**3 -(1/32)*pizl**2+1/24
            wide_beam_coeffs[5]=(7j/128)*pizl**4 + 1j/120
            wide_beam_coeffs[6]=(-21/512)*pizl**5-(1/1024)*pizl**4 + (1j/768)*pizl**3 - 1/720
            wide_beam_coeffs[7]=(33j/1024)*pizl**6-1j/5040
            wide_beam_coeffs[8]=(-429j/16384)*pizl**7-(25/8192)*pizl**6 + (1/6144)*pizl**4 + 1/(40320)

        for ind_wb in range(len(wide_beam_coeffs)-1):
            g_0=sub_grads[:,:,:, :,:, ind_wb]*this_obj_chopped[:, :,:, i_update:i_update+1, :]
            g_0=(pyptyfft.fft2(sub_grads[:,:,:, :, :,ind_wb], axes=(1,2))*propagator_phase_space + pyptyfft.fft2(g_0, axes=(1,2)))*cp.expand_dims(mask_clean, (-1,-2))
            g_0=pyptyfft.ifft2(g_0,axes=(1,2))
            sub_grads[:,:,:, :, :, ind_wb+1]=1*g_0
        
        dLoss_dP_out=cp.sum(cp.conjugate(wide_beam_coeffs)[None,None, None,None, None,:]*sub_grads, axis=-1)
        dLoss_dS=cp.zeros_like(this_obj_chopped)
        for n in range(1,len(wide_beam_coeffs)):
            for nprime in range(0, n):
                dLoss_dS+=cp.conjugate(wide_beam_coeffs[n])*cp.conjugate(waves_multislice[:,:,:, i_update, :, :,nprime])*sub_grads[:,:,:, :,:, n-nprime-1]
        if this_step_tilts>0 and (tilt_mode==0 or tilt_mode==3 or tilt_mode==4):
            sh=12.566370614359172*prop_distance/(waves_multislice.shape[1]*waves_multislice.shape[2])
            dLoss_dPropagator=cp.fft.fft2(dLoss_dS, axes=(0,1))
            tilts_grad[tiltind,3]+=sh*cp.sum(cp.real(dLoss_dPropagator*qx), (1,2))
            tilts_grad[tiltind,2]+=sh*cp.sum(cp.real(dLoss_dPropagator*qy), (1,2))
            dLoss_dP_out=pyptyfft.ifft2(dLoss_dP_out, axes=(1,2), overwrite_x=True) #  x1 cutoff
        waves_multislice[:,:,:, i_update, :, :,0]=dLoss_dS
    if this_step_obj>0:
        if waves_multislice.shape[-3]==1:
            this_grad=waves_multislice[:,:,:,:,0,:,0] #just  one probe mode, x2 cutoff
        else:
            this_grad=cp.sum(waves_multislice[:,:,:,:,:,:,0], -2) #sum over all probe modes, x2 cutoff
        this_grad=pyptyfft.fft2(this_grad, (1,2), overwrite_x=True)
        this_grad*=mask_clean[:,:,:,None,None]
        this_grad=pyptyfft.ifft2(this_grad, (1,2), overwrite_x=True)
        scatteradd(object_grad, masked_pixels_y, masked_pixels_x, this_grad)
    if helper_flag_4:
        if waves_multislice.shape[-2]==1:
            interm_probe_grad=dLoss_dP_out[:,:,:,:,0]
        else:
            interm_probe_grad=cp.sum(dLoss_dP_out,-1) ## sum over obj modes
    else:
        interm_probe_grad=None
    return object_grad,  interm_probe_grad, tilts_grad
    



def scatteradd(full, masky, maskx, chop):
    """
    Adds batched object updates to their respective positions in the full object array.
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
    """
    try:
        cupyx.scatter_add(full.real, (masky, maskx), chop.real)
        cupyx.scatter_add(full.imag, (masky, maskx), chop.imag)
    except:
        cp.add.at(full, (masky, maskx), chop)

