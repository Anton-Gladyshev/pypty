try:
    import cupyx
    import cupy as cp
except:
    import numpy as cp
from pypty import fft as pyptyfft
from pypty import utils as pyptyutils

def multislice(full_probe, this_obj_chopped, num_slices, n_obj_modes, n_probe_modes, this_distances, edge_slice_thickness_multiplier, this_wavelength, q2, qx, qy, exclude_mask, is_single_dist, this_tan_x,this_tan_y, damping_cutoff_multislice, smooth_rolloff, master_propagator_phase_space,  half_master_propagator_phase_space, mask_clean, waves_multislice, wave, tilt_mode, this_step_tilts, default_float, default_complex):
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
                propagator_phase_space=1*master_propagator_phase_space
            else:
                propagator_phase_space=cp.expand_dims((cp.exp(-3.141592654j*this_distances[ind_multislice]*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*mask_clean),(-1,-2))
            if (edge_slice_thickness_multiplier>1) and (ind_multislice==0):
                propagator_phase_space=propagator_phase_space**edge_slice_thickness_multiplier
                
            wave=pyptyfft.fft2(wave, axes=(1,2), overwrite_x=True)
            if this_step_tilts>0 and (tilt_mode==0 or tilt_mode==3 or tilt_mode==4):
                waves_multislice[:, :,:,ind_multislice, :, :, 1]=wave ### FFT of \psi^{out}_{ind\ multislice}, x2 cutoff, but it is needed only for tilts grads,  we will set the x1 cutoff later!
            wave*=propagator_phase_space
            wave=pyptyfft.ifft2(wave, axes=(1,2), overwrite_x=True) # x1 cutoff
    cp.conjugate(waves_multislice, out=waves_multislice)
    return waves_multislice, wave


def multislice_grads(dLoss_dP_out, waves_multislice, this_obj_chopped, object_grad, tilts_grad, is_single_dist,this_distances, edge_slice_thickness_multiplier, exclude_mask, this_wavelength, q2, qx, this_tan_x, qy, this_tan_y, num_slices, n_obj_modes,tiltind, master_propagator_phase_space, this_step_tilts, damping_cutoff_multislice, smooth_rolloff, tilt_mode,  compute_batch, mask_clean, this_step_probe, this_step_obj, this_step_pos_correction, masked_pixels_y, masked_pixels_x, default_float, default_complex, helper_flag_4):
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
                prop_distance=1*this_distances[0]
                propagator_phase_space=cp.conjugate(master_propagator_phase_space)
            else:
                prop_distance=1*this_distances[i_update-1]
                propagator_phase_space=cp.exp(3.141592654j*prop_distance*(this_wavelength*q2+2*(qx*this_tan_x+qy*this_tan_y)))*mask_clean ### also the same as a conjugate of forward propagator P*(d)=P(-d)
                propagator_phase_space=cp.expand_dims(propagator_phase_space,(-1,-2)) ## expanding
            
            if (edge_slice_thickness_multiplier>1) and (i_update==1):
                propagator_phase_space=propagator_phase_space**edge_slice_thickness_multiplier
                prop_distance*=edge_slice_thickness_multiplier

            dLoss_dP_out=1*pyptyfft.fft2(dLoss_dP_out, axes=(1,2)) ## x2 cutoff
            dLoss_dP_out*=propagator_phase_space ### reuse it for tilts update!!!!
            if this_step_tilts>0 and (tilt_mode==0 or tilt_mode==3 or tilt_mode==4):
                sh=-12.566370614359172*prop_distance/(waves_multislice.shape[1]*waves_multislice.shape[2])
                dLoss_dPropagator=cp.sum(waves_multislice[:, :,:,i_update-1,:,:,1]*dLoss_dP_out, (3,4))
                tilts_grad[tiltind,3]+=sh*cp.sum(cp.imag(dLoss_dPropagator*qx), (1,2))
                tilts_grad[tiltind,2]+=sh*cp.sum(cp.imag(dLoss_dPropagator*qy), (1,2))
            dLoss_dP_out=pyptyfft.ifft2(dLoss_dP_out, axes=(1,2)) #  x1 cutoff
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
        cp.add.at(full, (masky, maskx), chop)
    except:
        cupyx.scatter_add(full.real, (masky, maskx), chop.real)
        cupyx.scatter_add(full.imag, (masky, maskx), chop.imag)
        

