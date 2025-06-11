import numpy as np
import sys
import os
import csv
import time
import pickle
import types
import copy
import inspect
from scipy.interpolate import griddata
try:
    import cupy as cp
    import cupyx.scipy.fft as sf
except:
    import numpy as cp
    import scipy.fft as sf
    
from pypty import fft as pyptyfft

import h5py
import datetime
from scipy import ndimage
from scipy.ndimage import label
from skimage.filters import gaussian
from skimage.measure import label
from scipy.ndimage import binary_dilation
from scipy.optimize import minimize
from scipy.ndimage import binary_dilation
from skimage.morphology import binary_closing,disk
from scipy.optimize import minimize
from collections import defaultdict
# from scipy.stats import entropy
from scipy.ndimage import map_coordinates

 
sys.setrecursionlimit(10000)

def fourier_clean_3d(array, cutoff=0.66, mask=None, rolloff=0, default_float=cp.float32, xp=cp):
    """
    Apply a 3D Fourier filter to the input array.
    
    Parameters
    ----------
    array : array_like
        Input 3D array to be filtered.
    cutoff : float, optional
        Cutoff frequency (default is 0.66).
    mask : array_like or None, optional
        Predefined mask to apply. If None, a mask is generated.
    rolloff : float, optional
        Rolloff parameter for smoothing the mask (default is 0).
    default_float : data-type, optional
        Data type for computations (default is cp.float32).
    xp : module, optional
        Array module (default is cp).
    
    Returns
    -------
    array_like
        The filtered array after applying the Fourier filter.
    """
    if not(cutoff is None) or not(mask is None):
        shape=array.shape
        if mask is None:
            x=pyptyfft.fftshift(pyptyfft.fftfreq(shape[2]))
            y=pyptyfft.fftshift(pyptyfft.fftfreq(shape[1]))
            x,y=cp.meshgrid(x,y, indexing="xy")
            r=x**2+y**2
            max_r=0.5*cutoff ## maximum freq for fftfreq is 0.5 when the px size is not specified
            mask=(r<=max_r**2).astype(default_float)
            if rolloff!=0:
                r0=(0.5*(cutoff-rolloff))**2
                mask[r>r0]*=0.5*(1+cp.cos(3.141592654*(r-r0)/(max_r**2-r0)))[r>r0]
            del x,y, r
        arrayff=pyptyfft.shift_fft2(array, axes=(1,2), overwrite_x=True)
        if len(shape)==6:
            arrayff=arrayff*mask[None, :,:, None,None,None]
        if len(shape)==5:
            arrayff=arrayff*mask[None, :,:, None, None]
        if len(shape)==4:
            arrayff=arrayff*mask[None, :,:, None]
        if len(shape)==3:
            arrayff=arrayff*mask[None, :,:]
        arrayff=pyptyfft.ifft2_ishift(arrayff, axes=(1,2), overwrite_x=True)
        del mask
        return arrayff
    else:
        return array
    

def fourier_clean(array, cutoff=0.66, mask=None, rolloff=0, default_float=cp.float32, xp=cp):
    """
    Apply a Fourier filter to the input array. Supports 2D or 3D arrays.
    
    Parameters
    ----------
    array : array_like
        Input array (2D or 3D) to be filtered.
    cutoff : float, optional
        Cutoff frequency (default is 0.66).
    mask : array_like or None, optional
        Predefined mask to apply. If None, a mask is generated.
    rolloff : float, optional
        Rolloff parameter for smoothing the mask (default is 0).
    default_float : data-type, optional
        Data type for computations (default is cp.float32).
    xp : module, optional
        Array module (default is cp).
    
    Returns
    -------
    array_like
        The filtered array after applying the Fourier filter.
    """
    if not(cutoff is None) or not(mask is None):
        shape=array.shape
        if mask is None:
            x=pyptyfft.fftshift(pyptyfft.fftfreq(shape[1]))
            y=pyptyfft.fftshift(pyptyfft.fftfreq(shape[0]))
            x,y=cp.meshgrid(x,y, indexing="xy")
            r=x**2+y**2
            max_r=0.5*cutoff ## maximum freq for fftfreq is 0.5 when the px size is not specified
            mask=(r<=max_r**2).astype(default_float)
            if rolloff!=0:
                r0=(0.5*(cutoff-rolloff))**2
                mask[r>r0]*=0.5*(1+cp.cos(3.141592654*(r-r0)/(max_r**2-r0)))[r>r0]
            del x,y, r
        arrayff=pyptyfft.shift_fft2(array, axes=(0,1))
        if len(shape)==4:
            arrayff=arrayff*mask[:,:, None, None]
        if len(shape)==3:
            arrayff=arrayff*mask[:,:, None]
        if len(shape)==2:
            arrayff=arrayff*mask
        arrayff=pyptyfft.ifft2_ishift(arrayff, axes=(0,1))
        del mask
        return arrayff
    else:
        return array
        
        
def create_spatial_frequencies(px, py, shape, damping_cutoff_multislice, smooth_rolloff, default_float):
    """
    Generate spatial frequency grids and corresponding masks for multislice simulations.
    
    Parameters
    ----------
    px : float
        Pixel size in the x-direction.
    py : float
        Pixel size in the y-direction.
    shape : int
        Size of the grid.
    damping_cutoff_multislice : float
        Damping cutoff factor for multislice simulations.
    smooth_rolloff : float
        Smoothing rolloff parameter.
    default_float : data-type
        Data type for computations.
    
    Returns
    -------
    tuple
        Tuple containing:
        - q2: 2D array of squared spatial frequencies.
        - qx: 2D array of spatial frequencies in x.
        - qy: 2D array of spatial frequencies in y.
        - exclude_mask: Mask in Fourier space.
        - exclude_mask_ishift: Unshifted mask.
    """
    qx,qy= cp.meshgrid(pyptyfft.fftfreq(shape, px), pyptyfft.fftfreq(shape, py), indexing="xy")
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
    exclude_mask=pyptyfft.fftshift(exclude_mask_ishift)
    return q2, qx, qy, exclude_mask, exclude_mask_ishift
def shift_probe_fourier(probe, shift_px):
    """
    Shift a probe in Fourier space by applying a phase ramp.
    
    Parameters
    ----------
    probe : array_like
        The input probe array.
    shift_px : tuple of float
        Shift in pixels (y, x).
    
    Returns
    -------
    tuple
        Tuple containing the shifted probe, the phase mask, the Fourier transform of the probe,
        and the spatial frequency grids (maskx, masky).
    """
    maskx, masky=cp.meshgrid(pyptyfft.fftfreq(probe.shape[1]), pyptyfft.fftfreq(probe.shape[0]), indexing="xy")
    mask=cp.exp(-6.283185307179586j*(maskx*shift_px[1]+masky*shift_px[0]))
    phat=pyptyfft.fft2(probe, axes=(0,1))
    probe=pyptyfft.ifft2(mask[:,:,None]*phat, axes=(0,1))
    return probe, mask, phat, maskx, masky
def generate_mask_for_grad_from_pos(shapex, shapey, positions_list, shape_footprint_x,shape_footprint_y, shrink=0):
    """
    Construct a binary mask from given positions and footprint dimensions.
    
    Parameters
    ----------
    shapex : int
        Width of the mask.
    shapey : int
        Height of the mask.
    positions_list : list of tuple
        List of (y, x) positions where the mask should be activated.
    shape_footprint_x : int
        Footprint width.
    shape_footprint_y : int
        Footprint height.
    shrink : int, optional
        Shrink factor to adjust the footprint (default is 0).
    
    Returns
    -------
    array_like
        The constructed binary mask.
    """
    mask=cp.zeros((shapey,shapex))
    for p in positions_list:
        py,px=p
        mask[shrink+py:py+shape_footprint_y-shrink,shrink+px:px+shape_footprint_x-shrink]=1
    return mask
def complex_grad_to_phase_grad(grad, array):
    """
    Convert a Wirtinger derivative to the gradient with respect to the phase.
    
    Parameters
    ----------
    grad : array_like
        The Wirtinger derivative (dL/dz*).
    array : array_like
        The complex array (z = |z| exp(i*phase)).
    
    Returns
    -------
    array_like
        The phase gradient (dL/dp).
    """
    return 2*cp.real(-1j*cp.conjugate(array)*grad)
def complex_grad_to_phase_abs_grad(grad, array):
    """
    Compute the phase gradient and negative amplitude gradient from a complex gradient.
    
    Parameters
    ----------
    grad : array_like
        The Wirtinger derivative (dL/dz*).
    array : array_like
        The complex array (z = exp(-a + i*phase)).
    
    Returns
    -------
    tuple of array_like
        A tuple containing:
        - Phase gradient (dL/dp).
        - Negative amplitude gradient (dL/da).
    """
    array_real, array_imag, grad_real, grad_imag = cp.real(array), cp.imag(array), cp.real(grad), cp.imag(grad)
    return 2*(grad_imag*array_real - array_imag*grad_real), -2*(array_real*grad_real + array_imag*grad_imag)
    
def complex_grad_to_mag_grad(grad, abs, phase):
    """
    Calculate a magnitude gradient from a complex gradient and separate magnitude and phase arrays.
    
    Parameters
    ----------
    grad : array_like
        The complex gradient.
    abs : array_like
        The magnitude array.
    phase : array_like
        The phase array.
    
    Returns
    -------
    array_like
        The magnitude gradient.
    """
    array = abs * cp.exp(-1j * phase)
    return 4 * cp.real(grad * array)

def construct_update_abs_proto_phase(object_grad, obj):
    """
    Compute object updates projected along phase gradients.
    
    Parameters
    ----------
    object_grad : array_like
        The gradient of the object.
    obj : array_like
        The current object array.
    
    Returns
    -------
    array_like
        The computed update for the object.
    """
    obj_real, obj_imag, grad_real, grad_imag = cp.real(obj), cp.imag(obj), cp.real(object_grad), cp.imag(object_grad)
    phase_grad, abs_grad = 2*(grad_imag*obj_real - obj_imag*grad_real), -2*(obj_real*grad_real + obj_imag*grad_imag)
    abs_grad = phase_grad * cp.sum(phase_grad * abs_grad) / (1e-20 + cp.sum(phase_grad**2))
    real_imag_prod = obj_real * obj_imag
    obj_update = 0.5*(1j*abs_grad + phase_grad)*cp.conjugate(obj) / (cp.sign(real_imag_prod)*(1e-20+cp.abs(real_imag_prod)))
    return obj_update
    




def wolfe_1(value, new_value, d_value, step, wolfe_c1=0.5):
    """
    Check the Armijo condition (Wolfe condition 1) for line search.

    Parameters
    ----------
    value : float
        The current function value.
    new_value : float
        The function value after the step.
    d_value : float
        The directional derivative at the current point.
    step : float
        Step size.
    wolfe_c1 : float, optional
        Armijo condition constant (default is 0.5).

    Returns
    -------
    bool
        True if the condition is satisfied, False otherwise.
    """
    return new_value <= value+wolfe_c1*step*d_value
    
def wolfe_2(d_value, new_d_value, wolfe_c2=0.9):
    """
    Check the curvature condition (Wolfe condition 2) for line search.

    Parameters
    ----------
    d_value : float
        The directional derivative at the current point.
    new_d_value : float
        The directional derivative after the step.
    wolfe_c2 : float, optional
        Curvature condition constant (default is 0.9).

    Returns
    -------
    bool
        True if the condition is satisfied, False otherwise.
    """
    return -1*new_d_value <= -1* d_value * wolfe_c2

def upsample_something_3d(something, upsample, scale=True, xp=np):
    """
    Upsample a 3D array along the last two axes.

    Parameters
    ----------
    something : ndarray
        The 3D array to be upsampled.
    upsample : int
        Upsampling factor.
    scale : bool, optional
        If True, scales the upsampled result to conserve total sum (default is True).
    xp : module, optional
        Array module, e.g., numpy or cupy (default is numpy).

    Returns
    -------
    ndarray
        The upsampled 3D array.
    """
    if scale:
        something /= (upsample ** 2)
    something_new = xp.repeat(xp.repeat(something, upsample, axis=1), upsample, axis=2)
    return something_new
    


def downsample_something_3d(something, upsample, xp):
    """
    Downsample a 3D array along the last two axes.

    Parameters
    ----------
    something : ndarray
        The 3D array to be downsampled.
    upsample : int
        Downsampling factor.
    xp : module
        Array module, e.g., numpy or cupy.

    Returns
    -------
    ndarray
        The downsampled 3D array.
    """
    shape = something.shape
    #rem1, rem2=(shape[1]) % upsample, (shape[2]) % upsample
    something=something[:, :upsample*(shape[1] // upsample), :upsample*(shape[1] // upsample)]
    something_reshaped = something.reshape(shape[0], shape[1] // upsample, upsample, shape[2] // upsample, upsample)
    something_new = something_reshaped.sum(axis=(2, 4))
    return something_new


def upsample_something(something, upsample, scale=True, xp=np):
    """
    Upsample a 2D array.

    Parameters
    ----------
    something : ndarray
        The 2D array to be upsampled.
    upsample : int
        Upsampling factor.
    scale : bool, optional
        If True, scales the result to conserve total sum (default is True).
    xp : module, optional
        Array module (default is numpy).

    Returns
    -------
    ndarray
        The upsampled array.
    """
    if scale:
        something /= (upsample ** 2)
    something_new = xp.repeat(xp.repeat(something, upsample, axis=0), upsample, axis=1)
    return something_new
    
def downsample_something(something, upsample, xp):
    """
    Downsample a 2D array.

    Parameters
    ----------
    something : ndarray
        The 2D array to be downsampled.
    upsample : int
        Downsampling factor.
    xp : module
        Array module, e.g., numpy or cupy.

    Returns
    -------
    ndarray
        The downsampled array.
    """
    shape = something.shape
    something=something[ :upsample*(shape[1] // upsample), :upsample*(shape[1] // upsample)]
    something_reshaped = something.reshape(shape[0] // upsample, upsample, shape[1] // upsample, upsample)
    something_new = something_reshaped.sum(axis=(1, 3))
    return something_new
    

def preprocess_dataset(dataset, load_one_by_one, algorithm_type, recon_type, data_shift_vector, data_bin, data_pad, upsample_pattern, data_multiplier, xp, force_pad):
    """
    Apply preprocessing steps to the dataset including shifting, binning, padding, and scaling.

    Parameters
    ----------
    dataset : ndarray
        The input dataset.
    load_one_by_one : bool
        Whether data is loaded incrementally.
    algorithm_type : str
        Type of reconstruction algorithm.
    recon_type : str
        Type of reconstruction (e.g., near_field, far_field).
    data_shift_vector : list of int
        Vector indicating pixel shift in y and x.
    data_bin : int
        Binning factor.
    data_pad : int
        Padding size.
    upsample_pattern : int
        Upsampling factor for the pattern.
    data_multiplier : float
        Factor to scale data intensity.
    xp : module
        Array module, e.g., numpy or cupy.
    force_pad : bool
        If True, apply forced padding.

    Returns
    -------
    tuple
        Tuple containing:
        - preprocessed dataset
        - data_shift_vector
        - data_bin
        - data_pad
        - data_multiplier
    """
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
    """
    Create a circular cosine-tapered window mask.

    Parameters
    ----------
    shape : int
        Size of the square window.
    r0 : float
        Inner radius where tapering begins (normalized).
    r_max : float
        Outer radius where mask falls to zero (normalized).
    inverted : bool, optional
        If True, returns 1 - mask (default is True).

    Returns
    -------
    ndarray
        A 2D mask array of the specified shape.
    """
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
    """
    Convert a number of aberration terms to (n, m, ab) indices based on Krivanek notation.

    Parameters
    ----------
    num_abs : int
        Number of aberration coefficients.

    Returns
    -------
    tuple of lists
        Lists of n, m, and ab strings ('', 'a', or 'b') for each aberration mode.
    """
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
    """
    Convert aberration indices into string identifiers in Krivanek notation.

    Parameters
    ----------
    possible_n : list of int
        List of radial indices.
    possible_m : list of int
        List of azimuthal indices.
    possible_ab : list of str
        List of aberration mode types ('', 'a', 'b').

    Returns
    -------
    list of str
        List of formatted aberration identifiers like 'C30a', 'C11', etc.
    """
    stings=[]
    for i in range(len(possible_n)): stings.append("C%d%d%s"%(possible_n[i], possible_m[i], possible_ab[i]));
    return stings
    

def get_ctf_matrix(kx, ky, num_abs, wavelength, xp=cp):
    """
    Generate a matrix of phase contributions for all aberration modes.

    Parameters
    ----------
    kx : ndarray
        Spatial frequency in x-direction.
    ky : ndarray
        Spatial frequency in y-direction.
    num_abs : int
        Number of aberration coefficients.
    wavelength : float
        Electron wavelength.
    xp : module, optional
        Array module (default is cupy).

    Returns
    -------
    ndarray
        list of Zernike polynomials (num_abs, height, width) with phase contributions.
    """
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
    """
    Compute the scalar contrast transfer function (CTF) from aberrations.

    Parameters
    ----------
    aberrations : list or ndarray
        List of aberration coefficients.
    kx : ndarray
        Spatial frequency in x-direction.
    ky : ndarray
        Spatial frequency in y-direction.
    wavelength : float
        Electron wavelength.
    angle_offset : float, optional
        Additional rotation angle in radians (default is 0).

    Returns
    -------
    ndarray
        The computed CTF.
    """
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
    """
    Compute spatial derivatives of the CTF with respect to kx and ky.

    Parameters
    ----------
    aberrations : list or ndarray
        List of aberration coefficients.
    kx : ndarray
        Spatial frequency in x-direction.
    ky : ndarray
        Spatial frequency in y-direction.
    wavelength : float
        Electron wavelength.
    angle_offset : float, optional
        Additional rotation angle (default is 0).

    Returns
    -------
    tuple of ndarray
        Derivatives of CTF with respect to kx and ky.
    """
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
    """
    Compute the gradient of the phase with respect to rotation angle.

    Parameters
    ----------
    aberrations : list or ndarray
        List of aberration coefficients.
    kx : ndarray
        Spatial frequency in x-direction.
    ky : ndarray
        Spatial frequency in y-direction.
    wavelength : float
        Electron wavelength.
    angle_offset : float, optional
        Additional angular offset (default is 0).

    Returns
    -------
    tuple of ndarray
        Derivatives of the CTF gradient in x and y directions with respect to angular change.
    """
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

def apply_defocus_probe(probe, distance, acc_voltage, pixel_size_x_A, pixel_size_y_A, default_complex, default_float, xp):
    """
    Apply a defocus phase shift to a probe in Fourier space.

    Parameters
    ----------
    probe : ndarray
        The input probe wavefunction.
    distance : float
        Defocus distance in meters.
    acc_voltage : float
        Acceleration voltage in kiloelectronvolts (keV).
    pixel_size_x_A : float
        Pixel size along x in angstroms.
    pixel_size_y_A : float
        Pixel size along y in angstroms.
    default_complex : dtype
        Complex data type for computation.
    default_float : dtype
        Float data type for computation.
    xp : module
        Numerical backend (NumPy or CuPy).

    Returns
    -------
    ndarray
        The defocused probe.
    """
    wavelength=12.4/np.sqrt(acc_voltage*(acc_voltage+2*511))
    q2, qx, qy, exclude_mask, exclude_mask_ishift=create_spatial_frequencies(pixel_size_x_A, pixel_size_y_A,probe.shape[0], 2/3, 0, default_float)
    q2, exclude_mask_ishift, =xp.asarray(q2), xp.asarray(exclude_mask_ishift)
    propagator_phase_space=np.exp(-1j*xp.pi*distance*wavelength*q2)*exclude_mask_ishift
    if len(probe.shape)==3:
        wave_fourier=pyptyfft.fft2(probe, axes=(0,1))*propagator_phase_space[:,:,None]
    else:
        wave_fourier=pyptyfft.fft2(probe, axes=(0,1))*propagator_phase_space[:,:,None, None]
    return pyptyfft.ifft2(wave_fourier, axes=(0,1))


def padfft(array, pad):
    """
    Pad the input array in Fourier space by padding its FFT.

    Parameters
    ----------
    array : ndarray
        Input array to be padded.
    pad : int
        Number of pixels to pad on each side.

    Returns
    -------
    ndarray
        The padded array in spatial domain.
    """
    array=np.fft.fftshift(np.fft.fft2(array, axes=(0,1)), axes=(0,1))
    a2=np.zeros((array.shape[0]+2*pad, array.shape[1]+2*pad, array.shape[2]), dtype=np.complex128)
    a2[pad:-pad, pad:-pad,:]=array
    return np.fft.ifft2(np.fft.ifftshift(a2, axes=(0,1)), axes=(0,1))


def padprobetodatafarfield(probe, measured_data_shape, data_pad, upsample_pattern):
    """
    Pad or crop a probe in Fourier space to match far-field data dimensions.

    Parameters
    ----------
    probe : ndarray
        The probe wavefunction.
    measured_data_shape : tuple
        Shape of the measured data.
    data_pad : int
        Padding applied to the data.
    upsample_pattern : int
        Upsampling factor used in the reconstruction.

    Returns
    -------
    ndarray
        Adjusted probe wavefunction.
    """
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
    """
    Pad or crop a probe for near-field reconstruction.
    
    This function adjusts the probe wavefunction by padding or cropping it to match the
    near-field measured data dimensions after upsampling and padding.
    
    Parameters
    ----------
    probe : ndarray
        The input probe wavefunction.
    measured_data_shape : tuple
        Shape of the measured data.
    data_pad : int
        Padding size applied to the data.
    upsample_pattern : int
        Upsampling factor applied to the measured data.
    
    Returns
    -------
    ndarray
        The adjusted probe wavefunction.
    """
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
    
    



def save_updated_arrays(output_folder, epoch,current_probe_step, current_probe_pos_step, current_tilts_step,current_obj_step, obj, probe, tilts_correction, full_pos_correction, positions, tilts, static_background, current_aberrations_array_step, current_static_background_step,count, current_loss, current_sse, aberrations, beam_current, current_beam_current_step, save_flag, save_loss_log, constraint_contributions, actual_step, count_linesearch, d_value, new_d_value,current_update_step_bfgs, t0, xp, warnings):
    """
    Save current reconstruction state and log loss metrics during training.

    This function saves checkpoints for object, probe, tilts, scan positions, static background, 
    aberration coefficients, and beam current if specified. It also logs loss and constraint 
    contributions in a CSV file if logging is enabled.

    Parameters
    ----------
    output_folder : str
        Directory where files will be saved.
    epoch : int
        Current epoch number.
    current_probe_step : bool
        Whether to save the current probe.
    current_probe_pos_step : bool
        Whether to save current scan positions.
    current_tilts_step : bool
        Whether to save current tilts.
    current_obj_step : bool
        Whether to save the current object.
    obj : ndarray or xp.ndarray
        Object array to save.
    probe : ndarray or xp.ndarray
        Probe array to save.
    tilts_correction : ndarray
        Tilt correction values.
    full_pos_correction : ndarray
        Sub-pixel scan position correction.
    positions : ndarray
        Integer scan positions.
    tilts : ndarray
        Original tilt values.
    static_background : ndarray or xp.ndarray
        Static background array.
    current_aberrations_array_step : bool
        Whether to save aberration array.
    current_static_background_step : bool
        Whether to save static background.
    count : int
        Not used inside the function.
    current_loss : float
        Current loss value.
    current_sse : float
        Current sum of squared errors.
    aberrations : ndarray or xp.ndarray
        Array of aberration coefficients.
    beam_current : ndarray or xp.ndarray
        Array of beam current values.
    current_beam_current_step : bool
        Whether to save beam current.
    save_flag : bool
        Whether to trigger checkpoint saving.
    save_loss_log : bool or int
        Whether to log loss. If set to 2, log full breakdown of constraints.
    constraint_contributions : list
        List of constraint term contributions to the loss.
    actual_step : float
        Step size applied in the optimizer.
    count_linesearch : int
        Number of line search iterations.
    d_value : float
        Initial directional derivative.
    new_d_value : float
        New directional derivative after the step.
    current_update_step_bfgs : float
        Step size suggested by BFGS or optimizer.
    t0 : float
        Start time of the epoch (used for timing).
    xp : module
        NumPy or CuPy module used for computation.
    warnings : str
        Warning string to be logged.

    Returns
    -------
    None
    """
    if save_loss_log:
        with open(output_folder+"loss.csv", mode='a', newline='') as loss_list:
            if save_loss_log==2:
                fieldnames=["epoch", "time / s", "loss", "sse", "initial step", "matching step", "N linesearch iterations", "dir. derivative", "new dir. derivative", "F-axis postions reg.", "Deformation positons reg.", "Deformation tilts reg.", "F-axis tilts reg.", "l1 object reg. (phase)","l1 object reg. (abs)", "Q-space probe reg.", "R-space probe reg.", "TV object reg.", "V-object reg.", "MW-object reg.",  "S-axis postions reg", "S-axis tilts reg", "Free GiB", "Total GiB", "Warnings"]

            else:
                fieldnames=["epoch", "time / s", "loss", "sse", "initial step", "matching step", "N linesearch iterations",
                "dir. derivative", "new dir. derivative", "Constraints contribution", "Free GiB", "Total GiB", "Warnings"]
            if xp==np:
                total_allocated,total_reserved, total_mem_device, free_mem_device=0,0,0,0
            else:
                device = cp.cuda.Device(0)
                total_mem_device=  device.mem_info[1] / (1024 **3)
                free_mem_device=   device.mem_info[0] / (1024 **3)
            write_loss=csv.DictWriter(loss_list,fieldnames=fieldnames)
            if save_loss_log==2:
                write_loss.writerow({"epoch": epoch,
                                "time / s": time.time()-t0,
                                "loss": current_loss,
                                "sse": current_sse,
                                "initial step": current_update_step_bfgs,
                                "matching step": actual_step,
                                "N linesearch iterations": count_linesearch+1,
                                "dir. derivative": d_value,
                                "new dir. derivative": new_d_value,
                                "F-axis postions reg.": constraint_contributions[0],
                                "Deformation positons reg.": constraint_contributions[1],
                                "Deformation tilts reg.": constraint_contributions[2],
                                "F-axis tilts reg.": constraint_contributions[3],
                                "l1 object reg. (phase)": constraint_contributions[4],
                                "l1 object reg. (abs)": constraint_contributions[5],
                                "Q-space probe reg.": constraint_contributions[6],
                                "R-space probe reg.": constraint_contributions[7],
                                "TV object reg.": constraint_contributions[8],
                                "V-object reg.": constraint_contributions[9],
                                "MW-object reg.": constraint_contributions[10],
                                "S-axis postions reg": constraint_contributions[11],
                                "S-axis tilts reg": constraint_contributions[12],
                                "Free GiB":  free_mem_device,
                                "Total GiB": total_mem_device,
                                "Warnings": warnings,
                                })
            else:
                for dumbi1 in range(len(constraint_contributions)):
                    try:
                        constraint_contributions[dumbi1]=constraint_contributions[dumbi1].get()
                    except:
                        pass
                ssum_constr=np.sum(constraint_contributions)
                write_loss.writerow({"epoch": epoch,
                                "loss": current_loss,
                                "time / s": time.time()-t0,
                                "sse": current_sse,
                                "initial step": current_update_step_bfgs,
                                "matching step": actual_step,
                                "N linesearch iterations": count_linesearch+1,
                                "dir. derivative": d_value,
                                "new dir. derivative": new_d_value,
                                "Constraints contribution": ssum_constr,
                                "Free GiB": free_mem_device,
                                "Total GiB":total_mem_device,
                                "Warnings": warnings,
                                })
                        
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
    """
    Generate an initial static background if none is provided.

    Parameters
    ----------
    static_background : float or ndarray
        Initial static background value or None.
    probe : ndarray
        Probe wavefunction.
    damping_cutoff_multislice : float
        Maximum spatial frequency used.
    data_pad : int
        Padding to be applied.
    upsample_pattern : int
        Upsampling factor used.
    default_float : dtype
        Data type for output.
    recon_type : str
        Type of reconstruction ('near_field' or 'far_field').

    Returns
    -------
    ndarray
        Initialized static background.
    """
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
    
def create_probe_from_nothing(probe, data_pad, mean_pattern, aperture_mask, tilt_mode, tilts, dataset, estimate_aperture_based_on_binary, pixel_size_x_A, acc_voltage, data_multiplier, masks, data_shift_vector, data_bin, upsample_pattern, default_complex_cpu, print_flag, algorithm, measured_data_shape, n_obj_modes, probe_marker, recon_type, defocus_array, Cs, skip_preprocessing):
    """
    Generate an initial probe guess when no valid probe is provided.
    
    Depending on the input, this function either uses an aperture mask, computes a mean pattern
    from the dataset, or adjusts an existing mean pattern to generate a probe. It applies shifting,
    binning, padding, and scaling to produce a probe suitable for the specified reconstruction type.
    
    Parameters
    ----------
    probe : ndarray, str, or None
        Input probe. If set to "aperture", the aperture mask is used. If None, the probe is generated
        based on the mean pattern.
    data_pad : int
        Padding size applied to the data.
    mean_pattern : ndarray or None
        Mean pattern used to generate the probe if no probe is provided.
    aperture_mask : ndarray
        Aperture mask used when probe is set to "aperture".
    tilt_mode : bool
        Flag indicating if tilt mode is active.
    tilts : ndarray
        Tilt values.
    dataset : ndarray
        Measured dataset.
    estimate_aperture_based_on_binary : bool or float
        Factor used to threshold the dataset for aperture estimation.
    pixel_size_x_A : float
        Pixel size in the x-direction in angstroms.
    acc_voltage : float
        Acceleration voltage in keV.
    data_multiplier : float
        Factor to scale the data intensity.
    masks : ndarray or None
        Optional masks to apply to the mean pattern.
    data_shift_vector : list or tuple of int
        Vector indicating the shift to be applied to the data.
    data_bin : int
        Binning factor.
    upsample_pattern : int
        Upsampling factor applied to the pattern.
    default_complex_cpu : dtype
        Complex data type for CPU computations.
    print_flag : int
        Flag controlling verbosity.
    algorithm : str
        Identifier for the reconstruction algorithm.
    measured_data_shape : tuple
        Shape of the measured data.
    n_obj_modes : int
        Number of object modes.
    probe_marker : ndarray or None
        Marker array for probe scenarios.
    recon_type : str
        Type of reconstruction ("near_field" or "far_field").
    defocus_array : ndarray
        Array of defocus values.
    Cs : float
        Spherical aberration coefficient.
    
    Returns
    -------
    ndarray
        The generated probe as a complex array.
    """
    meanpat_was_None=False
    if type(probe)!=np.ndarray:
        if probe=="aperture":
            probe=np.expand_dims(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift((aperture_mask)))),-1)
        if probe is None:
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
                    meanpat_was_None=True
                    mean_pattern=np.mean(dataset[:1000], axis=0)*data_multiplier
                else:
                    mean_pattern*=data_multiplier
            if not(masks is None) and meanpat_was_None:
                mean_pattern=np.sum(masks*mean_pattern[:,None, None], axis=0)
                if data_pad!=0:
                    mean_pattern=mean_pattern[data_pad:-data_pad, data_pad:-data_pad]
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
    if not("compressed" in algorithm) and (masks is None):
        if recon_type=="far_field":
            probe=padprobetodatafarfield(probe, measured_data_shape, data_pad, upsample_pattern)
        else:
            probe=padprobetodatanearfield(probe, measured_data_shape, data_pad, upsample_pattern)
    if skip_preprocessing:
        if recon_type=="far_field":
            probe_counts_must = np.sum(dataset[:1000])/((dataset[:1000]).shape[0] * probe.shape[0] * probe.shape[1])
        else:
            probe_counts_must = np.sum(dataset[:1000])/((dataset[:1000]).shape[0])
        
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
    """
    Generate Hermite polynomial-based probe modes from a main mode.

    Parameters
    ----------
    main_mode : ndarray
        The main probe mode.
    n_herm_x : int
        Max Degree of Hermite polynomials in x.
    n_herm_y : int
        Max Degree of Hermite polynomials in y.
    default_complex : dtype
        Complex data type to use.
    xp : module
        Numerical backend.

    Returns
    -------
    ndarray
        Stack of Hermite-based probe modes.
    """
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
    """
    Apply defocus, aberrations, Hermite mode generation, and other modulations to the probe.

    Parameters
    ----------
    probe : ndarray
        Initial probe.
    extra_probe_defocus : float
        Defocus distance to apply.
    acc_voltage : float
        Accelerating voltage in keV.
    pixel_size_x_A : float
        Pixel size in x (Å).
    pixel_size_y_A : float
        Pixel size in y (Å).
    aberrations : list or ndarray
        List of aberration coefficients.
    print_flag : bool
        Whether to print info.
    beam_ctf : ndarray or None
        Optional beam CTF to apply.
    n_hermite_probe_modes : tuple or None
        Number of Hermite modes in (y, x).
    defocus_spread_modes : list or None
        Defocus values to generate additional modes.
    probe_marker : ndarray or None
        Probe assignment array for multi-scenario.
    default_complex : dtype
        Complex type.
    default_float : dtype
        Float type.
    xp : module
        Numerical backend.

    Returns
    -------
    ndarray
        Modulated probe array.
    """
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
            probe=pyptyfft.ifft2_ishift(pyptyfft.shift_fft2(probe)*xp.exp(-1j*ctf)[:,:,None])
        else:
            probe=pyptyfft.ifft2_ishift(pyptyfft.shift_fft2(probe)*xp.exp(-1j*ctf)[:,:,None, None])
    if not(beam_ctf is None):
        beam_ctf= xp.asarray(beam_ctf)
        if beam_ctf.shape[0]!=probe.shape[0]:
            beam_ctf=np.pad(beam_ctf, data_pad)
        if len(probe.shape)==3:
            probe=pyptyfft.ifft2_ishift(pyptyfft.shift_fft2(probe)*xp.exp(-1j*beam_ctf)[:,:,None])
        else:
            probe=pyptyfft.ifft2_ishift(pyptyfft.shift_fft2(probe)*xp.exp(-1j*beam_ctf)[:,:,None,None])
        if print_flag:
            sys.stdout.write("\nUsing the provided CTF for the beam initial guess!\n")
    if not(defocus_spread_modes is None):
        if probe_marker is None:
            p_final=cp.zeros((probe.shape[0], probe.shape[1], len(defocus_spread_modes)), dtype=default_complex)
            for inddef,defocus in enumerate(defocus_spread_modes):
                p_final[:,:,inddef]=apply_defocus_probe(probe[:,:,:1], defocus, acc_voltage, pixel_size_x_A, pixel_size_y_A, default_complex, default_float, xp)[:,:,0]
            probe=p_final
        else:
            probe2=cp.zeros((probe.shape[0], probe.shape[1], len(defocus_spread_modes), probe.shape[3]), dtype=default_complex)
            for i_sc in range(0, probe.shape[3]):
                p_final=cp.zeros((probe.shape[0], probe.shape[1], len(defocus_spread_modes)), dtype=default_complex)
                for inddef,defocus in enumerate(defocus_spread_modes):
                    p_final[:,:,inddef]=apply_defocus_probe(probe[:,:,:1, i_sc], defocus, acc_voltage, pixel_size_x_A, pixel_size_y_A, default_complex, default_float, xp)[:,:,0]
                probe2[:,:,:, i_sc]=p_final
            probe=probe2*1
            del probe2, p_final
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


def prepare_main_loop_params(algorithm,probe, obj,positions,tilts, measured_data_shape, acc_voltage,allow_subPixel_shift=True, sequence=None, use_full_FOV=False, print_flag=0, default_float_cpu=np.float64, default_complex_cpu=np.complex128, default_int_cpu=np.int64, probe_constraint_mask=None, aperture_mask=None, extra_space_on_side_px=0, ignore_positions=False):
    """
    Prepare main loop parameters for reconstruction.
    
    This function adjusts scan positions, pads the object if necessary, handles subpixel corrections,
    and computes the electron wavelength based on the accelerating voltage.
    
    Parameters
    ----------
    algorithm : any
        Identifier for the reconstruction algorithm.
    probe : ndarray
        The probe array.
    obj : ndarray
        The object array.
    positions : ndarray
        Array of scan positions.
    tilts : ndarray
        Array of tilt angles.
    measured_data_shape : tuple
        Shape of the measured data.
    acc_voltage : float
        Accelerating voltage in keV.
    allow_subPixel_shift : bool, optional
        If True, compute subpixel corrections (default is True).
    sequence : list or callable, optional
        Sequence of indices for positions (default is None, which uses full range).
    use_full_FOV : bool, optional
        If True, use full field-of-view adjustments (default is False).
    print_flag : int, optional
        Verbosity flag (default is 0).
    default_float_cpu : data-type, optional
        Float data type for CPU computations (default is np.float64).
    default_complex_cpu : data-type, optional
        Complex data type for CPU computations (default is np.complex128).
    default_int_cpu : data-type, optional
        Integer data type for CPU computations (default is np.int64).
    probe_constraint_mask : ndarray or None, optional
        Optional mask for probe constraints.
    aperture_mask : ndarray or None, optional
        Optional aperture mask.
    extra_space_on_side_px : int, optional
        Extra padding (in pixels) to add to scan positions (default is 0).
    
    Returns
    -------
    tuple
        A tuple containing:
        - obj : ndarray
            The padded object array.
        - positions : ndarray
            Adjusted (rounded) scan positions.
        - int
            A placeholder zero (reserved for future use).
        - sequence : list
            The sequence of indices used.
        - wavelength : float
            Computed electron wavelength in angstroms.
        - full_pos_correction : ndarray
            Subpixel corrections for scan positions.
        - tilts_correction : ndarray
            Array of zeros with same shape as tilts (tilt corrections).
        - aperture_mask : ndarray or None
            The probe constraint mask or aperture mask if provided.
    """
    wavelength  = default_float_cpu(12.398 / np.sqrt((2 * 511.0 + acc_voltage) * acc_voltage))  # angstrom
    if sequence is None:
        sequence=list(np.arange(0, measured_data_shape[0],1))
    try:
        full_sequence=sequence(0)
    except:
        full_sequence=sequence
    if ignore_positions:
        extra_space_on_side_px=0
    else:
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
    
def prepare_saving_stuff(output_folder, save_loss_log, epoch_prev):
    """
    Prepare folder and loss CSV for saving training logs.

    Parameters
    ----------
    output_folder : str
        Directory for results.
    save_loss_log : bool
        Whether to save loss values.
    epoch_prev : int
        Previous epoch index.

    Returns
    -------
    None
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
    except:
        pass
    try:
        os.remove(output_folder+"params.pkl")
    except:
        pass
    if save_loss_log and epoch_prev==0:
        os.system("touch "+output_folder+"loss.csv")
        if save_loss_log==2:
            fieldnames=["epoch", "time / s", "loss", "sse", "initial step", "matching step", "N linesearch iterations", "dir. derivative", "new dir. derivative", "F-axis postions reg.", "Deformation positons reg.", "Deformation tilts reg.", "F-axis tilts reg.", "l1 object reg. (phase)","l1 object reg. (abs)", "Q-space probe reg.", "R-space probe reg.", "TV object reg.", "V-object reg.", "MW-object reg.",  "S-axis postions reg", "S-axis tilts reg", "Free GiB", "Total GiB", "Warnings"]
        else:
            fieldnames=["epoch", "time / s", "loss", "sse", "initial step", "matching step", "N linesearch iterations",
                "dir. derivative", "new dir. derivative", "Constraints contribution", "Free GiB", "Total GiB", "Warnings"]
        with open(output_folder+"loss.csv", 'w+', newline='') as loss_list:
            write_loss=csv.DictWriter(loss_list,fieldnames=fieldnames)
            write_loss.writeheader()
        
def print_pypty_header(data_path, output_folder, save_loss_log):
    """
    Print formatted header announcing start of reconstruction.

    Parameters
    ----------
    data_path : str
        Path to the dataset.
    output_folder : str
        Directory where results are saved.
    save_loss_log : bool
        Whether loss logging is enabled.

    Returns
    -------
    None
    """
    sys.stdout.write("\n***************************************************** *************************\n************************ Starting PyPty Reconstruction ***********************\n******************************************************************************\n")
    sys.stdout.write("\nPath to the dataset: %s"%data_path)
    sys.stdout.write("\nSaving the results in %s" %output_folder)
    sys.stdout.write("\nSaving the parameters in %s" %output_folder+"params.pkl")
    if save_loss_log:
        sys.stdout.write("\nThe log file will be saved as %s" %output_folder+"loss.csv")
    sys.stdout.write("\n******************************************************************************\n******************************************************************************\n******************************************************************************")
    sys.stdout.flush()

def save_current_checkpoint_obj_probe(output_folder, obj, probe, tilts_correction, full_pos_correction, positions, tilts, static_background, current_probe_step, current_obj_step, current_probe_pos_step, current_tilts_step, current_static_background_step, current_aberrations_array_step, aberrations_array, beam_current, bcstep, xp):
    """
    Save intermediate reconstruction data as checkpoints.
    
    This function saves the current state of the object, probe, tilt corrections, scan positions,
    static background, and aberrations to disk. It is intended to allow resuming reconstruction
    from the last checkpoint.
    
    Parameters
    ----------
    output_folder : str
        Directory where checkpoint files will be saved.
    obj : ndarray or GPU array
        The current object array.
    probe : ndarray or GPU array
        The current probe array.
    tilts_correction : ndarray
        Correction values for tilt angles.
    full_pos_correction : ndarray
        Sub-pixel correction values for scan positions.
    positions : ndarray
        Scan positions array.
    tilts : ndarray
        Tilt angles array.
    static_background : ndarray
        Static background array.
    current_probe_step : bool
        Flag indicating whether to save the probe.
    current_obj_step : bool
        Flag indicating whether to save the object.
    current_probe_pos_step : bool
        Flag indicating whether to save the scan positions.
    current_tilts_step : bool
        Flag indicating whether to save the tilt angles.
    current_static_background_step : bool
        Flag indicating whether to save the static background.
    current_aberrations_array_step : bool
        Flag indicating whether to save the aberrations array.
    aberrations_array : ndarray or GPU array
        The current aberrations array.
    beam_current : ndarray or GPU array or None
        The current beam current array.
    bcstep : bool
        Flag indicating whether to save the beam current.
    xp : module
        Numerical backend (e.g., numpy or cupy).
    
    Returns
    -------
    None
    """
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
    """
    Display current reconstruction progress including loss, optimization state, and updates.

    Parameters
    ----------
    t0 : float
        Start time of the epoch (Unix timestamp).
    algorithm : str
        Name of the loss or optimization algorithm used.
    epoch : int
        Current training epoch.
    current_loss : float
        Loss value at current epoch.
    current_sse : float
        Sum of squared errors.
    current_obj_step : bool
        Whether the object is being updated.
    current_probe_step : bool
        Whether the probe is being updated.
    current_probe_pos_step : bool
        Whether the scan grid is being updated.
    current_tilts_step : bool
        Whether tilt corrections are being updated.
    current_static_background_step : bool
        Whether static background is being updated.
    current_aberrations_array_step : bool
        Whether aberration coefficients are being updated.
    current_beam_current_step : bool
        Whether beam current is being updated.
    current_hist_length : int
        Optimizer memory length (0=GD, 1=CG, >1=BFGS).
    print_flag : int
        Verbosity flag: 0 = silent, 1 = single-line print, 2 = verbose.

    Returns
    -------
    None
    """
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
            sys.stdout.write("\r---------> Time: %d:%2d:%2d. Epoch %i. Using %s error metric with %s optimzer. Loss: %.2e. SSE: %.2e. %s" % (hours, minutes, seconds,  epoch, algorithm, print_optimizer, current_loss, current_sse, string))
        else:
            sys.stdout.write("\n---------> Time: %d:%2d:%2d. Epoch %i. Using %s error metric with %s optimzer. Loss: %.2e. SSE: %.2e. %s" % (hours, minutes, seconds,  epoch, algorithm, print_optimizer, current_loss, current_sse, string))
        sys.stdout.flush()
        
def try_to_gpu(obj, probe, positions,full_pos_correction, tilts, tilts_correction, masks, defocus_array, slice_distances, aperture_mask, dataset, load_one_by_one, static_background, aberrations_array, beam_current, default_float, default_complex, default_int, xp):
    """
    Convert all key reconstruction variables to GPU arrays if using CuPy.

    Parameters
    ----------
    obj : ndarray
        Object array.
    probe : ndarray
        Probe array.
    positions : ndarray
        Integer scan positions.
    full_pos_correction : ndarray
        Sub-pixel scan grid correction.
    tilts : ndarray
        Tilt values.
    tilts_correction : ndarray
        Tilt corrections.
    masks : ndarray or None
        Optional segmentation or region masks.
    defocus_array : ndarray
        Array of defocus values per position.
    slice_distances : ndarray
        Slice spacing in multislice simulations.
    aperture_mask : ndarray or None
        Probe aperture mask.
    dataset : ndarray
        Measured dataset.
    load_one_by_one : bool
        Whether dataset is streamed from disk.
    static_background : ndarray or None
        Static background array.
    aberrations_array : ndarray or None
        Array of aberration coefficients.
    beam_current : ndarray or None
        Beam current scaling factor.
    default_float : dtype
        Float precision dtype for casting.
    default_complex : dtype
        Complex precision dtype for casting.
    default_int : dtype
        Integer dtype for casting.
    xp : module
        Numerical backend (`numpy` or `cupy`).

    Returns
    -------
    tuple
        The same variables in GPU format (if using CuPy), with proper types.
    """
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
    """
    Initialize beam current array or pad if it's too short.

    Parameters
    ----------
    beam_current : ndarray or None
        Existing beam current values.
    measured_data_shape : tuple
        Shape of measured dataset.
    default_float : dtype
        Float type for the array.
    xp : module
        NumPy or CuPy.

    Returns
    -------
    ndarray
        Initialized or padded beam current.
    """
    if beam_current is None:
        beam_current=beam_current=xp.ones(measured_data_shape[0], dtype=default_float)
    if beam_current.shape[0]<measured_data_shape[0]: beam_current=xp.pad(beam_current, pad_width=measured_data_shape[0]-beam_current.shape[0], mode='constant', constant_values=1)
    return beam_current

def get_value_for_epoch(func_or_value, epoch, default_float):
    """
    Evaluate a list of values or functions at the current epoch.

    Parameters
    ----------
    func_or_value : list
        List of fixed values or callables.
    epoch : int
        Current epoch number.
    default_float : dtype
        Float precision type.

    Returns
    -------
    list
        Evaluated values.
    """
    out=[]
    for f in func_or_value:
        try:
            x=f(epoch)
        except:
            x=f
        out.append(x)
    return out
    
def get_steps_epoch(steps, epoch, default_float):
    """
    Evaluate step values for the current epoch.

    Parameters
    ----------
    steps : list
        List of (multiplier, callable) or fixed values.
    epoch : int
        Current training epoch.
    default_float : dtype
        Float precision type.

    Returns
    -------
    list
        List of step values.
    """
    out=[]
    for s in steps:
        try:
            curr_s=s[0]*s[1](epoch)
        except:
            curr_s=s
        out.append(default_float(curr_s))
    return out


def lambda_to_string(f):
    """
    Extract lambda function source as a string.

    Parameters
    ----------
    f : function
        Lambda function.

    Returns
    -------
    str
        Extracted string source of the lambda.
    """
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
        
def convert_to_string(dicti2, strip_dataset_from_params=True):
    """
    Convert parameter dictionary to string format, including lambda serialization.

    Parameters
    ----------
    dicti2 : dict
        Original parameter dictionary.
    strip_dataset_from_params : bool, optional
        Whether to exclude 'dataset' key (default is True).

    Returns
    -------
    dict
        Dictionary with string values.
    """
    string_params={}
    for key, value in dicti2.items():
        if strip_dataset_from_params and key=="dataset":
            continue
        if isinstance(value, types.LambdaType):
            string_params[key] = lambda_to_string(value)
        else:
            string_params[key] = value
    return string_params
    
def string_to_lambda(lambda_string):
    """
    Convert stringified lambda expression to a Python function.
    Parameters
    ----------
    lambda_string : str
        Lambda string to evaluate.

    Returns
    -------
    callable or str
        The resulting function or original string if evaluation fails.
    """
    try:
        return eval(lambda_string)
    except:
        return lambda_string

def load_params(path):
    """
    Load parameter dictionary from a .pkl file.

    Parameters
    ----------
    path : str
        Path to the .pkl parameter file.

    Returns
    -------
    dict
        Loaded parameters.
    """
    with open(path, 'rb') as handle:
        params = pickle.load(handle)
    return params
    
def string_params_to_usefull_params(params):
    """
    Convert string-encoded lambdas in parameter dictionary back to callables.

    Parameters
    ----------
    params : dict
        Parameter dictionary possibly containing lambda strings.

    Returns
    -------
    dict
        Updated dictionary with callables.
    """
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

def save_params(params_path, params, strip_dataset_from_params):
    """
    Save parameters to a .pkl file, optionally removing the dataset.

    Parameters
    ----------
    params_path : str
        Output path for the parameter file.
    params : dict
        Parameter dictionary to save.
    strip_dataset_from_params : bool
        If True, remove the dataset entry.

    Returns
    -------
    None
    """
    if params_path[-4:]==".pkl":
        try:
            os.remove(params_path)
        except:
            pass
    params_pkl=convert_to_string(params, strip_dataset_from_params)
    with open(params_path, 'wb') as file:
        pickle.dump(params_pkl, file)
    del params_pkl



def phase_cross_corr_align(im_ref_fft, im_2_fft, refine_box_dim, upsample, x_real, y_real, shift_y=None, shift_x=None):
    """
    Align two FFT-transformed images using phase cross-correlation.

    Parameters
    ----------
    im_ref_fft : ndarray
        Reference image FFT.
    im_2_fft : ndarray
        FFT of the image to be aligned.
    refine_box_dim : int
        Size of the interpolation box for sub-pixel alignment.
    upsample : int
        Upsampling factor for interpolation.
    x_real : ndarray
        Real space x grid.
    y_real : ndarray
        Real space y grid.
    shift_y : float or None
        Predefined shift in y (optional).
    shift_x : float or None
        Predefined shift in x (optional).

    Returns
    -------
    ndarray
        Shifted FFT of the second image.
    """
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



def get_cupy_memory_usage():
    """
    Print current CuPy GPU memory usage statistics.

    Returns
    -------
    None
    """
    mempool = cp.get_default_memory_pool()
    total_allocated = mempool.used_bytes()  # Total allocated GPU memory
    total_reserved = mempool.total_bytes()  # Total reserved GPU memory (including fragmentation)
    print("\n")
    print(f"Total GPU Memory Allocated: {total_allocated / (1024 ** 3):.4f} GB\n")
    print(f"Total GPU Memory Reserved: {total_reserved / (1024 ** 3):.4f} GB\n")
    memory_usage = []
    for var_name, var_value in nonlocals().items():
        if isinstance(var_value, cp.ndarray):  # Only check CuPy arrays
            mem_usage_gb = var_value.nbytes / (1024 ** 3)  # Convert bytes to GB
            memory_usage.append((var_name, mem_usage_gb, var_value.shape, var_value.dtype))
    memory_usage.sort(key=lambda x: x[1], reverse=True)
    for var_name, mem_usage_gb, shape, dtype in memory_usage:
        print(f"Variable: {var_name}\nMemory: {mem_usage_gb:.4f} GB\nShape: {shape}\nDtype: {dtype}\n")


def get_compute_batch(compute_batch, load_one_by_one, hist_size, measured_data_shape, memory_saturation, smart_memory, data_pad, obj_shape, probe_shape,  dtype, propmethod, print_flag):
    """
    Estimate the optimal compute batch size based on GPU memory usage.

    Parameters
    ----------
    compute_batch : int
        Initial guess or default.
    load_one_by_one : bool
        Whether data is streamed instead of fully loaded.
    hist_size : int
        History size for optimizers.
    measured_data_shape : tuple
        Shape of the input dataset.
    memory_saturation : float
        Proportion of GPU memory to use.
    smart_memory : callable or bool
        User-provided memory strategy.
    data_pad : int
        Padding applied to data.
    obj_shape : tuple
        Shape of the object array.
    probe_shape : tuple
        Shape of the probe array.
    dtype : str
        Data type string ('single' or 'double').
    propmethod : str
        Propagation method name.
    print_flag : int
        Verbosity.

    Returns
    -------
    tuple
        Suggested batch size, load_one_by_one flag, and memory strategy.
    """
    if cp==np:
        total_mem_device_Gb=0.2
    else:
        device = cp.cuda.Device(0)
        total_mem_device_Gb=  device.mem_info[1] / (1024 **3)
    if propmethod=="multislice":
        waves_shape=2
        inter_wave_multi=2
    if propmethod=="yoshida":
        waves_shape=7
        inter_wave_multi=16
    if propmethod=="better_multislice":
        waves_shape=10
        inter_wave_multi=26
    if dtype=="double":
        n_bytes=16
    else:
        n_bytes=8
    n_meas=measured_data_shape[0]
    probexy=probe_shape[0]*probe_shape[1]
    probexym=probexy*probe_shape[2]
    probexyms= probexym if len(probe_shape)==3 else probexym*probe_shape[3]
    load_one_by_one_memory = (1-load_one_by_one)*np.prod(measured_data_shape) *n_bytes * 0.5 /(1024 ** 3)
    update_memory= n_bytes*((6+2*hist_size) * np.prod(obj_shape) + (6+2*hist_size)*probexyms +  17*probexym + 9*n_meas*(3+1*hist_size)+ (3+1*hist_size)*probexy)/(1024 ** 3)
    per_compute_batch_memory= probexym * obj_shape[2]*obj_shape[3]*waves_shape + 3*probexy + 17 + 7*probexym*obj_shape[3] + probexym*13  + inter_wave_multi* probexym*obj_shape[3]
    per_compute_batch_memory*=n_bytes/(1024 ** 3)
    suggested_compute_batch=int(np.floor((total_mem_device_Gb*memory_saturation -update_memory - load_one_by_one_memory)/per_compute_batch_memory))
    if suggested_compute_batch<=5 and not(load_one_by_one):
        suggested_compute_batch=int(np.floor((total_mem_device_Gb*memory_saturation -update_memory)/per_compute_batch_memory))
        if suggested_compute_batch==0:
            suggested_compute_batch=1
        if print_flag:
            sys.stdout.write("\nWe do not suggest to keep the dataset in the memory and manually set load_one_by_one to True. Optimal compute batch is %d. Predicted memory usage: %.2f GiB"%(suggested_compute_batch, total_mem_device_Gb*memory_saturation))
    elif print_flag:
        sys.stdout.write("\nWe suggest to use compute batch of %d. Predicted memory usage: %.2f GiB"%(suggested_compute_batch, total_mem_device_Gb*memory_saturation))
    sys.stdout.flush()
    try:
        test=smart_memory(0)
    except:
        if total_mem_device_Gb<20:
            smart_memory=True
    return suggested_compute_batch, load_one_by_one, smart_memory



def load_nexus_params(path_nexus):
    """
    Load reconstruction parameters from a NeXus (.nxs) HDF5 file.

    Parameters
    ----------
    path_nexus : str
        Path to the .nxs file.

    Returns
    -------
    dict
        Dictionary of extracted parameters.
    """
    f=h5py.File(path_nexus, "r")
    path_inside='entry/reconstruction/reconstruction parameters'
    pypty_params={}
    for k in f[path_inside].keys():
        if k != "dataset":
            value=f[path_inside+"/"+k][()]
            if isinstance(value, bytes):
                value = value.decode('utf-8')
                if value=="None":
                    value=None
            pypty_params[k]=value
    return pypty_params

def delete_dataset_from_params(params_path):
    """
    Delete the 'dataset' key from saved parameter file.

    Parameters
    ----------
    params_path : str
        Path to the pickled parameters file.

    Returns
    -------
    None
    """
    with open(params_path,'rb') as file:
        data = pickle.load(file)
    if 'dataset' in data:
        del data['dataset']
    with open(params_path,'wb') as file:
        pickle.dump(data,file)




def convert_to_nxs(folder_path, output_file):
    """
    Convert saved PyPty reconstruction data to NeXus (.nxs) format.

    Parameters
    ----------
    folder_path : str
        Directory containing saved reconstruction files.
    output_file : str
        Path where the NeXus file will be saved.

    Returns
    -------
    None
    """
    co_path = os.path.join(folder_path, "co.npy")
    cp_path = os.path.join(folder_path, "cp.npy")
    cg_path = os.path.join(folder_path, "cg.npy")
    pkl_path = os.path.join(folder_path, "params.pkl")
    if not all(os.path.exists(p) for p in [co_path, cp_path, cg_path, pkl_path]):
        raise FileNotFoundError("Missing required files.")
    co = np.load(co_path)
    cpr = np.load(cp_path)
    cg = np.load(cg_path)
    creation_time = datetime.datetime.fromtimestamp(os.path.getmtime(co_path)).isoformat()
    with open(pkl_path, "rb") as f:
        metadata = pickle.load(f)
    pixel_size_y = metadata["pixel_size_y_A"]
    pixel_size_x = metadata["pixel_size_x_A"]
    slice_spacing = metadata.get("slice_distances", [1])[0]
    chemical_formula = metadata.get("chemical_formula", "")
    sample_name = metadata.get("sample_name", None)
    if sample_name is None:
        sample_name= (metadata.get("data_path", "").split("/")[-1]).split(".")[0]
    cg[:, 0] *= pixel_size_y
    cg[:, 1] *= pixel_size_x

    probe_shape = cpr.shape
    is_probe_4d = len(probe_shape) == 4
    
    co = co[::-1, :, :, :].transpose(3, 2, 0, 1)

    # Flip y-axis and reorder axes for probe (modes, scenarios?, y, x)
    if is_probe_4d:
        cpr = cpr[::-1, :, :, :].transpose(2, 3, 0, 1)
    else:
        cpr = cpr[::-1, :, :].transpose(2, 0, 1)

    with h5py.File(output_file, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "object" # open object by default
        
        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        sample.create_dataset("name", data=sample_name)
        if chemical_formula != "":
            sample.create_dataset("chemical_formula", data=chemical_formula)
        # Object data
        obj_grp = entry.create_group("object")
        obj_grp.attrs["NX_class"] = "NXdata"
        obj_grp.create_dataset("data", data=co)
        obj_grp.attrs["axes"] = ["modes", "z", "y", "x"]
        obj_grp.create_dataset("modes", data=np.arange(co.shape[0]))
        obj_grp["modes"].attrs["units"] = "mode index"
        obj_grp.create_dataset("z", data=np.arange(co.shape[1]) * slice_spacing)
        obj_grp["z"].attrs["units"] = "angstrom"
        obj_grp.create_dataset("y", data=np.arange(co.shape[2]) * pixel_size_y)
        obj_grp["y"].attrs["units"] = "angstrom"
        obj_grp.create_dataset("x", data=np.arange(co.shape[3]) * pixel_size_x)
        obj_grp["x"].attrs["units"] = "angstrom"

        # Instrument
        instr_grp = entry.create_group("instrument")
        instr_grp.attrs["NX_class"] = "NXinstrument"

        # Probe data
        probe_grp = instr_grp.create_group("probe")
        probe_grp.attrs["NX_class"] = "NXbeam"
        probe_grp.create_dataset("data", data=cpr)
        probe_axes = ["modes"] + (["scenarios"] if is_probe_4d else []) + ["y", "x"]
        probe_grp.attrs["axes"] = probe_axes
        probe_grp.create_dataset("modes", data=np.arange(cpr.shape[0]))
        probe_grp["modes"].attrs["units"] = "mode index"
        offset = 1
        if is_probe_4d:
            probe_grp.create_dataset("scenarios", data=np.arange(cpr.shape[1]))
            probe_grp["scenarios"].attrs["units"] = "scenario index"
            offset += 1
        probe_grp.create_dataset("y", data=np.arange(cpr.shape[offset]) * pixel_size_y)
        probe_grp["y"].attrs["units"] = "angstrom"
        probe_grp.create_dataset("x", data=np.arange(cpr.shape[offset + 1]) * pixel_size_x)
        probe_grp["x"].attrs["units"] = "angstrom"

        # Scan positions
        scan_grp = entry["instrument"].create_group("scan")
        scan_grp.attrs["NX_class"] = "NXpositioner"
        scan_grp.create_dataset("positions", data=cg)
        scan_grp["positions"].attrs["units"] = "angstrom"
        scan_grp.attrs["axes"] = ["positions", "coordinates"]

        # Reconstruction parameters
        recon_grp = entry.create_group("reconstruction")
        recon_grp.attrs["NX_class"] = "NXprocess"
        recon_grp.create_dataset("software", data="PyPty")
        recon_grp.create_dataset("version", data="v2.0")
        recon_grp.create_dataset("date", data=creation_time)
        recon_grp.create_dataset("folder",  data=metadata.get("output_folder", ""))
        recon_grp.create_dataset("dataset", data=metadata.get("data_path", "").split("/")[-1])
        p_grp=recon_grp.create_group("reconstruction parameters")
        p_grp.attrs["NX_class"] = "NXcollection"
        p_grp.create_dataset("software", data="PyPty")
        for key, value in metadata.items():
            if value is None: value="None";
            p_grp.create_dataset(key, data=value)
    print(f"NeXus file saved as: {output_file}")

def segment_regions_from_image(image: np.ndarray, threshold: float = 0.2, sigma: float = 1.0, dilation_size: int = 3):
    """
    Segment regions in an image based on edge detection and boundary blocking.

    This function applies Gaussian smoothing to the input image, thresholds it to detect edges, 
    dilates the edges to thicken boundaries, and inverts the result to identify regions. 
    Finally, it labels connected regions in the image.

    Parameters
    ----------
    image : np.ndarray
        Input 2D grayscale image.
    threshold : float
        Intensity threshold for edge detection.
    sigma : float
        Standard deviation for Gaussian blur.
    dilation_size : int
        Size of the dilation kernel for boundaries.

    Returns
    -------
    labeled_regions : np.ndarray
        2D array where each region has a unique label.
    num_regions : int
        Total number of regions.
    edge_mask : np.ndarray
        Binary edge map.
    """
    # Step 1: Gaussian smoothing
    img_blur = gaussian(image, sigma=sigma)

    # Step 2: Thresholding to extract edge-like regions
    edge_mask = img_blur > threshold

    # Step 3: Dilate edge mask to thicken boundaries
    dilated_edges = binary_dilation(edge_mask, structure=np.ones((dilation_size, dilation_size)))
    closed = binary_closing(dilated_edges,disk(1))
    # Step 4: Invert to get region mask (where it's not edge)
    region_mask = ~closed

    # Step 5: Label connected regions
    labeled_regions = label(region_mask)
    num_regions = labeled_regions.max()

    return labeled_regions, num_regions, edge_mask

    return new_label_map, num_regions, edge_mask




def interpolate_label0_points(points: np.ndarray, method='linear', fallback='nearest'):
    """
    Interpolate x, y for label == 0 points using (i, j) grid coordinates.
    If primary interpolation fails (e.g. NaNs), use fallback method.

    Parameters:
        points (np.ndarray): (N, 5) array with columns [x, y, i, j, label]
        method (str): Primary interpolation method ('linear', 'cubic', 'nearest')
        fallback (str): Backup method if primary interpolation returns NaN

    Returns:
        interpolated_points (np.ndarray): points with updated x, y for label==0
    """
    x, y, i, j, label = points[:, 0], points[:, 1], points[:, 2], points[:, 3], points[:, 4]

    mask_valid = label != 0
    mask_target = label == 0

    known_coords = np.column_stack((i[mask_valid], j[mask_valid]))
    known_x = x[mask_valid]
    known_y = y[mask_valid]

    target_coords = np.column_stack((i[mask_target], j[mask_target]))

    interp_x = griddata(known_coords, known_x, target_coords, method=method)
    interp_y = griddata(known_coords, known_y, target_coords, method=method)

    # fallback for NaNs
    if fallback and (np.isnan(interp_x).any() or np.isnan(interp_y).any()):
        if np.isnan(interp_x).any():
            interp_x[np.isnan(interp_x)] = griddata(known_coords, known_x, target_coords[np.isnan(interp_x)], method=fallback)
        if np.isnan(interp_y).any():
            interp_y[np.isnan(interp_y)] = griddata(known_coords, known_y, target_coords[np.isnan(interp_y)], method=fallback)

    interpolated_points = points.copy()
    interpolated_points[mask_target, 0] = interp_x
    interpolated_points[mask_target, 1] = interp_y

    return interpolated_points
def optimize_2d_block_shifts(points: np.ndarray):
    """
    Optimize shifts for each block (label) such that the distances between adjacent points
    (based on 2D scan grid) are as uniform as possible.

    Parameters:
        points (np.ndarray): (N, 5) array with columns [x, y, i, j, label]

    Returns:
        shifted_points (np.ndarray): (N, 5) array with updated x, y positions after optimal shifts
        block_shifts (dict): label -> (dx, dy)
    """
    # Step 1: Filter valid (label ≠ 0) points
    x, y, i, j, label = points[:, 0], points[:, 1], points[:, 2].astype(int), points[:, 3].astype(int), points[:, 4].astype(int)
    valid_mask = label > 0
    valid_points = points[valid_mask]
    xv, yv = valid_points[:, 0], valid_points[:, 1]
    iv, jv, lv = valid_points[:, 2].astype(int), valid_points[:, 3].astype(int), valid_points[:, 4].astype(int)
    unique_labels = np.unique(lv)
    label_to_idx = {l: idx for idx, l in enumerate(unique_labels)}
    idx_to_label = {idx: l for l, idx in label_to_idx.items()}
    num_blocks = len(unique_labels)


    shift_init = np.ones((num_blocks, 2)).flatten()
    def allign_shifts(shifts_flat):
        shifts = shifts_flat.reshape((num_blocks, 2))
        shifted_coords = np.zeros((len(valid_points), 2))
        for k in range(len(valid_points)):
            l = lv[k]
            dx, dy = shifts[label_to_idx[l]]
            shifted_coords[k] = np.array([xv[k] + dx, yv[k] + dy])
        shifted_points = points.copy()
        shifted_points[valid_mask,:2] = shifted_coords
        shifted_points = interpolate_label0_points(shifted_points)
        return(shifted_points[:,:2])
    
        
    def objective(shifts_flat):
        shifted_coords = allign_shifts(shifts_flat)
        ## laplace error
        shifted_coords_2d = shifted_coords.reshape(i.max(), j.max(),shifted_coords.shape[-1])
        # Shift along axis 0 (i / row direction)
        roll_p0 = np.roll(shifted_coords_2d, 1, axis=0)
        roll_m0 = np.roll(shifted_coords_2d, -1, axis=0)

        # Shift along axis 1 (j / column direction)
        roll_p1 = np.roll(shifted_coords_2d, 1, axis=1)
        roll_m1 = np.roll(shifted_coords_2d, -1, axis=1)

        # 2D discrete Laplacian: ∆ = f(i+1,j) + f(i-1,j) + f(i,j+1) + f(i,j-1) - 4*f(i,j)
        laplace = roll_p0 + roll_m0 + roll_p1 + roll_m1 - 4 * shifted_coords_2d

        # Optionally set Laplacian at boundaries to 0 to avoid wrap-around artifacts
        laplace[0, :, :] = 0
        laplace[-1, :, :] = 0
        laplace[:, 0, :] = 0
        laplace[:, -1, :] = 0
        laplace_error = (laplace**2).sum()
        ## fast axis loss and slow axis loss
        fast_axis_loss = 0
        slow_axis_loss = 0
        # Group by global column index j
        from collections import defaultdict
        columns = defaultdict(list)
        rows = defaultdict(list)
        for k in range(len(points)):
            columns[j[k]].append((i[k], shifted_coords[k]))  # (scan row, shifted position)
            rows[i[k]].append((j[k], shifted_coords[k]))  # (column j, position)
        # Evaluate loss per column
        for jj, ij_pts in columns.items():
            if len(ij_pts) < 2:
                continue

            ij_pts_sorted = sorted(ij_pts, key=lambda t: t[0])  # sort by i
            pts = np.array([p for _, p in ij_pts_sorted])

            # Fit line using SVD (PCA style)
            mean = pts.mean(axis=0)
            U, S, Vt = np.linalg.svd(pts - mean)
            direction = Vt[0]
            projections = (pts - mean) @ direction

            # 1. Line alignment loss
            residuals = pts - (mean + np.outer(projections, direction))
            alignment_loss = np.sum(np.linalg.norm(residuals, axis=1)**2)

            # 2. Ordering penalty (projection must be increasing along i)
            diffs = np.diff(projections)
            ordering_penalty = np.sum(np.maximum(0, -diffs)**2)
            fast_axis_loss += alignment_loss + ordering_penalty
        for ii, ji_pts in rows.items():
            if len(ji_pts) < 2:
                continue

            # Sort by scan column index j
            ji_pts_sorted = sorted(ji_pts, key=lambda t: t[0])
            pts = np.array([p for _, p in ji_pts_sorted])

            # Fit line via SVD
            mean = pts.mean(axis=0)
            U, S, Vt = np.linalg.svd(pts - mean)
            direction = Vt[0]
            projections = (pts - mean) @ direction

            # 1. Alignment loss (distance to best-fit line)
            residuals = pts - (mean + np.outer(projections, direction))
            alignment_loss = np.sum(np.linalg.norm(residuals, axis=1) ** 2)

            # 2. Monotonicity penalty: enforce projection along direction increases with j
            ordering_penalty = np.sum(np.maximum(0, -np.diff(projections)) ** 2)

            slow_axis_loss += alignment_loss + ordering_penalty
        return(laplace_error+fast_axis_loss+slow_axis_loss)
        
    
    result = minimize(objective, shift_init, method='Powell')
    # print(result)
    optimal_shifts = result.x.reshape((num_blocks, 2))

    shifted_points = points.copy()
    shifted_points[:,:2] = allign_shifts(optimal_shifts)

    block_shifts = {idx_to_label[idx]: tuple(shift) for idx, shift in enumerate(optimal_shifts)}
    return shifted_points[:,:], block_shifts

def position_puzzling(points,scan_size,sigma=0.4,score_threshold=1.2):
    """
    Correct scan positions by segmenting regions and then puzzel them.
    Parameters:
        points (np.ndarray): (N, 2) array with scan positions.
        scan_size (tuple): Size of the scan grid (rows, columns).\
    return:
        num_regions (int): Number of segmented regions.
        shifted_points (np.ndarray): (N, 5) array with corrected scan positions and labels.
    """
    if isinstance(points, cp.ndarray):
        points_np = cp.asnumpy(points.copy())
    else:
        points_np = points.copy()
    print(points_np.shape)
    print(points_np)
    import matplotlib.pyplot as plt
    plt.scatter(points_np[:,1], points_np[:,0], s=1, c='b', marker='o', label="Original positions")
    plt.show()
    x = points_np[:,0].reshape(scan_size)
    y = points_np[:,1].reshape(scan_size)
    laplacex = ndimage.laplace(x)
    laplacey = ndimage.laplace(y)
    laplace = np.sqrt(laplacex**2 + laplacey**2)



    # Step 2: Compute score based on normalized variance
    variance = np.var(laplace)
    mean_abs = np.mean(np.abs(laplace)) + 1e-8  # avoid division by zero
    score = variance / (mean_abs ** 2)
    #  # Step 3: Compute entropy of absolute Laplace (structure complexity)
    # hist, _ = np.histogram(np.abs(laplace).ravel(), bins=100, density=True)
    # ent = entropy(hist + 1e-12)  # avoid log(0)
    puzzling_worked = False
    if score>1 :
        mask = (laplace-laplace.min())/(laplace.max()-laplace.min())
        labeled_regions, num_regions, edge_mask = segment_regions_from_image(mask, threshold=mask.mean(), sigma=sigma, dilation_size=1)
        if num_regions < 2:
            print(f"{num_regions} segmentations found,no need to puzzle,score is {score}")
        else:
            puzzling_worked = True
            x = np.linspace(1,scan_size[1],scan_size[1])
            y = np.linspace(1,scan_size[0],scan_size[0])
            X,Y = np.meshgrid(x,y)
            X = X.reshape(-1)
            Y = Y.reshape(-1)
            points_new = np.zeros((X.shape[0],5))
            points_new[:,0] = points_np[:,0]
            points_new[:,1] = points_np[:,1]
            points_new[:,2] = X[:]
            points_new[:,3] = Y[:]
            points_new[:,4] = labeled_regions.reshape(-1)[:]
            shifted_points,block_shifts = optimize_2d_block_shifts(points_new)
            import matplotlib.pyplot as plt
            plt.scatter(points_new[:,1], points_new[:,0], s=1, c='b', marker='o', label="Full positions")
            plt.scatter(shifted_points[:,1], shifted_points[:,0], s=1, c='r', marker='x', label="Shifted positions")
            plt.legend()
            plt.show()
            points_np = shifted_points[:,:2]
            print(f"{num_regions} segmentations found,start to puzzle,score is {score}")
    else:
        print(f"no segmentation found, using original points,score is {score}")
    return puzzling_worked,points_np


def fit_vector_field(x, y, u, v, Xi,Yi,method='nearest'):
    x,y,u,v=x.flatten(),y.flatten(),u.flatten(),v.flatten()
    Ui = griddata((x, y), u, (Xi, Yi), method=method, fill_value=0)
    Vi = griddata((x, y), v, (Xi, Yi), method=method, fill_value=0)
    return Ui, Vi
def warp_image(image, vector_field, order=1):
    U, V = vector_field
    rows, cols = np.indices(image.shape[:2])
    rows_flatten, cols_flatten = rows.flatten(), cols.flatten()
    warped_rows = rows_flatten + U.flatten()
    warped_cols = cols_flatten + V.flatten()
    warped_image = map_coordinates(image, (warped_rows, warped_cols), order=order, mode='nearest')
    warped_image = warped_image.reshape(image.shape)
    return warped_image
def intorplate_puzzled_obj(array,pos_old,pos_new,order=1,method = 'linear'):
    if isinstance(array, cp.ndarray):
        array = cp.asnumpy(array)
    if isinstance(pos_old, cp.ndarray):
        pos_old = cp.asnumpy(pos_old)
    if isinstance(pos_new, cp.ndarray):
        pos_new = cp.asnumpy(pos_new)
    Xi=np.arange(array.shape[0])
    Yi=np.arange(array.shape[1])
    Xi, Yi=np.meshgrid(Xi,Yi)
    out=np.copy(array)
    U, V=fit_vector_field(pos_old[:,0],pos_old[:,1],pos_new[:,0]-pos_old[:,0],pos_new[:,1]-pos_old[:,1],Xi,Yi)
    for i_slices in range(array.shape[2]):
        for i_modes in range(0,array.shape[3]):
            print(1)
            warped_image=warp_image(array[:,:,i_slices,i_modes], (U, V), order=order)
            out[:,:,i_slices, i_modes]=warped_image
    return out