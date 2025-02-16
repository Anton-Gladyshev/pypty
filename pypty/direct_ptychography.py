import numpy as np
import sys
import os
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
try:
    import cupy as cp
    cpu_mode=False
except:
    import numpy as cp
    cpu_mode=True
from pypty.fft import *
from pypty.utils import *


def wdd(pypty_params, eps_wiener=1e-3, thresh=None, save=0):
    """
    This function performs Wigner distribution deconvolution.
    Inputs:
        pypty_params - dictionary with callibrated parametes
        eps_wiener- float, default 1e-3: epsilon parameter for wiener filter
        thresh- float, default None. Controlls an alternative way for deconvolution. If it is provided, then eps_wiener is ignored and denominator values below this threshold are set to 1 while the corresponding norminator values are set to 0.
        save- default False, ignored if you provided save_preprocessing_files in pypty_params
    Outputs:
        o - 2d complex Object
        probe - 2d complex beam
    """
    global cpu_mode
    if not(cpu_mode):
        cp.fft.config.clear_plan_cache()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    data_path = pypty_params.get('data_path', "")
    data_is_numpy_and_flip_ky=pypty_params.get("data_is_numpy_and_flip_ky", False)
    scan_size=np.copy(np.array(pypty_params.get('scan_size', [0,0])))
    if data_path[-3:]==".h5":
        data=h5py.File(data_path, "r")["data"]
        data=np.asarray(data)
    elif data_path[-4:]==".npy":
        data=np.load(data_path)
        if len(data.shape)==4:
            scan_size=[data.shape[0], data.shape[1]]
            data=data.reshape(data.shape[0]* data.shape[1], data.shape[2],data.shape[3])
        if data_is_numpy_and_flip_ky:
            data=data[:,::-1, :]
    data_pad = pypty_params.get('data_pad', 0)
    acc_voltage=pypty_params.get('acc_voltage', 60)
    probe=pypty_params.get('probe', None)
    aberrations = pypty_params.get('aberrations', [0])
    pixel_size_x_A = pypty_params.get('pixel_size_x_A', 1)
    pixel_size_y_A = pypty_params.get('pixel_size_y_A', 1)
    save=pypty_params.get('save_preprocessing_files', save)
    mean_pattern=pypty_params.get('mean_pattern', None)

    
    try:
        os.makedirs(pypty_params["output_folder"], exist_ok=True)
        os.makedirs(pypty_params["output_folder"]+"wdd/", exist_ok=True)
    except:
        sys.stdout.write("output folder was not created!")
    pixel_size_x_A*=(data.shape[2]+2*data_pad)/data.shape[2]
    pixel_size_y_A*=(data.shape[1]+2*data_pad)/data.shape[1]
    scan_step_A= pypty_params.get('scan_step_A', 1)
    PLRotation_deg=pypty_params.get('PLRotation_deg', 0)
    rez_pixel_size_A=pypty_params.get('rez_pixel_size_A', 1)
    rot_scan_grid_rad=-np.pi*PLRotation_deg/180 ## real-space coordinate rotation!
    data=cp.asarray(np.asarray(data).reshape(scan_size[0], scan_size[1], data.shape[1], data.shape[2]).astype(np.complex64))
    window =  pypty_params.get('window', None)
    extra_probe_defocus=pypty_params.get('extra_probe_defocus', 0)
    wavelength=12.4 /((2*511.0+acc_voltage)*acc_voltage)**0.5
    if mean_pattern is None:
        mean_pattern=np.mean(data[:100,:100], axis=(0,1))
    mean_pattern=cp.asarray(mean_pattern)
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
    try:
        o=o.get()
        p=p.get()
    except:
        pass
    if save:
        np.save(pypty_params["output_folder"]+"wdd/object.npy", o)
        np.save(pypty_params["output_folder"]+"wdd/probe.npy", probe)
    return o, probe

