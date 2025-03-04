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
    
from pypty.fft import *


 
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
    
    



def save_updated_arrays(output_folder, epoch,current_probe_step, current_probe_pos_step, current_tilts_step,current_obj_step, obj, probe, tilts_correction, full_pos_correction, positions, tilts, static_background, current_aberrations_array_step, current_static_background_step,count, current_loss, current_sse, aberrations, beam_current, current_beam_current_step, save_flag, save_loss_log, constraint_contributions, actual_step, count_linesearch, d_value, new_d_value,current_update_step_bfgs, t0, xp, warnings):
    if save_loss_log:
        with open(output_folder+"loss.csv", mode='a', newline='') as loss_list:
            if save_loss_log==2:
                fieldnames=["epoch", "time / s", "loss", "sse", "initial step", "matching step", "N linesearch iterations",
                "dir. derivative", "new dir. derivative", "F-axis postions reg.", "Deformation positons reg.", "Deformation tilts reg.", "F-axis tilts reg.", "l1 object reg.", "Q-space probe reg.", "R-space probe reg.", "TV object reg.", "V-object reg.", "Free GiB", "Total GiB", "Warnings"]
            else:
                fieldnames=["epoch", "time / s", "loss", "sse", "initial step", "matching step", "N linesearch iterations",
                "dir. derivative", "new dir. derivative", "Constraints contribution", "Free GiB", "Total GiB", "Warnings"]
            if xp!=np:
                device = cp.cuda.Device(0)
                total_mem_device=  device.mem_info[1] / (1024 **3)
                free_mem_device=   device.mem_info[0] / (1024 **3)
            else:
                total_allocated,total_reserved, total_mem_device=0,0,0
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
                                "l1 object reg.": constraint_contributions[4],
                                "Q-space probe reg.": constraint_contributions[5],
                                "R-space probe reg.": constraint_contributions[6],
                                "TV object reg.": constraint_contributions[7],
                                "V-object reg.": constraint_contributions[8],
                                "Free GiB":  free_mem_device,
                                "Total GiB": total_mem_device,
                                "Warnings": warnings,
                                })
            else:
                for dumbi1 in range(9):
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
    
def create_probe_from_nothing(probe, data_pad, mean_pattern, aperture_mask, tilt_mode, tilts, dataset, estimate_aperture_based_on_binary, pixel_size_x_A, acc_voltage, data_multiplier, masks, data_shift_vector, data_bin, upsample_pattern, default_complex_cpu, print_flag, algorithm, measured_data_shape, n_obj_modes, probe_marker, recon_type, defocus_array, Cs):
    if type(probe)!=np.ndarray:
        if probe is None or probe=="aperture":
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
        probe_counts_must = np.sum(dataset[:1000])/((dataset[:1000]).shape[0] * probe.shape[0] * probe.shape[1])
    else:
        probe_counts_must = np.sum(dataset[:1000])/((dataset[:1000]).shape[0])
    print(probe.shape)
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
    
def prepare_saving_stuff(output_folder, save_loss_log, epoch_prev):
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
            fieldnames=["epoch", "time / s", "loss", "sse", "initial step", "matching step", "N linesearch iterations",
                "dir. derivative", "new dir. derivative", "F-axis postions reg.", "Deformation positons reg.", "Deformation tilts reg.", "F-axis tilts reg.", "l1 object reg.", "Q-space probe reg.", "R-space probe reg.", "TV object reg.", "V-object reg.", "Free GiB", "Total GiB", "Warnings"]
        else:
            fieldnames=["epoch", "time / s", "loss", "sse", "initial step", "matching step", "N linesearch iterations",
                "dir. derivative", "new dir. derivative", "Constraints contribution", "Free GiB", "Total GiB", "Warnings"]
        with open(output_folder+"loss.csv", 'w+', newline='') as loss_list:
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
            sys.stdout.write("\r---------> Time: %d:%2d:%2d. Epoch %i. Using %s error metric with %s optimzer. Loss: %.2e. SSE: %.2e. %s" % (hours, minutes, seconds,  epoch, algorithm, print_optimizer, current_loss, current_sse, string))
        else:
            sys.stdout.write("\n---------> Time: %d:%2d:%2d. Epoch %i. Using %s error metric with %s optimzer. Loss: %.2e. SSE: %.2e. %s" % (hours, minutes, seconds,  epoch, algorithm, print_optimizer, current_loss, current_sse, string))
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



def get_cupy_memory_usage():
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


def get_compute_batch(compute_batch, load_one_by_one, hist_size, measured_data_shape, memory_satiration, smart_memory, data_pad, obj_shape, probe_shape,  dtype, propmethod, print_flag):
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
    suggested_compute_batch=int(np.floor((total_mem_device_Gb*memory_satiration -update_memory - load_one_by_one_memory)/per_compute_batch_memory))
    if suggested_compute_batch<=5 and not(load_one_by_one):
        suggested_compute_batch=int(np.floor((total_mem_device_Gb*memory_satiration -update_memory)/per_compute_batch_memory))
        if suggested_compute_batch==0:
            suggested_compute_batch=1
        if print_flag:
            sys.stdout.write("\nWe do not suggest to keep the dataset in the memory and manually set load_one_by_one to True. Optimal compute batch is %d. Predicted memory usage: %.2f GiB"%(suggested_compute_batch, total_mem_device_Gb*memory_satiration))
    elif print_flag:
        sys.stdout.write("\nWe suggest to use compute batch of %d. Predicted memory usage: %.2f GiB"%(suggested_compute_batch, total_mem_device_Gb*memory_satiration))
    sys.stdout.flush()
    try:
        test=smart_memory(0)
    except:
        if total_mem_device_Gb<20:
            smart_memory=True
    return suggested_compute_batch, load_one_by_one, smart_memory




def update_weights_constraints(fast_axis_reg_weight_positions, deformation_reg_weight_positions, deformation_reg_weight_tilts, fast_axis_reg_weight_tilts, phase_norm_weight, abs_norm_weight, probe_reg_constraint_weight, window_weight, atv_weight, mixed_variance_weight,     updated_fast_axis_reg_weight_positions, updated_deformation_reg_weight_positions, updated_deformation_reg_weight_tilts, updated_fast_axis_reg_weight_tilts, updated_phase_norm_weight, updated_abs_norm_weight, updated_probe_reg_weight, updated_window_weight, updated_atv_weight, updated_mixed_variance_weight):
    if not(updated_fast_axis_reg_weight_positions is None):
        fast_axis_reg_weight_positions=updated_fast_axis_reg_weight_positions
        sys.stdout.write("\nUpdating fast_axis_reg_weight_positions to %.3e"%updated_fast_axis_reg_weight_positions)
    if not(updated_deformation_reg_weight_positions  is None):
        deformation_reg_weight_positions=updated_deformation_reg_weight_positions
        sys.stdout.write("\nUpdating deformation_reg_weight_positions to %.3e"%updated_deformation_reg_weight_positions)
    if not(updated_deformation_reg_weight_tilts  is None):
        deformation_reg_weight_tilts=updated_deformation_reg_weight_tilts
        sys.stdout.write("\nUpdating deformation_reg_weight_tilts to %.3e"%updated_deformation_reg_weight_tilts)
    if not(updated_fast_axis_reg_weight_tilts  is None):
        fast_axis_reg_weight_tilts=updated_fast_axis_reg_weight_tilts
        sys.stdout.write("\nUpdating fast_axis_reg_weight_tilts to %.3e"%updated_fast_axis_reg_weight_tilts)
    if not(updated_phase_norm_weight is None):
        phase_norm_weight=updated_phase_norm_weight
        sys.stdout.write("\nUpdating phase_norm_weight to %.3e"%updated_phase_norm_weight)
    if not(updated_abs_norm_weight  is None):
        abs_norm_weight=updated_abs_norm_weight
        sys.stdout.write("\nUpdating abs_norm_weight to %.3e"%updated_abs_norm_weight)
    if not(updated_probe_reg_weight  is None):
        probe_reg_constraint_weight=updated_probe_reg_weight
        sys.stdout.write("\nUpdating probe_reg_constraint_weight to %.3e"%updated_probe_reg_weight)
    if not(updated_window_weight is None):
        window_weight=updated_window_weight
        sys.stdout.write("\nUpdating window_weight to %.3e"%updated_window_weight)
    if not(updated_atv_weight  is None):
        atv_weight=updated_atv_weight
        sys.stdout.write("\nUpdating atv_weight to %.3e"%updated_atv_weight)
    if not(updated_mixed_variance_weight  is None):
        mixed_variance_weight=updated_mixed_variance_weight
        sys.stdout.write("\nUpdating mixed_variance_weight to %.3e"%updated_mixed_variance_weight)
    return fast_axis_reg_weight_positions, deformation_reg_weight_positions, deformation_reg_weight_tilts, fast_axis_reg_weight_tilts, phase_norm_weight, abs_norm_weight, probe_reg_constraint_weight, window_weight, atv_weight, mixed_variance_weight

