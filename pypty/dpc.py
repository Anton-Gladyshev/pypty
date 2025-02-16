import numpy as np
import sys
import os
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import minimize


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

def fft_based_dpc(pypty_params, hpass=0, lpass=0, save=True, comx=None, comy=None, plot=True):
    save=pypty_params.get("save_preprocessing_files", save)
    if save:
        try:
            os.makedirs(pypty_params["output_folder"], exist_ok=True)
            os.makedirs(pypty_params["output_folder"]+"dpc/", exist_ok=True)
        except:
            sys.stdout.write("output folder was not created!")
    dataset_h5=pypty_params.get("data_path", "")
    scan_size=pypty_params.get("scan_size", None)
    angle=pypty_params.get("PLRotation_deg", None)
    plot=pypty_params.get("plot", plot)
    data_is_numpy_and_flip_ky=pypty_params.get("data_is_numpy_and_flip_ky", False)
    if dataset_h5[-3:]==".h5":
        dataset_h5=h5py.File(dataset_h5, "r")
        dataset_h5=dataset_h5["data"]
    elif dataset_h5[-4:]==".npy":
        dataset_h5=np.load(dataset_h5)
        if len(dataset_h5.shape)==4:
            scan_size=[dataset_h5.shape[0], dataset_h5.shape[1]]
            dataset_h5=dataset_h5.reshape(dataset_h5.shape[0]* dataset_h5.shape[1], dataset_h5.shape[2],dataset_h5.shape[3])
        if data_is_numpy_and_flip_ky:
            dataset_h5=dataset_h5[:,::-1, :]
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
        ax.set_title("DPC potential")
        ax.axis("off")
        ax.axis("off")
        plt.show()
    pypty_params["comx"]=comx
    pypty_params["comy"]=comy
    if save:
        np.save(pypty_params["output_folder"]+"dpc/idpc.npy", pot)
    pypty_params["PLRotation_deg"]=angle*180/np.pi
    return pot, pypty_params




def iterative_dpc(pypty_params, num_iterations=100, beta=0.5, hpass=0, lpass=0, step_size=0.1,
                COMx=None, COMy=None, px_size=None,print_flag=False,save=True,
                select=None, plot=True, bin_fac=1, use_backtracking=True, pad_width=5):
    save=pypty_params.get("save_preprocessing_files", save)
    if save:
        try:
            os.makedirs(pypty_params["output_folder"], exist_ok=True)
            os.makedirs(pypty_params["output_folder"]+"dpc/", exist_ok=True)
        except:
            sys.stdout.write("output folder was not created!")
    dataset_h5=pypty_params.get("data_path", "")
    scan_size=pypty_params.get("scan_size", None)
    angle=pypty_params.get("PLRotation_deg", 0)*np.pi/180
    rez_pixel_size_A=pypty_params.get("rez_pixel_size_A", 1)
    scan_size=pypty_params.get("scan_size", None)
    plot=pypty_params.get("plot", plot)
    print_flag=pypty_params.get("print_flag", print_flag)
    px_size=pypty_params.get("scan_step_A", px_size)
    data_is_numpy_and_flip_ky=pypty_params.get("data_is_numpy_and_flip_ky", False)
    if dataset_h5[-3:]==".h5":
        dataset_h5=h5py.File(dataset_h5, "r")
        dataset_h5=dataset_h5["data"]
    elif dataset_h5[-4:]==".npy":
        dataset_h5=np.load(dataset_h5)
        if len(dataset_h5.shape)==4:
            scan_size=[dataset_h5.shape[0], dataset_h5.shape[1]]
            dataset_h5=dataset_h5.reshape(dataset_h5.shape[0]* dataset_h5.shape[1], dataset_h5.shape[2],dataset_h5.shape[3])
        if data_is_numpy_and_flip_ky:
            dataset_h5=dataset_h5[:,::-1, :]
    if (COMx is None) or (COMy is None):
        COMx=pypty_params.get('comx', None)
        COMy=pypty_params.get('comy', None)
    if (COMx is None) or (COMy is None):
        x,y=np.arange(dataset_h5.shape[-1])*rez_pixel_size_A, np.arange(dataset_h5.shape[-2])*rez_pixel_size_A
        x,y=np.meshgrid(x-np.mean(x), y-np.mean(y), indexing="xy")
        ssum=np.empty(scan_size)
        if (comx is None) or (comy is None):
            COMx=np.empty(scan_size)
            COMy=np.empty(scan_size)
            for index_data_y in range(scan_size[0]):
                for index_data_x in range(scan_size[1]):
                    dataindex=index_data_x+index_data_y*scan_size[1]
                    ssum[index_data_y, index_data_x]=np.sum(dataset_h5[dataindex])
                    COMx[index_data_y, index_data_x]=np.sum(dataset_h5[dataindex]*x)
                    COMy[index_data_y, index_data_x]=np.sum(dataset_h5[dataindex]*y)
            COMx=comx/ssum.astype(np.float32)
            COMy=comy/ssum.astype(np.float32)
    rcomx = COMx * np.cos(angle) + COMy * np.sin(angle)
    rcomy =-COMx * np.sin(angle) + COMy * np.cos(angle)
    COMx=rcomx-np.mean(rcomx)
    COMy=rcomy-np.mean(rcomy)
    if select is None:
        Ny, Nx = COMx.shape
    else:
        Ny, Nx = select.shape
    padded_phase = np.random.rand(Ny+pad_width, Nx+pad_width)*1e-3
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
    padded_phase=padded_phase[:Ny, :Nx]
    if save:
        np.save(pypty_params["output_folder"]+"dpc/iterative_dpc.npy", padded_phase)
    return padded_phase



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
