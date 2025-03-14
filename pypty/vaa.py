import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from matplotlib.patches import Rectangle
import matplotlib.patheffects as PathEffects
try:
    from skimage.restoration import unwrap_phase
except:
    pass
from pypty.utils import *
from tqdm import tqdm
import matplotlib
import csv


import h5py
import pickle
import datetime


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



def fit_aberrations_to_wave(wave, px_size_A, acc_voltage, thresh=0,
                            aberrations_guess=[0,0,0,0,0,0,0,0,0,0,0,0],
                            plot=True, ftol=1e-20, xtol=1e-20, loss="linear", max_mrad=np.inf):
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
    
    
def get_affine_tranform(positions,  scan_size, px_size_A):
    x_perfect, y_perfect=np.meshgrid(np.arange(scan_size[1]), np.arange(scan_size[0]))
    x_perfect, y_perfect, off_perfect= x_perfect.flatten(), y_perfect.flatten(), np.ones(scan_size[1]*scan_size[0])
    yxo_perf=np.swapaxes(np.array([y_perfect, x_perfect, off_perfect]), 0,1)
    matrix=positions.T @ yxo_perf @ np.linalg.inv(yxo_perf.T @ yxo_perf)
    matrix=np.array(matrix)
    matrix*=px_size_A
    deformation=matrix[:, :2]
    print("Deformation matrix yy: %.2f, yx: %.2f , xy: %.2f, xx: %.2f"%(matrix[0,0], matrix[0,1],matrix[1,0], matrix[1,1]))
    print("Shift y: %.2f A, x: %.2f A"%(matrix[0,2], matrix[1, 2]))
    return deformation

        
def add_scalebar_ax(ax, x,y, width, height, x_t, y_t, px_size, unit):
    rect=Rectangle([x,y], width/px_size, height, color="white", alpha=0.9)
    text2=ax.text(x_t, y_t, str(width)+" "+unit, c="w", fontsize=20)
    text2.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    ax.add_patch(rect)
    rect.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])



def outputlog_plots(loss_path, skip_first=0, plot_time=True):
    """
    Functon for plotting log file of PyPty.
    
    Inputs:
            loss_path- pass to PyPty-csv file
        skip_first- how many first iterations to skip (default 0)
        plot_time- boolean Flag. If True, second x-axis showing time in seconds will be added on top of the plot.
    Returns:
        figs- list of plotted figures.
    """
    dat=[]
    with open(loss_path, 'r') as file:
        data = csv.reader(file, delimiter = ',')
        for d in data:
            dat.append(d)
    dat=(np.array(dat)[1+skip_first:, :-1]).astype(float)
    
    if dat.shape[1]==12:
        fieldnames=["epoch", "time / s", "loss", "sse", "initial step", "matching step", "N linesearch iterations",
                "dir. derivative", "new dir. derivative", "Constraints contribution", "Free GiB", "Total GiB", "Warnings"]
    else:
        fieldnames=["epoch", "time / s", "loss", "sse", "initial step", "matching step", "N linesearch iterations",
                "dir. derivative", "new dir. derivative", "F-axis postions reg.", "Deformation positons reg.", "Deformation tilts reg.", "F-axis tilts reg.", "l1 object reg.", "Q-space probe reg.", "R-space probe reg.", "TV object reg.", "V-object reg.", "S-axis postions reg", "S-axis tilts reg", "Free GiB", "Total GiB", "Warnings"]
    epoch=dat[:,0]
    time=dat [:,1] / 3600
    def forward(x):
        return np.interp(x,epoch,time)
    def inverse(x):
        return np.interp(x,time,epoch)
    figs=[]
    for datai in range(2, dat.shape[1], 1):
        fig,ax = plt.subplots(figsize=(10,4), dpi=300)
        ax.plot(epoch, dat[:, datai], ".-",linewidth=2, alpha=0.7)#, label=tit[iii])
        ax.set_xlabel("Iteration", fontsize = 14)
        ax.set_ylabel(fieldnames[datai], fontsize = 14)
        if plot_time:
            ax2 = ax.secondary_xaxis("top", functions=(forward,inverse)) # Create a dummy plot
            ax2.set_xlabel("time / h", fontsize = 14)
            plt.setp(ax2.get_xticklabels()[0], visible=False)
        figs.append(fig)
        plt.show()
    return figs


def radial_average(ff, r_bins, r_max, r_min, px_size_A, plot=True):
    x_grid,y_grid=np.fft.fftshift(np.fft.fftfreq(ff.shape[1])), np.fft.fftshift(np.fft.fftfreq(ff.shape[0]))
    x_grid,y_grid=np.meshgrid(x_grid, y_grid, indexing="xy")
    mult=np.min([ff.shape[0], ff.shape[1]])
    r_max*=mult
    r_min*=mult
    r=mult*(x_grid**2+y_grid**2)**0.5
    unique_distances=np.arange(0, np.max(r), r_bins)
    unique_distances=unique_distances[((unique_distances<=r_max)*(unique_distances>=r_min)).astype(bool)]
    radial_avg=np.zeros_like(unique_distances)
    if plot:
        for iii in tqdm(range(len(unique_distances))):
            distance=unique_distances[iii]
            radial_mask = (r<=distance+r_bins)*(r>distance)
            radial_avg[iii] = np.mean(ff[radial_mask])
        fig=plt.figure(figsize=(10,3))
        plt.plot(unique_distances/(mult*px_size_A), radial_avg, "-")
        plt.xlabel("spatial freqency [A$^{-1}$]")
        plt.yscale("log")
        plt.show()
    return fig


def complex_pca(data, n_components):
    N_y, N_x, N_obs = data.shape
    reshaped_data = data.reshape(-1, N_obs)
    mean = np.mean(reshaped_data, axis=0)
    centered_data = reshaped_data - mean
    covariance_matrix = np.cov(centered_data, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    principal_components = eigenvectors[:, :n_components]
    reduced_data = centered_data @ principal_components
    data_reduced = reduced_data.reshape(N_y, N_x, n_components)
    return data_reduced


def complex_array_to_rgb(X, theme='dark', rmax=None):
    '''Takes an array of complex number and converts it to an array of [r, g, b],
    where phase gives hue and saturaton/value are given by the absolute value.
    Especially for use with imshow for complex plots.'''
    absmax = rmax or np.abs(X).max()
    Y = np.zeros(X.shape + (3,), dtype='float')
    Y[..., 0] = np.angle(X) / (2 * np.pi) % 1
    if theme == 'light':
        Y[..., 1] = np.clip(np.abs(X) / absmax, 0, 1)
    elif theme == 'dark':
        Y[..., 1] = 1
        Y[..., 2] = np.clip(np.abs(X) / absmax, 0, 1)
    Y = matplotlib.colors.hsv_to_rgb(Y)
    return Y


def plot_complex_modes(p, nm, sub):
    sub=1
    p2=np.abs(p)**2
    pint=np.sum(np.abs(p)**2, (0,1))
    pint=100*pint/np.sum(pint)
    sort=np.argsort(pint)[::-1]
    p=p[:,:,sort]
    pint=pint[sort]
    fig, axes=plt.subplots(sub,nm//sub, figsize=(4*nm//sub,4*sub))
    try:
        axes=axes.flatten()
    except:
        axes=[axes]
    for i, ax in enumerate(axes):
        ax.imshow(complex_array_to_rgb(p[::-1,::-1,i], theme='dark', rmax=np.max(np.abs(p))))
        ax.axis("off")
        ax.text(15,0.9*p.shape[0], "%.1e %%"%(pint[i]), fontsize=15)
    plt.show()
    return fig

