import numpy as np
import sys
import os
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import percentile_filter as cpu_percentile





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
