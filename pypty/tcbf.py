import numpy as np
import sys
import os
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import rotate, binary_closing
from tqdm import tqdm
try:
    import cupy as cp
    cpu_mode=False
except:
    import numpy as cp
    cpu_mode=True
from pypty.fft import *
from pypty.utils import *

def run_tcbf_alignment(params, binning_for_fit=[8],
                        save=True, optimize_angle=True,  aberrations=None, n_aberrations_to_fit=12,
                        reference_type="bf",
                        refine_box_dim=10, upsample=3,
                        cross_corr_type="phase", cancel_large_shifts=None,
                        
                        scan_pad=None, aperture=None, subscan_region=None,
    
                        compensate_lowfreq_drift=False, append_lowfreq_shifts_to_params=True,
                        interpolate_scan_factor=1,
                        binning_cross_corr=1, phase_cross_corr_formula=False,
                        f_scale_lsq=1,x_scale_lsq=1, loss_lsq="linear", tol_ctf=1e-8):
    """
    This function fits the beam CTF to the shifts between the individual pixel images of the 4d-stem dataset. The shift estimation is done via cross-correaltion.
    inputs:
        pypty_params - dictionary with experimental parameters and reconsturciton settings. For more please see functions append_exp_params() and run_ptychography()
        binning_for_fit- list of binning values at which the fit will be performed. To do 4 iterations at binning value of 8 it should be [8,8,8,8]

        save - boolean flag. If true, the intermidate tcBF images on the intial scan grid will be saved as .png. Ignored if you provided save_preprocessing_files in pypty_params.
        optimize_angle - boolean flag. Allows to fit the PL rotation angle with aberrations
        aberrations- list with an initial guess for aberrations. Ignored if provided in pypty_params.
        n_aberrations_to_fit - integer. If the inital guess for aberrations is not provided, the code with try to initialize them as a list of zeros with this length. Ignored if aberrations are provided in pypty_params.

        reference_type - by default is set to "bf". In this case all pixel images will be correlated to the tcbf estimate. The other option is "zero". In this case the images will be correlated to the image of the pixel closest to the optical axis
        refine_box_dim - radius of a box in which the shifts between the images will be refined. The cross-correlation can estimate the shift only on the initial grid. To get a more precise value, we have to interpolate. This is done here via a cubic spline.
        upsample - the upsampling of the cross-correlation for precise maximum estimation. (Note the image itself will not be upsampled. For this feature please reffer to upsampled_tcbf() function)

        cross_corr_type - type of cross correlation. Default "phase" for phase cross correlation that should perform better with noisy data. Anything but "phase" will result in a classical Foruier-correlation.
        cancel_large_shifts - None or float strictly between 0 and 1. If not None, the abnoramally large shifts will be ignored in the CTF fit.
        scan_pad - amount of scan positions to add to the edges in order to prevent wrap around artifacts. If None the code will figure it out automatically

        aperture - boolean mask for aperture. If None the code will try to get it from the pypty_params. Note that the function append_exp_params generates the aperture automatically
        subscan_region,  None or list of subscan boundaries (left, top, right, bottom) on which one should do the fit.
        compensate_lowfreq_drift - boolean flag. If True the code will try to cancel the drift of the patterns for larges fields of view.
        append_lowfreq_shifts_to_params - boolean flag. If true, the lowfreq drift correction will be stored in pypty_params. This should accelerate the later preparation of the data.

        interpolate_scan_factor, integer default 1. If larger than 1 the scan will be upsampled via interpolation. This is an experimental feature! For creating good upsampled images plese see upsampled_tcbf() function.
        The Fit of the CTF is done via scipy least squares. You can supply it with followng parameters: 
            f_scale_lsq - default 1 (controlls f_scale)
            x_scale_lsq - default 1 (controlls x_scale)
            loss_lsq - default "linear" (controlls loss type)
            tol_ctf - default is 1e-8, (controlls tolerance "ftol" parameter)
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
    save=pypty_params.get("save_preprocessing_files", save)
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
        aberrations=pypty_params.get("aberrations", None)
        if aberrations is None:
            aberrations=list(np.zeros(n_aberrations_to_fit))
            aberrations[0]=-1*pypty_params.get("extra_probe_defocus", 0)
    plot=pypty_params.get("plot", False)
    print_flag=pypty_params.get("print_flag", False)
    data_is_numpy_and_flip_ky=pypty_params.get("data_is_numpy_and_flip_ky", False)
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
        if data_is_numpy_and_flip_ky:
            dataset_h5=dataset_h5[:,::-1, :]
   
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
    if not(cpu_mode or smart_memory):
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
        if plot or save: ## plot the tcBF estimate
            image_bf_binned_plot=cp.real(ifft2(image_bf_binned_fourier))
            if not(cpu_mode):
                image_bf_binned_plot=image_bf_binned_plot.get()
            plt.imshow(image_bf_binned_plot, cmap="gray")
            plt.title("tcBF image at bin %d. Iteration %d"%(bin_fac, index_bin))
            plt.axis("off")
            if save:
                plt.savefig(pypty_params["output_folder"]+"/tcbf/tcbf_image_"+str(index_bin)+".png", dpi=200)
                np.save(pypty_params["output_folder"]+"/tcbf/tcbf_image_"+str(index_bin)+".npy", image_bf_binned_plot)
            if not(plot):
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
        if save:
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
        if plot or save: ## plot the results
            ctf_grad_x, ctf_grad_y=get_ctf_derivatives(aberrations, binned_kx_detector_suc,binned_ky_detector_suc,  wavelength, angle_offset)
            fig, ax=plt.subplots(1,2, figsize=(10,5))
            ap_show=rotate(aperture_binned, angle=0, axes=(1, 0), reshape=False, order=0)
            ax[0].imshow(ap_show, cmap="gray", extent=[np.min(binned_kx_detector_full),np.max(binned_kx_detector_full), np.min(binned_ky_detector_full), np.max(binned_ky_detector_full)])
            ax[0].quiver(binned_kx_detector_suc,binned_ky_detector_suc, estimated_shifts[0,:], estimated_shifts[1,:],  color="red", capstyle="round")
            ax[1].imshow(ap_show, cmap="gray", extent=[np.min(binned_kx_detector_full), np.max(binned_kx_detector_full), np.min(binned_ky_detector_full), np.max(binned_ky_detector_full)])
            ax[1].quiver(binned_kx_detector_suc,binned_ky_detector_suc, ctf_grad_x, ctf_grad_y,  color="red", capstyle="round")
            ax[0].set_title("Fitted shifts")
            ax[1].set_title("Fitted CTF grad")
            if save:
                fig.savefig(pypty_params["output_folder"]+"/tcbf/tcbf_shifts_"+str(index_bin)+".png", dpi=200)
            if not(plot):
                fig.close()
            else:
                fig.show()
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
    
    if save:
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
        if save:
            fig.savefig(pypty_params["output_folder"]+"/tcbf/aberrations_fit.png")
        plt.show()
        if optimize_angle:
            fig, ax=plt.subplots(figsize=(10, 2))
            ax.plot((fit_angle_array)*180/np.pi, "-.", linewidth=2, label="angle offset")
            ax.set_xlabel("iteration")
            ax.set_ylabel("angle (deg)")
            if save:
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


def upsampled_tcbf(pypty_params, upsample=5, pad=10,
                    compensate_lowfreq_drift=False,
                    default_float=64, round_shifts=False,
                    xp=cp, save=0, max_parallel_fft=100, bin_fac=1):
    """
    Run a tcBF reconstruction on an upsampled grid. Note that usually before doing so you need to execute run_tcbf_alignment fucntion to adjust pypty_params.
    inputs:
        pypty_params - dictionary with experimetal paramers and other settings. For more please see run_ptychography() and append_exp_params()
        upsample - integer upsampling factor
        pad - amount of scan positions to add to the sides to eliminate wrap-around artifacts
        compensate_lowfreq_drift - boolean flag. If true, the code will try to compensate drifts of an aperture. Requieres to run_tcbf_alignemnt beforehand!!!
        bin_fac - binning for the data in reciprocal space. Default 1 (no binning).

        default_float- 64 or 32 for better memory management
        round_shifts - boolean. If true, shifts will be rounded. And alignment will be done via roll(). If False, FFT-shift will be used. The last option is more precise, but creates artifacts on the edges
        xp - backend. numpy or Cupy
        save - boolean flag. Default false. Ignored if you provided save_preprocessing_files in pypty_params
        max_parallel_fft - amount of FFTs to do in a vectorized fashion.
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
    save=pypty_params.get("save_preprocessing_files", save)
    data_is_numpy_and_flip_ky=pypty_params.get("data_is_numpy_and_flip_ky", False)
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
        if data_is_numpy_and_flip_ky:
            patterns=patterns[:,::-1, :]
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
