import numpy as np
import sys
import os
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import percentile_filter as cpu_percentile
from scipy.interpolate import griddata

def getvirtualhaadf(pypty_params, save=True):
    """
    Compute a virtual HAADF image from a 4D-STEM dataset.

    Parameters
    ----------
    pypty_params : dict
        Dictionary containing parameters including data path, scan size, plotting option, and output folder.
    save : bool, optional
        Whether to save the resulting HAADF image as a .npy file. Default is True.

    Returns
    -------
    haadf : ndarray
        The computed virtual HAADF image.
    """
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
    """
    Generate a binary aperture mask based on the mean diffraction pattern.

    Parameters
    ----------
    params : dict
        Dictionary containing parameters including data path, data padding, plotting option, and bright threshold.

    Returns
    -------
    params : dict
        Updated parameters dictionary containing the generated aperture mask.
    """
    data_path=params.get("data_path", "")
    data_pad=params.get("data_pad", 0)
    plot=params.get("plot", False)
    threshold=params.get("bright_threshold", 0.4)
    flip_ky=params.get("flip_ky", False)
    if data_path[-2:]=="h5":
        this_file=h5py.File(data_path, "r")
        dataset=this_file['data']
    else:
        dataset=np.load(data_path, mmap_mode="r")
        if flip_ky:
            if len(dataset.shape)==4:
                dataset=dataset[:,:,::-1,:]
            else:
                dataset=dataset[:,::-1,:]
    if len(dataset.shape)==3:
        meanpat=np.mean(dataset, 0)
    if len(dataset.shape)==4:
        meanpat=np.mean(dataset, (0,1))
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

def create_binned_dataset(path_orig, path_new, bin):
    """
    Downsample a dataset by spatial binning and save it to a new file.

    Parameters
    ----------
    path_orig : str
        The file path of the original dataset.
    path_new : str
        The file path to save the binned dataset.
    bin : int
        The binning factor to downsample the dataset.
    """
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
    """
    Compensate for drift in diffraction patterns via phase correlation.

    Parameters
    ----------
    aperture : ndarray
        The binary aperture mask used for phase correlation.
    patterns : ndarray
        The diffraction patterns to be compensated for drift.

    Returns
    -------
    patterns : ndarray
        The compensated diffraction patterns.
    """
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

def get_virtual_annular_detector(pypty_params, inner_rad=0, outer_rad=1, save=False, offset_x=0, offset_y=0):
    """
    Create virtual detector signals from annular masks.

    Parameters
    ----------
    pypty_params : dict
        Dictionary containing parameters including data path, scan size, plotting option, and output folder.
    inner_rad : float, optional
        Inner radius of the annular mask. Default is 0.
    outer_rad : float, optional
        Outer radius of the annular mask. Default is 1.
    save : bool, optional
        Whether to save the resulting virtual detector signal as a .npy file. Default is False.
    offset_x : float, optional
        X-offset for the annular mask. Default is 0.
    offset_y : float, optional
        Y-offset for the annular mask. Default is 0.

    Returns
    -------
    signal : ndarray
        The computed virtual detector signal.
    """
    save=pypty_params.get("save_preprocessing_files", save)
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
    x,y=np.linspace(-1,1, dataset_h5.shape[2]), np.linspace(-1,1, dataset_h5.shape[1])
    x,y=np.meshgrid(x,y, indexing="xy")
    r=((x-offset_x)**2+(y-offset_y)**2)**0.5
    r=(r>=inner_rad)* (r<=outer_rad)
    signal=np.sum(dataset_h5*r[None,:,:], (-1,-2)).reshape(scan_size)
    if plot:
        fig,axes=plt.subplots(1,2, figsize=(10,5))
        ax=axes[0]
        im=ax.imshow(signal, cmap="gray")
        ax.set_title("Virtual Detector signal")
        ax.axis("off")
        ax=axes[1]
        im=ax.imshow(r, cmap="gray")
        ax.set_title("Virtual Detector")
        ax.axis("off")
        plt.show()
    if save:
        np.save(pypty_params["output_folder"]+"/virtual_signal_i_%.2f_o_%.2f.npy"%(inner_rad, outer_rad), signal)
    return signal




def unwarp_im(warp_im, ab, method="linear", plot_flag=False, fig_num=900, test=False):
    """
    Originally written by Wouter Van den Broek.
    This takes in a warped image, and a transformation matrix (ab), and outputs a tuple of two images. The first image has the illumination corrected, the second only does the geometric distortions, not the illumination correction
    """
    if test is True:
        warp_im = np.zeros_like(warp_im)
        warp_im[warp_im.shape[0] // 2 :, : warp_im.shape[1] // 2] = 1
        warp_im[: warp_im.shape[0] // 2, warp_im.shape[1] // 2 :] = 2
        warp_im[warp_im.shape[0] // 2 :, warp_im.shape[1] // 2 :] = 3

    # Read the warped image
    g = 1* warp_im
    gshape=g.shape
    offset = 2
    assert np.all(g >= 0), "Values are below 0"
    g = np.log(g + offset)

    # Start the unwarping
    # coordinates of the warped image in uv-space
    u_g = np.arange(g.shape[0])
    v_g = np.arange(g.shape[1])
    uv_g = (u_g, v_g)

    # The mean scaling, probed over many different directions
    sc = np.sqrt(coordinate_transformation_2d_areamag(np.zeros((1, 2)), ab))

    # Coordinates in xy-space
    x_tmp = np.linspace(-0.5, 0.5, g.shape[0]) * (g.shape[0] - 1) / sc
    y_tmp = np.linspace(-0.5, 0.5, g.shape[1]) * (g.shape[1] - 1) / sc
    [x_tmp, y_tmp] = np.meshgrid(x_tmp, y_tmp, indexing="ij")
    x_tmp = np.ravel(x_tmp)
    y_tmp = np.ravel(y_tmp)

    # Transform those to uv-space
    uv_i = coordinate_transformation_2d((x_tmp, y_tmp), ab)

    # Do the unwarping
    g = spip.interpn(
        uv_g, g, uv_i, method=method, bounds_error=False, fill_value=np.log(offset)
    )
    g = np.reshape(g, (gshape[0], gshape[1]))

    # undo the logarithms
    g = np.exp(g) - offset

    area_mag = coordinate_transformation_2d_areamag((x_tmp, y_tmp), ab)
    area_mag = np.reshape(area_mag, (gshape[0], gshape[1]))
    tmp = area_mag[int(round(g.shape[0] / 2)), int(round(g.shape[1] / 2))]
    area_mag = area_mag / tmp  # Area magnification in the middle is 1 now
    
    g *= area_mag
    return g


def coordinate_transformation_2d_areamag(xy, ab):
    """
    Define the local area magnification of the mapping function

    Originally written by Wouter Van den Broek
    """
    # Transforms from xy to uv
    xy = np.asarray(xy)
    tmp = xy.shape
    if tmp[0] == 2:
        xy = np.transpose(xy)
    tmp = np.asarray(ab.shape)
    if tmp[0] == 3:
        trafo_flag = 0  # linear
    if tmp[0] == 6:
        trafo_flag = 1  # quadratic
    if tmp[0] == 10:
        trafo_flag = 2  # cubic

    x = np.ravel(xy[:, 0])
    y = np.ravel(xy[:, 1])

    a = ab[:, 0]
    b = ab[:, 1]

    # Derivative of u wrt x
    duv_dxy = a[1]
    if trafo_flag > 0:
        duv_dxy += a[3] * y + 2 * a[4] * x
    if trafo_flag > 1:
        duv_dxy += 2 * a[6] * x * y + a[7] * y ** 2 + 3 * a[8] * x ** 2
    mag_tmp = duv_dxy

    # Derivative of v wrt y
    duv_dxy = b[2]
    if trafo_flag > 0:
        duv_dxy += b[3] * x + 2 * b[5] * y
    if trafo_flag > 1:
        duv_dxy += b[6] * x ** 2 + 2 * b[7] * x * y + 3 * b[9] * y ** 2
    mag_tmp *= duv_dxy

    # Derivative of u wrt y
    duv_dxy = a[2]
    if trafo_flag > 0:
        duv_dxy += a[3] * x + 2 * a[5] * y
    if trafo_flag > 1:
        duv_dxy += a[6] * x ** 2 + 2 * a[7] * x * y + 3 * a[9] * y ** 2
    mag = duv_dxy

    # Derivative of v wrt x
    duv_dxy = b[1]
    if trafo_flag > 0:
        duv_dxy += b[3] * y + 2 * b[4] * x
    if trafo_flag > 1:
        duv_dxy += 2 * b[6] * x * y + b[7] * y ** 2 + 3 * b[8] * x ** 2
    mag *= duv_dxy

    mag_tmp -= mag
    mag = np.absolute(mag_tmp)
    mag = np.asarray(mag)

    return mag



