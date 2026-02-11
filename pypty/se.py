import numpy as np
import sys
import os
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import percentile_filter as cpu_percentile
from scipy.ndimage import map_coordinates

from scipy.interpolate import griddata
from scipy.optimize import minimize, brent, least_squares


import scipy as sp
import scipy.interpolate as spip
import scipy.ndimage as spim
from scipy.spatial import Voronoi, KDTree
from matplotlib import patches
from tqdm import tqdm
from typing import Union

from scipy.ndimage import rotate
try:
    from scipy.interpolate import RBFInterpolator
except:
    pass
from tqdm import tqdm


from pypty import iterative as pyptyit
from pypty import initialize as pyptyini
from pypty import utils as pyptyutils

def getvirtualhaadf(pypty_params, save=True):
    """
    Compute a virtual HAADF image from 4D-STEM data.

    Parameters
    ----------
    pypty_params : dict
        Dictionary with keys 'data_path', 'scan_size', 'plot', 'output_folder'.
    save : bool, optional
        Whether to save the image. Default is True.

    Returns
    -------
    haadf : numpy.ndarray
        HAADF intensities array.
    """
    
    scan_size=pypty_params.get("scan_size", None)
    plot=pypty_params.get("plot", False)
    dataset_h5=pypty_params.get("dataset", None)
    if dataset_h5 is None:
        dataset_h5=pypty_params.get("data_path", "")
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
    Generate a binary aperture mask from diffraction data.

    Parameters
    ----------
    params : dict
        Dictionary with keys 'data_path', 'data_pad', 'plot', 'bright_threshold', 'flip_ky'.

    Returns
    -------
    params : dict
        Updated dict including 'aperture_mask'.
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
    Downsample a dataset by spatial binning and save to a new file.

    Parameters
    ----------
    path_orig : str
        Path to the original dataset.
    path_new : str
        Path to save the binned dataset.
    bin : int
        Binning factor.

    Raises
    ------
    IOError
        If file operations fail.
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
    Compensate drift in diffraction patterns via phase correlation.

    Parameters
    ----------
    aperture : numpy.ndarray
        Binary aperture mask.
    patterns : numpy.ndarray
        4D diffraction patterns.

    Returns
    -------
    numpy.ndarray
        Drift-compensated patterns.
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
    Compute virtual detector signal using annular masks.

    Parameters
    ----------
    pypty_params : dict
        Dictionary with keys 'data_path', 'scan_size', 'plot', 'output_folder'.
    inner_rad : float, optional
        Inner radius in normalized units. Default is 0.
    outer_rad : float, optional
        Outer radius in normalized units. Default is 1.
    save : bool, optional
        Whether to save the signal. Default is False.
    offset_x : float, optional
        X-offset. Default is 0.
    offset_y : float, optional
        Y-offset. Default is 0.

    Returns
    -------
    numpy.ndarray
        2D detector signal array.
    """
    save=pypty_params.get("save_preprocessing_files", save)
    scan_size=pypty_params.get("scan_size", None)
    plot=pypty_params.get("plot", False)
    dataset_h5=pypty_params.get("dataset", None)
    if dataset_h5 is None:
        dataset_h5=pypty_params.get("data_path", "")
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

def coordinate_transformation_2d(xy, ab):
    """
    Map coordinates from XY space to UV space using polynomial coefficients.

    Parameters
    ----------
    xy : array_like
        Coordinates to transform. Shape (2, N) or (N, 2).
    ab : numpy.ndarray
        Transformation coefficients. Length 3 (linear),
        6 (quadratic), or 10 (cubic).

    Returns
    -------
    numpy.ndarray
        Transformed coordinates, shape (N, 2).

    Notes
    -----
    Original implementation by Wouter Van den Broek.
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
    u = a[0] + a[1] * x + a[2] * y
    b = ab[:, 1]
    v = b[0] + b[1] * x + b[2] * y
    if trafo_flag > 0:
        u += a[3] * x * y + a[4] * x ** 2 + a[5] * y ** 2
        v += b[3] * x * y + b[4] * x ** 2 + b[5] * y ** 2
    if trafo_flag > 1:
        u += a[6] * x ** 2 * y + a[7] * x * y ** 2 + a[8] * x ** 3 + a[9] * y ** 3
        v += b[6] * x ** 2 * y + b[7] * x * y ** 2 + b[8] * x ** 3 + b[9] * y ** 3

    uv = np.transpose([u, v])
    uv = np.asarray(uv)

    return uv

def find_ab(x, y, u, v, trafo_flag=2):
    """
    Estimate polynomial coefficients for coordinate mapping.

    Parameters
    ----------
    x : array_like
        Source x-coordinates.
    y : array_like
        Source y-coordinates.
    u : array_like
        Target u-coordinates.
    v : array_like
        Target v-coordinates.
    trafo_flag : int, optional
        Transformation order: 0=linear, 1=quadratic, 2=cubic. Default is 2.

    Returns
    -------
    numpy.ndarray
        Transformation coefficients.
        
    Notes
    -----
    Original implementation by Wouter Van den Broek.
    """
     # A cubic transformation
    no_params = 10  # number of parameters to estimate
    
    
    
    if trafo_flag == -1:
        print("Not enough valid calibration images: QUITING.")
        quit()
    if trafo_flag == 0:
        print("Calculating and correcting a LINEAR distortion model")
    if trafo_flag == 1:
        print("Calculating and correcting a QUADRATIC distortion model")
    if trafo_flag == 2:
        print("Calculating and correcting a CUBIC distortion model")

    no_params=10 if trafo_flag == 2 else (5 if trafo_flag == 1 else 2)

    A = np.ones((x.shape[0], no_params))
    A[:, 1] = x
    A[:, 2] = y
    if trafo_flag > 0:
        A[:, 3] = x * y
        A[:, 4] = x ** 2
        A[:, 5] = y ** 2
    if trafo_flag > 1:
        A[:, 6] = (x ** 2) * y
        A[:, 7] = x * (y ** 2)
        A[:, 8] = x ** 3
        A[:, 9] = y ** 3

    tmp = np.dot(np.transpose(A), A)
    ab = np.linalg.solve(tmp, np.dot(np.transpose(A), np.transpose([u, v])))

    print(ab)
    return ab


def unwarp_im(warp_im, ab, method="linear"):
    """
    Correct distortions and illumination in warped images.

    Parameters
    ----------
    warp_im : array_like
        Input image(s).
    ab : numpy.ndarray
        Transformation coefficients.
    method : str, optional
        Interpolation method. Default is 'linear'.

    Returns
    -------
    numpy.ndarray
        Unwarped image(s).
        
    Notes
    -----
    Original implementation by Wouter Van den Broek.
    """
    # Read the warped image
    g = 1* warp_im
    gshape=g.shape
    offset = 2
    assert np.all(g >= 0), "Values are below 0"
    g = np.log(g + offset)

    # Start the unwarping
    # coordinates of the warped image in uv-space
    u_g = np.arange(gshape[-2])
    v_g = np.arange(gshape[-1])
    uv_g = (u_g, v_g)

    # The mean scaling, probed over many different directions
    sc = np.sqrt(coordinate_transformation_2d_areamag(np.zeros((1, 2)), ab))

    # Coordinates in xy-space
    x_tmp = np.linspace(-0.5, 0.5, g.shape[-2]) * (g.shape[-2] - 1) / sc
    y_tmp = np.linspace(-0.5, 0.5, g.shape[-1]) * (g.shape[-1] - 1) / sc
    [x_tmp, y_tmp] = np.meshgrid(x_tmp, y_tmp, indexing="ij")
    x_tmp = np.ravel(x_tmp)
    y_tmp = np.ravel(y_tmp)

    # Transform those to uv-space
    uv_i = coordinate_transformation_2d((x_tmp, y_tmp), ab)
    
    # Area magnification in the middle is 1 now
    area_mag = coordinate_transformation_2d_areamag((x_tmp, y_tmp), ab)
    area_mag = np.reshape(area_mag, (gshape[-2], gshape[-1]))
    tmp = area_mag[int(round(gshape[-2] / 2)), int(round(gshape[-1] / 2))]
    area_mag = area_mag / tmp
    # Do the unwarping
    if len(gshape)==2:
        g = spip.interpn(uv_g, g, uv_i, method=method, bounds_error=False, fill_value=np.log(offset)
        )
        g = np.exp(np.reshape(g, (gshape[-2], gshape[-1]))-offset)*area_mag
    else:
        if len(gshape)==4:
            for i in tqdm(range(gshape[0])):
                for j in range(gshape[1]):
                    gij=spip.interpn( uv_g, g[i,j], uv_i, method=method, bounds_error=False, fill_value=np.log(offset))
                    g[i,j] = np.exp(np.reshape(gij, (gshape[-2], gshape[-1]))-offset)*area_mag
        elif len(gshape)==3:
            for i in tqdm(range(gshape[0])):
                gij=spip.interpn( uv_g, g[i], uv_i, method=method, bounds_error=False, fill_value=np.log(offset))
                g[i] = np.exp(np.reshape(gij, (gshape[-2], gshape[-1]))-offset)*area_mag
    return g


def coordinate_transformation_2d_areamag(xy, ab):
    """
    Compute local area magnification of polynomial mapping.

    Parameters
    ----------
    xy : array_like
        Coordinates of shape (2, N) or (N, 2).
    ab : numpy.ndarray
        Transformation coefficients.

    Returns
    -------
    numpy.ndarray
        Local magnification values.

    Notes
    -----
    Original implementation by Wouter Van den Broek.
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




def fit_vector_field(x, y, u, v, Xi, Yi):
    """
    Fit a smooth displacement field from calibration points.

    Parameters
    ----------
    x, y : array_like
        Nominal coordinates.
    u, v : array_like
        Observed coordinates.
    Xi, Yi : array_like
        Grid coordinates for mapping.

    Returns
    -------
    tuple of numpy.ndarray
        Displacement fields (dV, dU).
    """
    pts   = np.column_stack((y.ravel(), x.ravel()))
    u_flat, v_flat = (x-u).ravel(), (y-v).ravel()

    # by default it extrapolates everywhere
    rbf_u = RBFInterpolator(pts, u_flat)
    rbf_v = RBFInterpolator(pts, v_flat)

    Xi_f, Yi_f = Xi.ravel(), Yi.ravel()
    pts_new=np.column_stack((Yi_f, Xi_f))
    Ui_f = -rbf_u(pts_new)
    Vi_f = -rbf_v(pts_new)

    return Vi_f.reshape(Xi.shape), Ui_f.reshape(Xi.shape)

def warp_image(image, vector_field, order=1, extra_angle_rotation_compensation=0):
    """
    Warp image(s) based on a displacement field.

    Parameters
    ----------
    image : array_like
        Input image(s).
    vector_field : tuple of numpy.ndarray
        Displacement fields (U, V).
    order : int, optional
        Interpolation order. Default is 1.
    extra_angle_rotation_compensation : float, optional
        Rotation angle in degrees. Default is 0.

    Returns
    -------
    numpy.ndarray
        Warped image(s).
    """
    U, V = vector_field
    rows, cols = np.indices(image.shape[:2])
    rows_flatten, cols_flatten = rows.flatten(), cols.flatten()
    warped_rows = rows_flatten + U.flatten()
    warped_cols = cols_flatten + V.flatten()
    angle_rad=extra_angle_rotation_compensation*np.pi/180
    warped_image = map_coordinates(image, (warped_rows, warped_cols), order=order, mode='constant')
    warped_image = warped_image.reshape(image.shape)
    if extra_angle_rotation_compensation!=0:
        warped_image=rotate(warped_image, extra_angle_rotation_compensation, reshape=False,  order=order)
    
    return warped_image
    
    

def compose_fields(field1, field2, order=1, mode='nearest'):
    """
    Compose two displacement fields for use with warp_image.

    Resulting field corresponds to: first apply field1, then field2.

    Parameters
    ----------
    field1 : (U1, V1)
        First displacement field. Applied closest to the original image.
    field2 : (U2, V2)
        Second displacement field. Applied after field1.
    order : int, optional
        Interpolation order used to resample field1.
    mode : str, optional
        Boundary mode for map_coordinates (e.g. 'nearest', 'constant').

    Returns
    -------
    U_tot, V_tot : np.ndarray
        Composed displacement field such that:
        warp_image(warp_image(I, field1), field2)
        ≈ warp_image(I, (U_tot, V_tot))
    """
    U1, V1 = field1
    U2, V2 = field2

    assert U1.shape == V1.shape == U2.shape == V2.shape
    H, W = U1.shape

    rows, cols = np.indices((H, W), dtype=np.float64)
    sample_rows = rows + U2
    sample_cols = cols + V2
    U1_warped = map_coordinates(U1, (sample_rows, sample_cols),
                                order=order, mode=mode)
    V1_warped = map_coordinates(V1, (sample_rows, sample_cols),
                                order=order, mode=mode)

    # Total displacement: field1+ field2 evaluated at displaced coords
    U_tot = U2 + U1_warped
    V_tot = V2 + V1_warped

    return U_tot, V_tot

def warp_images_batch(images, vector_field, batches, warped_rows, warped_cols, order, extra_angle_rotation_compensation):
    """
    Apply warping to a batch of images using precomputed indices.

    Parameters
    ----------
    images : array_like
        Batch of images.
    vector_field : ignored
        Not used.
    batches : array_like
        Batch indices.
    warped_rows, warped_cols : array_like
        Flattened row and column indices.
    order : int
        Interpolation order.

    Returns
    -------
    numpy.ndarray
        Warped images.
    """
    warped_images = map_coordinates(images, (batches, warped_rows, warped_cols), order=order, mode='constant')
    warped_images = warped_images.reshape(images.shape)
    
    if extra_angle_rotation_compensation!=0:
        warped_images=rotate(warped_images,  extra_angle_rotation_compensation, axes=(2,1), reshape=False,  order=order)
    
    return warped_images
    
def unwarp_4dstem_batch(data_old, path_numpy_new=None, vector_field=None, batch_size=20, order=5, return_data=False, extra_angle_rotation_compensation=0):
    """
    Unwarp a 4D-STEM dataset in batches.

    Parameters
    ----------
    data_old : str or numpy.ndarray
        Input dataset.
    path_numpy_new : str, optional
        Path to save warped data.
    vector_field : tuple of numpy.ndarray
        Displacement fields.
    batch_size : int, optional
        Frames per batch. Default is 20.
    order : int, optional
        Interpolation order. Default is 5.
    return_data : bool, optional
        If True, return data instead of saving.

    Returns
    -------
    numpy.ndarray or None
        Warped dataset or None.

    Raises
    ------
    ValueError
        If total frames not divisible by batch_size.
    """
    print("Unwarping the dataset")
    if type(data_old)==str:
        data_old = np.load(path_numpy_old)
    data_old_shape=data_old.shape
    if len(data_old_shape)==4:
        data_old=data_old.reshape(data_old_shape[0]*data_old_shape[1], data_old_shape[2], data_old_shape[3])
    full_length=data_old_shape[0]*data_old_shape[1]
    U, V = vector_field[0], vector_field[1]
    batches, rows, cols = np.indices([batch_size,data_old.shape[1], data_old.shape[2]])
    U=np.tile(U, (batch_size,1,1))
    V=np.tile(V, (batch_size,1,1))
    batches, warped_rows, warped_cols = batches.flatten(), (rows + U).flatten(), (cols + V).flatten()
    if full_length%batch_size!=0:
        raise ValueError
    for index in tqdm(range(0, full_length, batch_size)):
        images=data_old[index:index+batch_size]
        images=warp_images_batch(images, vector_field, batches, warped_rows, warped_cols, order, extra_angle_rotation_compensation=extra_angle_rotation_compensation)
        data_old[index:index+batch_size]=images
    data_old=data_old.reshape(data_old_shape)
    data_old[data_old<0]=0
    if return_data:
        return data_old
    np.save(path_numpy_new)
    print("Dataset has been warped and saved in %s."%path_numpy_new)


def remove_hot_pixels(data, percentile=99.99):
    """
    Remove hot pixels from 4D-STEM data.

    Parameters
    ----------
    data : numpy.ndarray
        Input dataset.
    percentile : float, optional
        Threshold percentile. Default is 99.99.

    Returns
    -------
    numpy.ndarray
        Corrected dataset.
    """
    print("removing hot pixels")
    for i1 in tqdm(range(data.shape[0])):
        for i2 in range(data.shape[1]):
            pat=data[i1,i2,:,:]
            perc=np.percentile(pat, percentile)
            wy,wx=np.where(pat>perc)
            for i3 in range(len(wy)):
                wwy, wwx= wy[i3], wx[i3]
                try:
                    pat[wwy, wwx]=0.25*(pat[wwy+1, wwx]+ pat[wwy-1, wwx]+ pat[wwy, wwx+1]+ pat[wwy, wwx-1])
                except:
                    pat[wwy, wwx]=0
            pat[pat<0]=0
            data[i1,i2]=pat
    return data




def get_neg_contrast_ptycho(x, params, print_flag):
    try:
        angle=x[0]
    except:
        angle=x
    params=pyptyini.rotate_scan_grid(params, new_pl_rot=angle)
    pyptyit.run(params)
    
    transm=np.load(params["output_folder"]+"/co.npy")
    transm=pyptyutils.crop_ptycho_object_to_stem_roi(transm, params)
    
    transm=np.mean(np.angle(transm), (2,3))
    transm-=np.min(transm)

    transmbar=np.mean(transm)
    contrast=np.sum((transm-transmbar)**2)
    print("\nangle %.2f, contrast: %.3e"%(angle, -1*contrast))
    return -1*contrast


def find_twist_angle_ptycho_from_contrast(params, x0=0, print_flag=True, bounds=[(0,360)]):
    results=minimize(get_neg_contrast_ptycho, x0=x0, args=(params, print_flag), bounds=bounds, tol=1e-2, options={"maxiter":500})
    print("Angle %.2f"%results.x[0])
    return results.x[0]
    
        
