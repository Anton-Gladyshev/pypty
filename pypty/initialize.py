import numpy as np
import sys
import os
import h5py
import json
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
from pypty.utils import *
from pypty.dpc import *


def create_pypty_data(data, path_output, swap_axes=False,flip_ky=False,flip_kx=False, flip_y=False,flip_x=False,comcalc_len=1000, comx=None, comy=None, bin=1, crop_left=None, crop_right=None, crop_top=None, crop_bottom=None, normalize=False, cutoff_ratio=None, pad_k=0, data_dtype=np.float32, rescale=1, exist_ok=True):
    """
    Create a PyPty-style .h5 data.
        data - path to a dataset to be transformed (either .h5 or .npy array) or ndarray containing a 4D-STEM dataset.
        path_output - path where pypty-dataset will be stored
        
        swap_axes - boolean flag (default False) - swaps the last two coordianates
        flip_ky - boolean flag (default False) - flips the second last axis
        flip_kx - boolean flag (default False) - flips the  last axis
        flip_y  - boolean flag (default False) - flips the first axis
        flip_x  - boolean flag (default False) - flips the second axis
        comcalc_len - integer (default 1000), number of measurements to estimate the center 
        comx - integer or None (default None), x-center of the measuremts (if None it will be computed)
        comy - integer or None (default None), y-center of the measuremts (if None it will be computed)
        bin - integer (default 1), binning value applied to the last two axes.
        crop_left - integer (default None / 0) - left cropping of the patterns.
        crop_right - integer (default None / 0) - right cropping of the patterns.
        crop_top - integer (default None / 0) - top cropping of the patterns.
        crop_bottom - integer (default None / 0) - bottom cropping of the patterns.
        normalize - boolean flag (default False). If True, patterns will be rescaled so that on average the sum over a pattern is 1
        cutoff_ratio - float, default None. If not None, the values that are futher than cutoff_ratio x width/2 will be zeroed. 
        pad_k- integer (default 0), padding of the last two axes
        data_dtype- dtyle of the output file, default is np.float32, 
        rescale- float, default 1. If not 1, patterns will be divided by this number
        exist_ok- boolean flag (default True). Do not overwrite the file if it already exists.
    """
    sys.stdout.write("\n******************************************************************************\n************************ Creating an .h5 File ********************************\n******************************************************************************\n")
    sys.stdout.flush()
    if os.path.isfile(path_output):
        print("\n.h5 File exists!")
        if exist_ok:
            return None
        else:
            print("\nDeleting the exitsting one!")
            if path_output[-3:]==".h5" and data!=path_output:
                try:
                    os.remove(path_output)
                except:
                    pass
    if type(data)==str:
        if data[-4:]==".npy":
            data=np.load(data)
        else:
            f0=h5py.File(data, "r")
            data=np.array(f0["data"])
            f0.close()
    data=np.array(data)
    if flip_y:
        data=data[::-1,:,:,:]
    if flip_x:
        data=data[:,::-1,:,:]
    if flip_ky:
        data=data[:,:,::-1,:]
    if flip_kx:
        data=data[:,:,:,::-1]
    if swap_axes:
        data=np.swapaxes(data, 2,3)
    data=data.reshape(data.shape[0]*data.shape[1],data.shape[2], data.shape[3])
    data[data<0]=0
    x,y=np.arange(data.shape[2]), np.arange(data.shape[1])
    x,y=np.meshgrid(x-np.mean(x), y-np.mean(y), indexing="xy")
    r=(x**2+y**2)**0.5
    if comcalc_len is None:
        comcalc_len=data.shape[0]
    ssum=np.sum(data[:comcalc_len], axis=(1,2))
    if comx is None:
        comx=int(np.round(np.mean(np.sum(data[:comcalc_len]*x[None, :,:], axis=(1,2))/ssum)))
    if comy is None:
        comy=int(np.round(np.mean(np.sum(data[:comcalc_len]*y[None, :,:], axis=(1,2))/ssum)))
    sys.stdout.write("\nCOM x&y before correction: %d, %d"%( comx,comy))
    data=np.roll(data, (-comy, -comx), axis=(1,2))
    if comy>0:
        data[:,-comy:,:]=0
    elif comy<0:
        data[:, :-comy, :]=0
    if comx>0:
        data[:,:,-comx:]=0
    elif comx<0:
        data[:,:,:-comx]=0
    if not(cutoff_ratio is None):
        r= r>= np.max(x)*cutoff_ratio
        data[:, r]=0
    comx=int(np.round(np.mean(np.sum(data[:comcalc_len]*x[None, :,:], axis=(1,2))/ssum)))
    comy=int(np.round(np.mean(np.sum(data[:comcalc_len]*y[None, :,:], axis=(1,2))/ssum)))
    sys.stdout.write("\nCOM x&y after correction: %d, %d"%( comx,comy))
    if normalize:
        ssum=np.sum(data[:comcalc_len], axis=(1,2))
        data/=np.mean(ssum)
    if rescale!=1:
        data/=rescale
    if not(crop_bottom is None):
        data=data[:,:-crop_bottom, :]
    if not(crop_top is None):
        data=data[:,crop_top:, :]
    if not(crop_left is None):
        data=data[:,:, crop_left:]
    if not(crop_right is None):
        data=data[:,:, :-crop_right]
    if bin!=1:
        data2=np.zeros((data.shape[0], data.shape[1]//bin, data.shape[2]//bin))
        for i in range(bin):
            for j in range(bin):
                data2+=data[:,i:bin*(data.shape[1]//bin):bin, j:bin*(data.shape[1]//bin):bin]
        data=data2
        del(data2)
    if pad_k!=0:
        data=np.pad(data, [[0,0],[pad_k, pad_k],[pad_k,pad_k]])
    if not(data_dtype is None):
        data=data.astype(data_dtype)
    f=h5py.File(path_output, "a")
    f.create_dataset("data", data=data)
    f.close()


    
def get_offset(x_range, y_range, scan_step_A, detector_pixel_size_rezA, patternshape, rot_angle_deg=0):
    """
    get_offset, i.e. number of pixels from the top and left of the reconstruction grid and the first point of the scan grid. In PyPty framework, scan grid is usually rotated to compensate the misalignment between scan- and detector- axes. Also, a reconstruction grid is larger than the scanned FOV, this is done to accomodate the extent of the probe. 
    Inputs
        x_range- number of points on the fast axis
        y_range- number of points on the slow axis
        scan_step_A- STEM pixel size (in Angstrom)
        detector_pixel_size_rezA- pixel size in a detector plane (A^-1)
        patternshape- shape of the diffraction patterns.
        rot_angle_deg- angle between scan and detector axers
    Outputs
        offy- offset in y direction (reconstruction pixels)
        offx- offset in x direction (reconstruction pixels)
    """
    pixel_size=1/(detector_pixel_size_rezA*patternshape[-1])
    positions=np.empty((x_range*y_range,2))
    i=0
    for ind1 in range(0,y_range,1):
        for ind2 in range(0, x_range,1):
            positions[i,0]=ind1
            positions[i,1]=ind2
            i+=1
    if rot_angle_deg!=0:
        rot_ang=rot_angle_deg*np.pi/180
        positions_prime_x,positions_prime_y=positions[:,1] * np.cos(rot_ang) + positions[:,0] * np.sin(rot_ang), -1*positions[:,1] * np.sin(rot_ang) + positions[:,0] * np.cos(rot_ang)
        positions[:,0]=positions_prime_y
        positions[:,1]=positions_prime_x
    offy=-np.min(positions[:,0])*scan_step_A
    offx=-np.min(positions[:,1])*scan_step_A
    return offy, offx
       
def get_positions_pixel_size(x_range, y_range,scan_step_A, detector_pixel_size_rezA, patternshape, rot_angle_deg=0, flip_x=False,flip_y=False, print_flag=False):
    pixel_size=1/(detector_pixel_size_rezA*patternshape[-1])
    if print_flag:
        sys.stdout.write("\npixel size in A: %.3e"%pixel_size)
    positions=np.empty((x_range*y_range,2))
    i=0
    for ind1 in range(0,y_range,1):
        for ind2 in range(0, x_range,1):
            positions[i,0]=ind1
            positions[i,1]=ind2
            i+=1
    if rot_angle_deg!=0:
        rot_ang=rot_angle_deg*np.pi/180
        positions_prime_x,positions_prime_y=positions[:,1] * np.cos(rot_ang) + positions[:,0] * np.sin(rot_ang), -1*positions[:,1] * np.sin(rot_ang) + positions[:,0] * np.cos(rot_ang)
        positions[:,0]=positions_prime_y
        positions[:,1]=positions_prime_x
    if flip_x:
        positions[:,1]*=-1
    if flip_y:
        positions[:,0]*=-1
    positions[:,0]=positions[:,0]-np.min(positions[:,0])
    positions[:,1]=positions[:,1]-np.min(positions[:,1])
    positions*=scan_step_A/pixel_size
    return positions, pixel_size


def get_grid_for_upsampled_image(pypty_params, image,image_pixel_size, left_zero_of_scan_grid=0, top_zero_of_scan_grid=0):
    """
    This function calculates where pixel of an arbitary image (e.g. upsampled tcBF image) will land on a grid corresponding to a ptychographic reconstruction.
    Inputs:
        -pypty_params- dictionary with callibrated pypty parameters
        -image, the image (2D numpy array) for which the computation should be done
        -image_pixel_size, pixel size of the image (in Å)
        -left_zero_of_scan_grid- integer. 
        -top_zero_of_scan_grid- integer.
        Both left_zero_of_scan_grid and top_zero_of_scan_grid incicate after how many image pixels does the actual scan grid. For example, if your image extens beyond the actual scan grid, this two values are positive. If the image is created from a subscan, these two values should de negative.
    Outputs:
        -sc - flattened meshgrid [[y,x],..[]] with coordinates of image pixels on a reconsturction basis
    """

    scx, scy=np.meshgrid((np.arange(0, image.shape[1],1)-left_zero_of_scan_grid)*image_pixel_size,
                        (np.arange(0, image.shape[0],1)-top_zero_of_scan_grid)*image_pixel_size,
                        indexing="xy")
    rot_ang=pypty_params["PLRotation_deg"]*np.pi/180
    sc_prime_x,sc_prime_y=scx * np.cos(rot_ang) - scy * np.sin(rot_ang), scx * np.sin(rot_ang) + scy * np.cos(rot_ang)
    ofy, ofx=get_offset(x_range=pypty_params["scan_size"][1],
                        y_range=pypty_params["scan_size"][0],
                        scan_step_A=pypty_params["scan_step_A"],
                        detector_pixel_size_rezA=pypty_params["rez_pixel_size_A"],
                        patternshape=pypty_params["aperture_mask"].shape,
                        rot_angle_deg=-1*pypty_params["PLRotation_deg"])
    sc_prime_x, sc_prime_y=sc_prime_x+ofx, sc_prime_y+ofy
    sc=np.swapaxes(np.array([sc_prime_y.flatten(), sc_prime_x.flatten()]),0,1)
    return sc

    
    
def append_exp_params(experimental_params, pypty_params=None):
    """
    Callibrate an extisting PyPty preset to new data. 
    
    Inputs:
        -experimental_params - dictionary
        -pypty_params - dictionary / sting-path to an existing preset / None
    Output:
        -pypty_params - dictionary
    
    experimental_params should contain following entries:
        -data_path - path to a PyPty-style 3d .h5 file [N_measurements, ky,kx] or .npy Nion-style 4d-stem dataset (or 3d .npy dataset)
        -masks - 3d numpy array or None. if data is compressed provide the virtual detectors (masks) shape should be [N_masks,ky,kx]
        -output_folder - path to an outputfolder where the results will be stored
        -path_json - path to a nion-style json file with metadata (optional)
        -acc_voltage - float, accelerating voltage in kV
        
        One or multiple of the following callibrations:
            -rez_pixel_size_A - reciprocal pixel size in Å^-1
            -rez_pixel_size_mrad - reciprocal pixel size in mrad
            
            -conv_semiangle_mrad - beam convergence semi-angle in mrad
            -aperture - (optional)- binary 2D mask
            -bright_threshold - threshold to estimate an aperture, everything above threshold times maximum value in a pacbed will be concidered as bright field disk.
        
        -data_pad - int, reciprocal space padding. If None (default), pading is 1/4 of the total width of a diffraction pattern
        -upsample_pattern - int, default 1 (no upsampling)
        
        -aberrations - list or 1d numpy array containing beam aberrations (in Å). Aberrations are stored in Krivanek notation, e.g. C10, C12a, C12b, C21a, C21b, C23a, C23b, C30 etc
        -defocus - float, default 0. Extra probe defocus besides the one contained in aberrations.
        
        -scan_size - tuple of two ints, number of scan points along slow (y) and fast (x) axes. Optional. If no scan step or position grid is provided, it will be used to get the scan step
        -scan_step_A - float, scan step (STEM pixel size) in Å.
        -fov_nm - float, FOV along the fast axis in nm.
        -special_postions_A - 2d numpy array, default None. If you acquiered a data on a special non-rectangular grid, please specify the positions in Å via this array for all measurements in a following form: [y_0,x_0],[y_1,x_1],....[y_n,x_n]]
        
        -PLRotation_deg - float, rotation angle between scan and detector axes. Default None. If None, a DPC measurement will be exectuted to get this angle. !!!!!!! Note that negative PLRotation_deg values rotate scan counter clockwise and diffraction space clockwise !!!!!!!!!!!
        -data_is_numpy_and_flip_ky - boolean Flag. Default is False. If no PyPty-style h5 data was created, this flag will flip the y-axis of diffraction patterns.
        
        -total_thickness - total thickness of a sample in Å. Has no effect if num_slices is 1 and propagation method (pypty_params entry) is multislice 
        -num_slices - integer, number of slices, default is 1.
        
        -plot - boolean Flag, default is True 
        -print_flag - integer. Default is 1. If 0 nothing will be printed. 1 prints only thelatest state of the computation, 2 prints every state as a separate line. 3 prints the linesearch progress in iterative optimization. 4 prints everything that 3 does and if constraints are applied, it prints how they contribute so that a user can configure the weights properly.
        -save_preprocessing_files - Boolean Flag. Default is True. 
        
    """
    sys.stdout.write("\n******************************************************************************\n******** Attaching the experimental parameters to your PyPty preset. *********\n******************************************************************************\n")
    sys.stdout.flush()
    if type(pypty_params)==str:
        pypty_params=load_params(pypty_params)
    
    path_data_h5=experimental_params.get("data_path", "")
    masks=experimental_params.get("masks", None)
    output_folder=experimental_params.get("output_folder", "")
    path_json=experimental_params.get("path_json", "")
    
    acc_voltage=experimental_params.get("acc_voltage", None)
    rez_pixel_size_A=experimental_params.get("rez_pixel_size_A", None)
    rez_pixel_size_mrad=experimental_params.get("rez_pixel_size_mrad", None)
    conv_semiangle_mrad=experimental_params.get("conv_semiangle_mrad", None)
    aperture=experimental_params.get("aperture", None)
    bright_threshold=experimental_params.get("bright_threshold", 0.1)
    
    data_pad=experimental_params.get("data_pad", None)
    upsample_pattern=experimental_params.get("upsample_pattern",1)
    aberrations = experimental_params.get("aberrations", np.zeros(8))
    defocus=experimental_params.get("defocus", 0)

    scan_size=experimental_params.get("scan_size", None)
    scan_step_A=experimental_params.get("scan_step_A", None)
    fov_nm=experimental_params.get("fov_nm", None)
    special_postions_A=experimental_params.get("special_postions_A", None)
 
    PLRotation_deg=experimental_params.get("PLRotation_deg", None)
    data_is_numpy_and_flip_ky=experimental_params.get("data_is_numpy_and_flip_ky", False)
    
    total_thickness=experimental_params.get("total_thickness", 1)
    num_slices=experimental_params.get("num_slices", 1)
    
    plot=experimental_params.get("plot", True)
    print_flag=experimental_params.get("print_flag", True)
    save_preprocessing_files=experimental_params.get("save_preprocessing_files", True)
    
    comx=None
    comy=None
    try:
        os.makedirs(output_folder, exist_ok=True)
    except:
        pass
    if output_folder[:-1]!="/": output_folder+="/";
    try:
        with open(path_json, 'r') as file:
            jsondata = json.load(file)
    except:
        if print_flag:
            sys.stdout.write("\njson is not provided!")
    if path_data_h5[-3:]==".h5":
        h5file=h5py.File(path_data_h5, "r")
        h5data=h5file["data"]
    elif path_data_h5[-4:]==".npy":
        h5data=np.load(path_data_h5)
        if len(h5data.shape)==4:
            scan_size=[h5data.shape[0], h5data.shape[1]]
            h5data=h5file.reshape(h5data.shape[0]* h5data.shape[1], h5data.shape[2],h5data.shape[3])
        if data_is_numpy_and_flip_ky:
            h5data=h5data[:,::-1, :]
    if scan_size is None:
        try:
            scan_size=jsondata['metadata']['scan']['scan_size'];
        except:
            pass
    if acc_voltage is None:
        try:
            acc_voltage=jsondata['metadata']['hardware_source']['high_tension_v']*1e-3; ##kV
        except:
            print("\nAssuming 60kV of acc. voltage!")
            acc_voltage=60
    if defocus is None:
        try:
            defocus=-jsondata['metadata']['hardware_source']['defocus']; ## unknown units
        except:
            print("\nAssuming zero defocus!")
            defocus=0
    if fov_nm is None:
        try:
            fov_nm=jsondata['metadata']['scan']['fov_nm'];
        except:
            fov_nm=0
            print("\nNo scan-FOV provided, specify scan step!")
    if scan_step_A is None:
        try:
            scan_step_A=fov_nm*10/scan_size[0]
        except:
            pass
    
    # conv_semiangle_mrad
    # rez_pixel_size_A
    # PLRotation_deg
    wavelength=12.4/np.sqrt(acc_voltage*(acc_voltage+2*511))
    y_range,x_range=scan_size
    if PLRotation_deg is None:
        x,y=np.arange(h5data.shape[-1]), np.arange(h5data.shape[-2])
        x,y=np.meshgrid(x-np.mean(x), y-np.mean(y), indexing="xy")
        ssum=np.empty(scan_size)
        if (comx is None) or (comy is None):
            comx=np.empty(scan_size)
            comy=np.empty(scan_size)
            for index_data_y in range(scan_size[0]):
                for index_data_x in range(scan_size[1]):
                    dataindex=index_data_x+index_data_y*scan_size[1]
                    ssum[index_data_y, index_data_x]=np.sum(h5data[dataindex])
                    comx[index_data_y, index_data_x]=np.sum(h5data[dataindex]*x)
                    comy[index_data_y, index_data_x]=np.sum(h5data[dataindex]*y)
            comx=comx/ssum.astype(np.float32)
            comy=comy/ssum.astype(np.float32)
            comx-=np.mean(comx)
            comy-=np.mean(comy)
        PLRotation=GetPLRotation(comx,comy)
        PLRotation_deg=PLRotation*180/np.pi ## we need a negative angle for the scan grid rotation
        if print_flag:
            sys.stdout.write("\niDPC rotation angle is %.2f deg. (Rotation of the reciprocal space with respect to real space.)"%PLRotation_deg)
    mean_pattern_as_it_is=np.mean(h5data, axis=0)
    if not(masks is None):
        mean_pattern_as_it_is=np.sum(masks*mean_pattern_as_it_is[:,None,None], axis=0)
    if upsample_pattern!=1:
        x,y=np.meshgrid(np.linspace(0,1,mean_pattern_as_it_is.shape[1]),np.linspace(0,1,mean_pattern_as_it_is.shape[0]))
        points=np.swapaxes(np.array([x.flatten(),y.flatten()]), 0,1)
        x2, y2=np.meshgrid(np.linspace(0,1,upsample_pattern*mean_pattern_as_it_is.shape[1]), np.linspace(0,1,upsample_pattern*mean_pattern_as_it_is.shape[0]))
        mean_pattern=np.abs(griddata(points, mean_pattern_as_it_is.flatten(), (x2, y2), method='cubic'))
    else:
        mean_pattern=mean_pattern_as_it_is
    if aperture is None:
        aperture=mean_pattern>bright_threshold*np.max(mean_pattern)
    if plot:
        try:
            fig, ax=plt.subplots(1,2)
            ax[0].imshow(mean_pattern)
            ax[0].set_title("mean pattern")
            ax[1].imshow(aperture)
            ax[1].set_title("bright field disk")
            plt.show()
        except:
            pass
    if rez_pixel_size_mrad is None:
        if rez_pixel_size_A is None:
            r=np.sqrt(np.sum(aperture)/np.pi)
            if print_flag:
                sys.stdout.write("\nRadius of bright field is %.2f px"%r)
            rez_pixel_size_A=conv_semiangle_mrad*1e-3/(r*wavelength) ## A^-1
        else:
            rez_pixel_size_A/=upsample_pattern
    else:
        rez_pixel_size_A=rez_pixel_size_mrad*1e-3/(upsample_pattern*wavelength)
    old_shape=mean_pattern.shape[1]
    if special_postions_A is None:
        positions, old_pixel_size=get_positions_pixel_size(x_range, y_range, scan_step_A, detector_pixel_size_rezA=rez_pixel_size_A, patternshape=[mean_pattern.shape[0],mean_pattern.shape[1]], rot_angle_deg=-1*PLRotation_deg, flip_x=False,flip_y=False, print_flag=print_flag)
    else:
        old_pixel_size=1/(rez_pixel_size_A*old_shape)
        positions=special_postions_A/old_pixel_size
        positions[:,0]-=np.min(positions[:,0])
        positions[:,1]-=np.min(positions[:,1])
        
    
    if data_pad is None: data_pad=int(np.ceil(old_shape/4));
    new_shape=old_shape+2*data_pad
    new_pixel_size=old_pixel_size*old_shape/new_shape
    positions *= new_shape/old_shape
    if print_flag:
        sys.stdout.write("\nPixel size after padding: %.2e Å"%new_pixel_size)
    aperture=np.pad(aperture, data_pad, mode="constant", constant_values=0)
    if pypty_params is None:
        pypty_params={
        'data_path': path_data_h5,
        'output_folder': output_folder,
        'positions': positions,
        'obj': np.ones((1,1,num_slices,1), dtype=np.complex128),
        'acc_voltage': acc_voltage,
        'slice_distances': np.array([total_thickness/num_slices]),
        'pixel_size_x_A': new_pixel_size,
        'pixel_size_y_A': new_pixel_size,
        'aperture_mask': aperture,
        'extra_probe_defocus': defocus,
        'data_pad': data_pad,
        'probe': None,
        'epoch_max': 1000,
        'epoch_prev': 0,
        'print_flag': 3,
        }
    else:
        pypty_params['data_path']=path_data_h5
        pypty_params['output_folder']=output_folder
        pypty_params['positions']=positions
        pypty_params['acc_voltage']=acc_voltage
        pypty_params['slice_distances']=np.array([total_thickness/num_slices])
        pypty_params['pixel_size_x_A']=new_pixel_size
        pypty_params['pixel_size_y_A']=new_pixel_size
        pypty_params['aperture_mask']=aperture
        pypty_params['extra_probe_defocus']=defocus
        pypty_params['data_pad']=data_pad
        pypty_params['probe']=None
        pypty_params['obj']=np.ones((1,1,num_slices,1), dtype=np.complex128)
    pypty_params["masks"]=masks
    if not(masks is None): pypty_params["algorithm"]="lsq_compressed";
    pypty_params["data_is_numpy_and_flip_ky"]=data_is_numpy_and_flip_ky
    pypty_params["save_preprocessing_files"]=save_preprocessing_files
    pypty_params["aberrations"]=aberrations
    pypty_params["mean_pattern"]=mean_pattern_as_it_is
    pypty_params["upsample_pattern"]=upsample_pattern
    pypty_params["rez_pixel_size_A"]=rez_pixel_size_A
    pypty_params["conv_semiangle_mrad"]=conv_semiangle_mrad
    pypty_params["scan_size"]=scan_size
    pypty_params["scan_step_A"]=scan_step_A
    pypty_params["fov_nm"]=fov_nm
    pypty_params["PLRotation_deg"]=PLRotation_deg
    pypty_params["bright_threshold"]=bright_threshold
    pypty_params["plot"]=plot
    pypty_params["comx"]=comx
    pypty_params["comy"]=comy
    pypty_params["print_flag"]=print_flag
    pypty_params["num_slices"] = num_slices
    pypty_params["total_thickness"] = total_thickness
    try:
        h5file.close()
    except:
        pass
    return pypty_params
    
    
def get_ptycho_obj_from_scan(params, num_slices=None, array_phase=None,array_abs=None, scale_phase=1, scale_abs=1,  scan_array_A=None, fill_value_type=None):
    """
    Create an intial guess for the object based on another arrays via interpolation. You can use output of dpc, wdd of tcBF reconstructions to generate it.
    Inputs:
        params- dictionary with callibrated pypty parameters.
        num_slices- integer, number of slices. Dominates over the entry "num_slices" in pypty parameters. Can be also specified as "auto". Then the number of slices will be computed such that the phase shift in each slice does not exceed 0.75*pi
        array_phase- 2D array for the phase (Default none, meaning 0 phase shift)
        array_abs- 2D array for the amplitude (Default none, meaning no absorption or gain)
        scale_phase- float, default 1 (rescale for the phase array)
        scale_abs- float, default 1 (rescale for the abs array)
        
        scan_array_A- default None. Array indicating the pixels of the array_phase and array_abs on the reconstruction grid. Not requiered if array_phase and array_abs are sampled on the scan grid. 
        fill_value_type- Default is None. Indicates what to fill on the edges of the arrays and beyond. None  fills 0s for phase and 1s for amplitude. Other options are "edge" and "median".
    Outputs:
        params- updated dictionary
    """
    data_path=params.get("data_path", "")
    try:
        if data_path[-2:]=="h5":
            this_file=h5py.File(data_path, "r")
            dataset=this_file['data']
        else:
            dataset=np.load(data_path, mmap_mode="r")
        if len(dataset.shape)==2:
            masks=params.get("masks", None)
            psx,psy=masks.shape[-2], masks.shape[-1]
            psx, psy=psx+2*data_pad, psy+2*data_pad
        else:
            psx,psy=dataset.shape[-2], dataset.shape[-1]
            psx, psy=psx+2*data_pad, psy+2*data_pad
        try:
            this_file.close()
        except:
            pass
    except:
        try:
            aperture=params.get("aperture_mask", None)
            psx,psy=aperture.shape[1], aperture.shape[0]
        except:
            probe=params.get("probe", None)
            psx,psy=probe.shape[1], probe.shape[0]
    data_pad=params.get("data_pad", 0)
    if num_slices is None:
        num_slices=params.get("num_slices", 1)
    if num_slices == "auto":
        num_slices= int(np.ceil((np.max(array_phase)-np.min(array_phase))/(0.75*np.pi)))
    positions=np.copy(np.array(params.get("positions", [[0,0]])))
    sequence=params.get("sequence", None)
    pixel_size_x_A=params.get("pixel_size_x_A", 1)
    scan_step_A=params.get("scan_step_A", 1)
    total_thickness=params.get("total_thickness", 1)
    
    cutoff=pixel_size_x_A/scan_step_A
    use_full_FOV=params.get("use_full_FOV", True)
    positions=np.round(positions).astype(int)
    if scan_array_A is None:
        scan_y, scan_x=positions[:,0], positions[:,1]
    else:
        scan_y, scan_x=scan_array_A[:,0]/pixel_size_x_A, scan_array_A[:,1]/pixel_size_x_A
    if not(use_full_FOV):
        scan_y, scan_x=scan_y-np.min(scan_y), scan_x-np.min(scan_x)
    scan_y, scan_x=scan_y+psy//2, scan_x+psx//2
    scan_y, scan_x = scan_y.astype(int), scan_x.astype(int)
    max_y_px,max_x_px=np.max(scan_y)+psy-psy//2,np.max(scan_x)+psx-psx//2
    max_y_px,max_x_px=np.max(positions[:,0])+psy, np.max(positions[:,1])+psx
    image_gird_y, image_gird_x=np.arange(max_y_px), np.arange(max_x_px)
    im_X,im_Y=np.meshgrid(image_gird_x, image_gird_y, indexing="xy")
    
    interpolator=CloughTocher2DInterpolator
    #else:
     #   interpolator=cupyx.scipy.interpolate.CloughTocher2DInterpolator
    #cupyx.scipy.interpolate.CloughTocher2DInterpolator
    if array_phase is None:
        phase_ptycho=0
    else:
        if fill_value_type=="median":
            fill_value=np.median(array_phase)
        else:
            if fill_value_type=="edge":
                fill_value=np.max([np.min(array_phase[:3,:]), np.min(array_phase[-3:,:]), np.min(array_phase[:,:3]), np.min(array_phase[:,-3:])])
            else:
                fill_value=0
        f=interpolator(list(zip(scan_x, scan_y)), array_phase.flatten(), fill_value=fill_value)
        phase_ptycho=f(im_X,im_Y)
       
    if array_abs is None:
        abs_ptycho=1
    else:
        if fill_value_type=="median":
            fill_value=np.median(array_abs)
        else:
            if fill_value_type=="edge":
                fill_value=np.mean([np.mean(array_abs[:1,:]), np.mean(array_abs[-1:,:]), np.mean(array_abs[:,:1]), np.mean(array_abs[:,-1:])])
            else:
                fill_value=0
        f_abs=CloughTocher2DInterpolator(list(zip(scan_x, scan_y)), array_abs.flatten(), fill_value=fill_value)
        abs_ptycho=f_abs(im_X,im_Y)
    ptycho=scale_abs*(abs_ptycho**(1/num_slices))*np.exp(1j*phase_ptycho*scale_phase/num_slices)
    if not(sequence is None) and not(use_full_FOV):
        minx, miny, maxx, maxy=np.min(positions[sequence,1]), np.min(positions[sequence,0]), np.max(positions[sequence,1]), np.max(positions[sequence,0])
        ptycho=ptycho[miny:maxy+psy, minx: maxx+ psx]
    
    ptycho=np.expand_dims(np.tile(np.expand_dims(ptycho,-1), (num_slices)),-1)
    params["slice_distances"] =  np.array([total_thickness / num_slices])
    params["obj"] = ptycho
    return params



def create_aberrations_chunks(pypty_params,chop_size, n_abs):
    """
    Creates chunks, i.e. multiple subscans with independent beam aberrations. Usefull for large fields of view where the beam is varyying. If applied, the iterative reconstruction will have the same beam in each subscan, but apply a different CTF in each of these regions.
    Inputs:
        pypty_params- dictionary with callibrated pypty parameters.
        chop_size- size of each subscan (width in scan points)
        n_abs- number of aberrations in each chop.
    Outputs:
        pypty_params- updatedd dictionary
    """
    scan_size=pypty_params.get("scan_size", None)
    sh0,sh1=scan_size
    aberration_marker=np.zeros((sh0,sh1))
    n_chops_0=int(np.ceil(sh0/chop_size))
    n_chops_1=int(np.ceil(sh1/chop_size))
    for i in range(n_chops_0):
        for j in range(n_chops_1):
            aberration_marker[i*chop_size:(i+1)*chop_size,j*chop_size:(j+1)*chop_size]=i*n_chops_0+j
    pypty_params['aberrations_array']  = np.zeros((n_chops_0*n_chops_1, n_abs), dtype=np.float32)
    pypty_params['aberration_marker'] = (aberration_marker.flatten()).astype(int)
    return pypty_params


def create_probe_marker_chunks(pypty_params,chop_size):
    """
    Creates chunks, i.e. multiple subscans with independent beam aberrations. Usefull for large fields of view where the beam is varyying. If applied, the iterative reconstruction will have the a differenet beam in each of these subscans.
    Inputs:
        pypty_params- dictionary with callibrated pypty parameters.
        chop_size- size of each subscan (width in scan points)
    Outputs:
        pypty_params- updatedd dictionary
    """
    scan_size=pypty_params.get("scan_size", None)
    sh0,sh1=scan_size
    probe_marker=np.zeros((sh0,sh1))
    n_chops_0=int(np.ceil(sh0/chop_size))
    n_chops_1=int(np.ceil(sh1/chop_size))
    for i in range(n_chops_0):
        for j in range(n_chops_1):
            probe_marker[i*chop_size:(i+1)*chop_size,j*chop_size:(j+1)*chop_size]=i*n_chops_0+j
    pypty_params['probe_marker'] = (probe_marker.flatten()).astype(int)
    return pypty_params





    
    
def create_sub_sequence(pypty_params, left, top, width, height, sub):
    """
    Creates subsequence for a quick reconstruction in a small ROI. 
    Inputs:
        pypty_params- dictionary with callibrated pypty parameters.
        left- integer (left edge in scan points)
        top- integer (top edge in scan points)
        width- integer, (width in scan points)
        height- integer (height in scan points)
        sub- integer. If larer than 1, only every sub's measurement along fast- and slow- axes will end up in a sequence
    Outputs:
        pypty_params- updatedd dictionary
    """
    seq=[]
    scan_size=pypty_params["scan_size"]
    ii1_list=np.arange(top, top+height, sub)
    jj1_list=np.arange(left, left+width, sub)
    for ii1 in ii1_list:
        for jj1 in jj1_list:
            seq.append(int(scan_size[1]*ii1+jj1))
    seq=list(np.unique(seq))
    pypty_params["sequence"]=seq
    return pypty_params
    
    
    
    
    
def create_sequence_from_points(pypty_params, yf,xf, width_roi=20):
    """
    Creates subsequence for a quick reconstruction in a small ROI. This function is useful if you have only a few interesting features in a scan and want to perform a reconsturction around them.
    Inputs:
        pypty_params- dictionary with callibrated pypty parameters.
        yf- list of integers containing y-coordinates of features (in scan points)
        yf- list of integers containing x-coordinates of features (in scan points)
        width_roi- integer, (width in scan points)
    Outputs:
        pypty_params- updatedd dictionary
    """
    scan_size=pypty_params["scan_size"]
    seq=[]
    for i in range(len(yf)):
        ty,tx=yf[i], xf[i]
        for w in range(-width_roi//2,width_roi-width_roi//2,1):
            for h in range(-width_roi//2,width_roi-width_roi//2,1):
                seq.append((ty+w)*scan_size[1]+h+tx)
    seq=np.array(seq)
    seq=seq[(seq>=0)*(seq<(scan_size[0]*scan_size[1]))]
    seq=list(np.unique(seq))
    return seq




def rotate_scan_grid(pypty_params, angle_deg):
    """
    This function rotates the scan grid.
    Inputs:
        pypty_params- dictionary with callibrated pypty parameters.
        angle_deg- angle of rotation is degrees.
    Outputs:
        pypty_params- updatedd dictionary
    """
    old_pl_rot   = pypty_params["PLRotation_deg"]
    new_pl_rot   = old_pl_rot + angle_deg
    old_postions = pypty_params["positions"]
    opy, opx=old_postions[:,0], old_postions[:,1]
    rot_ang=np.pi*angle_deg/180
    opx_prime, opy_prime=opx * np.cos(rot_ang) + opy * np.sin(rot_ang), -1*opx * np.sin(rot_ang) + opy * np.cos(rot_ang)
    opx_prime-=np.min(opx_prime)
    opy_prime-=np.min(opy_prime)
    old_postions[:,1]=opx_prime
    old_postions[:,0]=opy_prime
    pypty_params["PLRotation_deg"]=new_pl_rot
    pypty_params["positions"]=old_postions
    return pypty_params
    

def conjugate_beam(pypty_params):
    aberrations=pypty_params.get("aberrations", None)
    if not(aberrations is None):
        pypty_params["aberrations"]=-1*aberrations
    defocus=pypty_params.get("extra_probe_defocus", None)
    if not(defocus is None):
        pypty_params["extra_probe_defocus"]=-1*defocus
    probe=pypty_params.get("probe", None)
    if not(probe is None):
        pypty_params["probe"]=np.fft.ifft2(np.conjugate(np.fft.fft2(probe, axes=(0,1))), axes=(0,1))
    return pypty_params
    
    
    
def get_focussed_probe_from_vacscan(pypty_params, mean_pattern):
    """
    This function creates a focussed probe from a vacuum measurement.
    Inputs:
        pypty_params- dictionary with callibrated pypty parameters.
        mean_pattern- 2D diffraction pattern acquiered in vacuum.
    Outputs:
        pypty_params- updatedd dictionary
    """
    upsample_pattern= pypty_params.get("upsample_pattern", 1)
    data_pad=pypty_params.get("data_pad", 0)
    if upsample_pattern!=1:
        x,y=np.meshgrid(np.linspace(0,1,mean_pattern.shape[1]),np.linspace(0,1,mean_pattern.shape[0]))
        points=np.swapaxes(np.array([x.flatten(),y.flatten()]), 0,1)
        x2, y2=np.meshgrid(np.linspace(0,1,upsample_pattern*mean_pattern.shape[1]), np.linspace(0,1,upsample_pattern*mean_pattern.shape[0]))
        mean_pattern=np.abs(griddata(points, mean_pattern.flatten(), (x2, y2), method='cubic'))
    mean_pattern=np.pad(mean_pattern, data_pad)
    focussed_probe=np.expand_dims(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift((mean_pattern**0.5)))),-1)
    pypty_params["probe"]=focussed_probe
    return pypty_params

def append_aperture_to_params(pypty_params, mean_pattern):
    """
    Append a measured aperture (vacuum measurement). This function will apply padding an upsampling to the aprture.
    Inputs:
        pypty_params- dictionary with callibrated pypty parameters.
        mean_pattern- apperture to be appended
    returns:
        pypty_params- updated dictionary
    """
    upsample_pattern= pypty_params.get("upsample_pattern", 1)
    data_pad=pypty_params.get("data_pad", 0)
    if upsample_pattern!=1:
        x,y=np.meshgrid(np.linspace(0,1,mean_pattern.shape[1]),np.linspace(0,1,mean_pattern.shape[0]))
        points=np.swapaxes(np.array([x.flatten(),y.flatten()]), 0,1)
        x2, y2=np.meshgrid(np.linspace(0,1,upsample_pattern*mean_pattern.shape[1]), np.linspace(0,1,upsample_pattern*mean_pattern.shape[0]))
        mean_pattern=np.abs(griddata(points, mean_pattern.flatten(), (x2, y2), method='cubic'))
    mean_pattern=np.pad(mean_pattern, data_pad)
    pypty_params["aperture_mask"]=mean_pattern
    return pypty_params



def tiltbeamtodata(pypty_params, align_type="com"):
    """
    This function alines the momentum space of the beam with the diffraction patterns in a dataset.
    Inputs:
        pypty_params- dictionary with callibrated pypty parameters.
        align_type- string, defualt is "com", meaning center of mass. Other option is phase cross-correlation, which is activated by any value other than "com".
    Outputs:
        pypty_params- updatedd dictionary
    """
    probe=pypty_params["probe"]
    data_path=pypty_params["data_path"]
    data_pad=pypty_params.get("data_pad", 0)
    upsample_pattern=pypty_params["upsample_pattern"]
    data_is_numpy_and_flip_ky=pypty_params.get("data_is_numpy_and_flip_ky", False)
    if data_path[-3:]==".h5":
        h5file=h5py.File(data_path, "r")
        h5data=h5file["data"]
    elif data_path[-4:]==".npy":
        h5data=np.load(data_path)
        if len(h5data.shape)==4:
            scan_size=[h5data.shape[0], h5data.shape[1]]
            h5data=h5file.reshape(h5data.shape[0]* h5data.shape[1], h5data.shape[2],h5data.shape[3])
        if data_is_numpy_and_flip_ky:
            h5data=h5data[:,::-1, :]
    pacbed=pypty_params.get("mean_pattern", np.sum(h5data, 0))
    beam_fft=np.sum(np.abs(np.fft.fftshift(np.fft.fft2(probe, axes=(0,1)), axes=(0,1)))**2, -1)
    if data_pad!=0:
        beam_fft=beam_fft[data_pad:-data_pad,data_pad:-data_pad]
    if upsample_pattern!=1:
        beam_fft=downsample_something(beam_fft, upsample_pattern, np)
    if align_type=="com":
        x=np.arange(pacbed.shape[0])
        x=x-np.mean(x)
        x,y=np.meshgrid(x,x)
        comxpac=np.average(x, weights=pacbed)
        comypac=np.average(y, weights=pacbed)
        comxbeam=np.average(x, weights=beam_fft)
        comybeam=np.average(y, weights=beam_fft)
        shift_x=upsample_pattern*(comxpac-comxbeam)
        shift_y=upsample_pattern*(comypac-comybeam)
    else:
        beam_fft=np.fft.fft2(beam_fft)
        pacbed=np.fft.fft2(pacbed)
        
        cross=np.conjugate(beam_fft)*pacbed
        cross=np.angle(cross)
        cross=np.exp(1j*cross)
        cross=np.real(np.fft.fftshift(np.fft.ifft2(cross)))
        indy, indx=np.unravel_index(cross.argmax(), cross.shape)
        peak_center=cross[indy, indx]
        test_peak_left=cross[indy, indx-1]
        test_peak_right=cross[indy, indx+1]
        test_peak_top=cross[indy-1, indx]
        test_peak_bottom=cross[indy+1, indx]
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
                
        shift_x=upsample_pattern*(indx-shift_x-cross.shape[1]//2)
        shift_y=upsample_pattern*(indy-shift_y-cross.shape[1]//2)
    print("\nShifting beam by %.2e px along x and by %.2e px along y."%(shift_x, shift_y))
    kx,ky=np.meshgrid(np.fft.fftshift(np.fft.fftfreq(probe.shape[1])), np.fft.fftshift(np.fft.fftfreq(probe.shape[0])))
    kernel=np.exp(2j*np.pi*(kx*shift_x+ky*shift_y))[:,:,None]
    probe*=kernel
    pypty_params["probe"]=probe
    return pypty_params


def get_approx_beam_tilt(pypty_params, power=3, make_binary=False, percentile_filter_value=None,percentile_filter_size=10):
    """
    This function estimates the drift (scan-postion dependent tilt of the diffraction pattern). 
    Inputs:
        pypty_params- dictionary with callibrated pypty parameters.
        power- power of the polynominal fit, Possible values are: integers, np.inf or "inf"
        make_binary- boolean flag, default is False. If float larger than 0, the patterns will be made binary based on a treshold mean_value*make_binary
        percentile_filter_value- default is None, if not None a percentile filter with a given value will be applied to the resulting tilt maps to make them smooth.
        percentile_filter_size- intereger, default 10. Size of the percentile filter.
    Outputs:
        pypty_params- updatedd dictionary
    """
    dataset_h5=pypty_params.get("data_path", "")
    pixel_size_x_A=pypty_params.get("pixel_size_x_A", 1)
    rez_pixel_size_A=pypty_params.get("rez_pixel_size_A", 1)
    acc_voltage=pypty_params.get("acc_voltage", 60)
    scan_size=pypty_params.get("scan_size", None)
    data_is_numpy_and_flip_ky=pypty_params.get("data_is_numpy_and_flip_ky", False)
    if dataset_h5[-3:]==".h5":
        dataset_h5=h5py.File(dataset_h5,  "r")["data"]
    elif dataset_h5[-4:]==".npy":
        dataset_h5=np.load(dataset_h5)
        if len(dataset_h5.shape)==4:
            scan_size=[dataset_h5.shape[0],dataset_h5.shape[1]]
            dataset_h5=dataset_h5.reshape(dataset_h5.shape[0]*dataset_h5.shape[1],dataset_h5.shape[2], dataset_h5.shape[3] )
        if data_is_numpy_and_flip_ky:
            dataset_h5=dataset_h5[:,::-1, :]
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

