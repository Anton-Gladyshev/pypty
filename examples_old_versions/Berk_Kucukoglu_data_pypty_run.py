import numpy as np
import cupy as cp
import sys
path_scipt="/testpool/ops/antongladyshev/pypty"
sys.path.append(path_scipt)
from pypty_gpu_full import *

##### Hey! If you have enough memory, you may try to set a 'load_one_by_one' argument in the pypty_params to False. This will load the full dataset in the memory and everything should run faster (basically lazy loading is disabled).


for ind in range(17,21,1): ### add more datasets here
    path="/testpool/ops/antongladyshev/pypty/berk_apo/"
    name="pos_"+str(ind)+"_dataset"
    path_json=""
    path_raw=path+name+".h5" ### this are the datasets from Berk
    path_h5=path+name+"_proc.h5"
    output_folder=path+str(ind)+"_13/"
    create_h5_file_from_numpy(path_raw, path_h5, swap_axes=False, 
                          flip_ky=0,flip_kx=0, comcalc_len=128*128,  
                          comx=None, comy=None, bin=1, crop_left=None, 
                          crop_right=None, crop_top=None, crop_bottom=None)  ### i changed the create_h5_file_from_numpy in order to accept the 4d h5 files and convert them into 3d datasets. So now the code will create some extra data, but this may be changed in the future

    experimental_params={
    'output_folder': output_folder,
    'path_json': path_json,
    'data_path': path_h5,
    'rez_pixel_size_A': 0.0025481981054367433,
    'conv_semiangle_mrad': 4,
    'scan_size': [128,128],
    'scan_step_A': 20,
    'fov_nm': 254,
    'acc_voltage': 300,
    'upsample_pattern': 1,
    'defocus': 0,
    'PLRotation_deg': 89, # deg
    'bright_threshold': 0.2,
    'total_thickness': 500, ## Angstrom
    'num_slices': 1,
    'sequence': None,
    'plot': True,
    'print_flag': 3,
    'data_pad': None,
    } 
    

    pypty_params={
              'obj_step': [1e0, lambda x: x<=5 or x%2==0],
              'probe_step':[1e-10, lambda x: (x>=1)*(x<=5 or x%2 == 1)],
              'update_batch': lambda x: 'full' if x<10 else 50,
              'tilts_step': [1e-8, lambda x:  (x>=1)*(x<=5 or x%2 == 0)],
              'probe_pos_step': [1e-8, lambda x: (x>=5)*(x%2 ==0)],          
              'beam_current': np.ones((128*128), dtype=np.float64),
              'beam_current_step': [1e-1, lambda x:  (x>=1)*(x<=5 or x%2 == 0)],
              'save_dict': True,
              'adaptive_steps': True,
              'save_inter_checkpoints': 1,
              'save_checkpoints_every_epoch': False,
              'algorithm':  'lsq_sqrt',
              'use_full_FOV': False,
              'optimizer': "gd",
              'max_count': None,
              'min_step': 1e-20,
              'propmethod': 'multislice',
              'allow_subPixel_shift': True,
              'epoch_max': 80,
              'coupled_update': False,
              'allow_subPixel_shift': False, ## only at the beginning
              'optimism': 15,
              'reduce_factor': 1/15,
              'dynamically_resize_yx_object': True,
              'lazy_clean': False,
              'smart_memory': False,
              'wolfe_c1_constants': np.array([0.5]),  
              'damping_cutoff_multislice': 0.63,
              'smooth_rolloff': 0.1,
              'apply_gaussian_filter_amplitude': 30,  # the absorption will be blurred!
              'n_parallel_meas': 100, ### compute batch
              }

    pypty_params['print_flag']=0
    pypty_params=append_exp_params(experimental_params, pypty_params)
    pypty_params=run_tcbf_alignment(
    pypty_params,
    aberrations=[-2e4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    binning_for_fit=np.tile([5], 10),
    optimize_angle= False,
    compensate_lowfreq_drift=2.5,
    aperture="none",
    plot_inter_image=False,
    plot_CTF_shifts=False,
    save_inter_imags=False,
    
    cross_corr_type="abs",
    refine_box_dim=3,
    upsample=10,
    cancel_large_shifts=0.9,
    scan_pad=None,

    reference_type="bf",
    binning_cross_corr=1,
    f_scale_lsq=1,
    x_scale_lsq=1,
    loss_lsq="linear",
    tol_ctf=1e-8,
    )
    pypty_params['print_flag']=3 ### set it to 0 if you don't need any printing!
    pypty_params=get_approx_beam_tilt(pypty_params, power=3, make_binary=False)
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.fft.config.clear_plan_cache()
    run_loop(pypty_params)

