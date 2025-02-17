import numpy as np
import cupy as cp
import sys
from pypty import *
import scipy

path="/testpool/ops/antongladyshev/pypty/14032024/"

for dataind in [1,4,5,6,7,8]:
    name="20240314_hemo_pos"+str(dataind)+"_unwarped"

    path_json=""
    path_numpy=path+name+".npy"
    path_h5=path+name+".h5"
    output_folder=path+"/up"+str(dataind)+"_2/"
    create_h5_file_from_numpy(path_numpy, path_h5,
                          swap_axes=False, flip_y=False,flip_x=False,
                          comcalc_len=1000,
                          comx=None, comy=None, bin=1,
                          crop_left=None, crop_right=None, crop_top=None, crop_bottom=None)
    experimental_params={
    'output_folder': output_folder,
    'path_json': path_json,
    'data_path': path_h5,
    'rez_pixel_size_A': 0.002192762548990178,
    'conv_semiangle_mrad': 4,
    'scan_size': [200,200],
    'scan_step_A': 8320/199,
    'fov_nm': 832,
    'acc_voltage': 200, ## kV
    'upsample_pattern': 2,
    'defocus': 0, ## Angstrom
    'PLRotation_deg': 91, # deg
    'bright_threshold': 0.4,
    'total_thickness': 1500*10, ## Angstrom
    'num_slices': 1,
    'sequence': None,
    'plot': False,
    'print_flag': 1,
    'data_pad': None,
    }
    pypty_params={'obj_step': 1e0,
              'probe_step': 1e-10,
              'probe_pos_step': 0,#[1e4, lambda x:x>=1],
              'tilts_step': 1e-9,
              'save_dict': True,
              'save_inter_checkpoints': True,
              'save_checkpoints_every_epoch': 10,
              'update_batch': 'full',
              'randomize': True,
              'algorithm': "lsq_sqrt",
               'use_full_FOV': False,
              'max_count': None,
              'min_step': 1e-20,
              'propmethod': 'multislice',
              'phase_only_obj':True,  #lambda x: x<4,
              'epoch_max': 40,
              'optimism': 2,
              'reduce_factor': 1/2,
              'wolfe_c1_constants': np.array([1e-1, 0.5, 0.5, 0.1, 0.1]),
              'damping_cutoff_multislice': 0.65,
              'mixed_variance_weight': 0,
              'mixed_variance_sigma': 0.03,
              'smart_memory': True
              }
    pypty_params=append_exp_paramas(experimental_params, pypty_params)
    pypty_params['print_flag']=1

    if True:
        im, p, pypty_params=get_tcbf_image_probe(
        pypty_params,
        aberrations=[-2.8e4,0,0,0,0,0,0,0],
        binning_for_fit=np.tile([4], 15), ### repeat the fitting 40 times at binning of 4
        tol_ctf=1e-10,
        get_image=False,
        plot_inter_image=False,
        plot_CTF_shifts=False,
        save_inter_imags=False,
        refine_box_dim=20,
        upsample=10,
        reference_type="bf")

    pypty_params['print_flag']=3
    pypty_params['extra_probe_defocus']=0
    pypty_params['probe']=None
    pypty_params=get_approx_beam_tilt(pypty_params,"inf", None)
    run_loop(pypty_params)
