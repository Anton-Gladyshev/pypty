import numpy as np
import cupy as cp
import sys
from pypty import *
from pypty.cmdline import parse_command_line_arguments

path_json, path_numpy, path_h5, output_folder, create_hdf5 = \
    parse_command_line_arguments(
        default_input_folder="/testpool/ops/antongladyshev/pypty/bruker/01082024/",
        default_name="20240801_cryo_Apo_pos10_1060nm_200_dose_20eA2_4DSTEM_unwarped",
        default_output_subfolder="10_4")


if create_hdf5:
    create_h5_file_from_numpy(path_numpy, path_h5, swap_axes=False, flip_ky=1,flip_kx=1, comcalc_len=10,  comx=0, comy=0, bin=1, crop_left=None, crop_right=None, crop_top=None, crop_bottom=None)

experimental_params={
    'output_folder': output_folder,
    'path_json': path_json,
    'data_path': path_h5,
    'rez_pixel_size_A': 0.00199342049908198,
    'conv_semiangle_mrad': 4,
    'scan_size': [200,200],
    'scan_step_A': 53.26,
    'fov_nm': 1060,
    'acc_voltage': 200, ## kV
    'upsample_pattern': 1,
    'defocus': 0, ## Angstrom
    'PLRotation_deg': 0,#-97.7, # deg
    'bright_threshold': 0.4,
    'total_thickness': 500, ## Angstrom
    'num_slices': 2,
    'sequence': None,
    'plot': False,
    'print_flag': 3,
    'data_pad': None,
    }


pypty_params={'obj_step': 1e2,
              'probe_step':  [1e-10, lambda x:x>=1],
              'probe_pos_step': [1e2, lambda x:x>=40],
              'tilts_step':  [1e-7, lambda x:x>=1],
              'save_dict': True,
              'adaptive_steps': True,
              'save_inter_checkpoints': True,
              'save_checkpoints_every_epoch': 10,
              'update_batch': 'full',
              'algorithm': "lsq_sqrt",
              'use_full_FOV': False,
              'optimizer': "gd",
              'max_count': None,
              'min_step': 1e-20,
              'propmethod': 'better_multislice',
              'allow_subPixel_shift': True,
              'epoch_max': 1000,
              'optimism': 2,
              'reduce_factor': 1/2,
              'phase_only_obj': lambda x: x<10,
              'smart_memory': True,
              'wolfe_c1_constants': np.array([0.9, 0.9, 0.5, 0.5, 0.1]),
              'damping_cutoff_multislice': 0.65,
              }

pypty_params=append_exp_params(experimental_params, pypty_params)
pypty_params['print_flag']=1

pypty_params=run_tcbf_alignment(
    pypty_params,
    aberrations=[-1.2e4,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0],
    binning_for_fit=np.tile([5], 50), ### repeat the fitting 50 times at binning of 5
    tol_ctf=1e-20,
    plot_inter_image=False,
    plot_CTF_shifts=False,
    save_inter_imags=False,
    refine_box_dim=5,
    upsample=20,
    cancel_large_shifts=0.9,
    scan_pad=None,
    reference_type="bf",# "zero"
    compensate_lowfreq_drift=2,
    aperture="none",
    return_full_ctf=False
    )

pypty_params['print_flag']=3
pypty_params=get_approx_beam_tilt(pypty_params, power='inf', make_binary=False)
pypty_params['tilt_mode']=2

## if you need it
# tcbf_image=upsampled_tcbf(pypty_params,upsample=4, pad=5,bin_fac=3,
                   # default_float=64,round_shifts=1,compensate_lowfreq_drift=False,
                #    xp=np,save=0,max_parallel_fft=50)


run_loop(pypty_params)
