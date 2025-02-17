import argparse
from pathlib import Path
import numpy as np
import cupy as cp
import sys
import os
if 'WING_PRO_DEBUG' in os.environ:
    import wingdbstub
from pypty import *

p = argparse.ArgumentParser()
p.add_argument(
    '--scans', type=Path, metavar='FILE',
    help='the file containing the scans to reconstruct')
p.add_argument(
    '--continue', type=Path, metavar='DIRECTORY', dest='path_prev',
    help='the directory containing the reconstruction to continue with')
args = p.parse_args()

if args.scans is None:
    # Keep Anton's code if the script is called without --scan
    path="/testpool/ops/antongladyshev/pypty/bruker/01082024/"
    name="20240801_cryo_Apo_pos10_1060nm_200_dose_20eA2_4DSTEM_unwarped"
    path_json=""
    path_numpy=path+name+".npy"
    path_h5=path+name+"_2.h5"
    output_folder=path+"/10_48/"

    path_prev="/testpool/ops/antongladyshev/pypty/bruker/01082024/10_48/"
else:
    # Otherwise, do it Luc's way
    path_numpy:Path=args.scans
    path_h5=path_numpy.with_suffix('.h5')
    output_folder=Path.cwd()
    path_numpy, path_h5, path_prev, output_folder = map(
        str, (path_numpy, path_h5, args.path_prev, output_folder))
    path_json=""



create_h5_file_from_numpy(path_numpy, path_h5, swap_axes=False, flip_ky=0,flip_kx=0, comcalc_len=200*200,  comx=None, comy=None, bin=1, crop_left=None, crop_right=None, crop_top=None, crop_bottom=None)

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
    'PLRotation_deg': 0,
    'bright_threshold': 0.2,
    'total_thickness': 500, ## Angstrom
    'num_slices': 1,
    'sequence': None,
    'plot': False,
    'print_flag': 1,
    'data_pad': None,
    }

pypty_params={'obj_step': 2e3,
              'probe_step': 6.4e-9,
              'tilts_step': 8e-8,
              'save_dict': True,
              'adaptive_steps': True,
              'save_inter_checkpoints': True,
              'save_checkpoints_every_epoch': 5,
              'update_batch': 'full',
              'algorithm': "lsq_sqrt",
              'use_full_FOV': False,
              'optimizer': "gd",
              'max_count': None,
              'min_step': 1e-20,
              'propmethod': 'multislice',
              'allow_subPixel_shift': True,
              'epoch_max': 10,
              'optimism': 2,
              'reduce_factor': 1/2,
              'smart_memory': True,
              'wolfe_c1_constants':  np.array([0.9]),
              'damping_cutoff_multislice': 0.65,
              'phase_only_obj': True,
              'probe_constraint_mask': 0.5, #0.375,
              'probe_reg_constraint_weight': '1e-6', #lambda x: '1e-6' if x<10 else 0,
              }

pypty_params=append_exp_params(experimental_params, pypty_params)



pypty_params=create_probe_marker_chunks(pypty_params,25)


o=np.load(os.path.join(path_prev, "checkpoint_obj_epoch_10.npy"))
p=np.load(os.path.join(path_prev, "checkpoint_probe_epoch_10.npy"))*0.125
t=np.load(os.path.join(path_prev, "checkpoint_tilts_epoch_10.npy"))

pypty_params["probe"]=p
pypty_params["tilts"]=t
pypty_params["obj"]=o
pypty_params["tilt_mode"]=1

run_loop(pypty_params)
