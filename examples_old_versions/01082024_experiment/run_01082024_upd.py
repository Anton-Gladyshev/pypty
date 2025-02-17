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
    '--scans', type=Path, metavar='FILE', dest='scans',
    help='the file containing the scans to reconstruct')
args = p.parse_args()

if args.scans is None:
    # Keep Anton's code if the script is called without --scan
    path="/testpool/ops/antongladyshev/pypty/bruker/01082024/"
    name="20240801_cryo_Apo_pos10_1060nm_200_dose_20eA2_4DSTEM_unwarped"
    path_json=""
    path_numpy=path+name+".npy"
    path_h5=path+name+"_2.h5"
    output_folder=path+"/10_48/"
    haadf_path="/Users/anton/Desktop/ptychography_datasets/01082024/20240801_cryo_Apo_pos10_1060nm_200_dose_20eA2_HAADF.npy"
else:
    # Otherwise, do it Luc's way
    path_numpy:Path=args.scans
    path_h5=path_numpy.with_suffix('.h5')
    numpy_stem = path_numpy.stem
    i = numpy_stem.find('4DSTEM')
    haadf_stem = numpy_stem[:i] + 'HAADF'
    haadf_path = path_numpy.with_stem(haadf_stem)
    output_folder=Path.cwd()
    path_numpy, path_h5, haadf_path, output_folder = map(
        str, (path_numpy, path_h5, haadf_path, output_folder))
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

pypty_params={'obj_step': 1e0,
              'probe_step':[1e-10, lambda x:x>=1],
              'tilts_step': [1e-8, lambda x:x>=1],
             # 'probe_pos_step': [1e3, lambda x: x>=10],
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
              'epoch_max': 20,
              'optimism': 2,
              'reduce_factor': 1/2,
              'smart_memory': True,
              'wolfe_c1_constants': np.array([0.5]),
              'damping_cutoff_multislice': 0.65,
              'phase_only_obj': True,
              'probe_constraint_mask': 0.5, #0.375,
              'probe_reg_constraint_weight': '1e-6', #lambda x: '1e-6' if x<10 else 0,
              }

pypty_params=append_exp_params(experimental_params, pypty_params)


##### Until here was the usual stuff, now we will load haadf to track the gold NP and pick one


haa_ini=np.load(haadf_path)
haa=haa_ini[10:-10,10:-10] ## we want to find a particle with enogh scan points around it!
shy, shx=haa.shape
bf=4 ## bin the haadf
haa=cp.sum(haa.reshape(shy//bf,bf, shx//bf,bf), axis=(-3, -1))
indy, indx= np.unravel_index(haa.argmax(), haa.shape)
indy=indy*bf+10
indx=indx*bf+10 ## find maximum
left, top, right, bottom= indx-5, indy-5, indx+5, indy+5 ## this will be the ROI to tcBF!

pypty_params=run_tcbf_alignment(
    pypty_params,
    aberrations=[-1.2e4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    binning_for_fit=np.tile([5], 100),
    subscan_region=[left, top, right, bottom],

    compensate_lowfreq_drift=2.5,
    aperture="none",

    return_full_ctf=False,

    plot_inter_image=False,
    plot_CTF_shifts=False,
    save_inter_imags=False,
    cross_corr_type="abs",
    phase_cross_corr_formula=False,
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

pypty_params['print_flag']=3
pypty_params=get_approx_beam_tilt(pypty_params, power=3, make_binary=False) ## here we create the beam tilts
pypty_params=create_probe_marker_chunks(pypty_params,chop_size=25) ## here we create 64 beams, each for a small 25x25 grid


run_loop(pypty_params)
