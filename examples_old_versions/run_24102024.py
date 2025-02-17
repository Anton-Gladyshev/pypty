import numpy as np
import cupy as cp
import sys
from pypty import *


## What is needed: 4dstem data as npy file, full tcBF metadata in json format


path="/testpool/ops/antongladyshev/pypty/bruker/24102024/"
name="20241025_Nr_8_Apo_AuNP_Gr_HUB_0p3msec_5pA_300nm_2p5um"
path_json=""
path_h5=path+name+"_4DSTEM.h5"
path_raw=path+name+"_4DSTEM.npy"
path_tcbf_full_json=path+name+"_tcBF_full.json"
output_folder=path+"8_1/"

## here i load the metadata from tcBF
with open(path_json_tcbf, 'r') as file:
    jsondata_tcbf = json.load(file)
jsondata_tcbf=jsondata_tcbf["metadata"]["nion.tilt_corrected_bright_bright_field_full_res.parameters"]["TCBF fit coefficients"]
PL_rot_tcbf=jsondata_tcbf["rotation"]
tcbf_aberrations=[-1*jsondata_tcbf["defocus"], -1*jsondata_tcbf["astig a"],-1*jsondata_tcbf["astig b"]]
create_h5_file_from_numpy(path_raw, path_h5, swap_axes=False,
                          flip_ky=0,flip_kx=0, comcalc_len=200*200,
                          comx=None, comy=None, bin=1, crop_left=None,
                          crop_right=None, crop_top=None, crop_bottom=None, normalize=True)
experimental_params={
    'output_folder': output_folder,
    'path_json': "",
    'data_path': path_h5,
    'rez_pixel_size_A': 5.7176870952051917e-05/(12.4/np.sqrt(200*(200+2*511))),
    'conv_semiangle_mrad': 4,
    'scan_size': [200,200],
    'scan_step_A': 15,
    'fov_nm': 1.5*199,
    'acc_voltage': 200, ## kV
    'defocus': 0, ## Angstrom
    'PLRotation_deg': PL_rot_tcbf, ### 63 # deg
    'bright_threshold': 0.3,
    'total_thickness': 400, ## Angstrom
    'num_slices': 4,
    'sequence': None,
    'plot': False,
    'print_flag': 3,
    'data_pad': None,
    }
    
pypty_params={
          'obj_step':   [1e0, 'lambda x: x%1==0'],
          'probe_step':   [1e-10, 'lambda x:  x>=1'],
          'update_batch': 'full',
          'tilts_step':   [1e-8, 'lambda x: x>=1'],
          'probe_pos_step': [1e2, 'lambda x: x>=1'],
          'sequence_to_reg_grid': 'lambda x: [x//200, x%200]',
          'postions_poly_grid': 5, ## positions will be a 5th order polynomial of the data index
          'tilts_poly_grid': 5, # tilts will be a 5th order polynomial of the data index
          'tilts': np.zeros((200*200, 6)),
          'tilt_mode': 2,
          'save_dict': True,
          'adaptive_steps': True,
          'save_inter_checkpoints': True,
          'save_checkpoints_every_epoch': 50,
          'algorithm':  'lsq_sqrt',
          'use_full_FOV': False,
          'optimizer': "gd",
          'max_count': None,
          'min_step': 1e-20,
          'propmethod': 'better_multislice',
          'allow_subPixel_shift': True,
          'epoch_max': 500,
          'coupled_update': False,
          'optimism': 2,
          'reduce_factor': 1/2,
          'load_one_by_one': True,
          'preload_to_cpu': False,
          'smart_memory': lambda x: x%6==0, ## after every 6th epoch, the pool with be cleaner to prevent memory fragmentation!
          'default_dtype': 'double',
          'wolfe_c1_constants': np.array([0.1]),
          'damping_cutoff_multislice': 0.65,
          'smooth_rolloff': 0.1,
          'n_parallel_meas': 8, ## you may try to set it to a larger number, depending on available memory
          'aberrations': tcbf_aberrations,
          'dynamically_resize_yx_object': True,
        'probe_constraint_mask': 0.52,
        'probe_reg_constraint_weight': np.inf,
          }
pypty_params=append_exp_params(experimental_params, pypty_params)
pypty_params=create_probe_marker_chunks(pypty_params,50)

## just to be safe: remove all the junk
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()
cp.fft.config.clear_plan_cache()

run_loop(pypty_params)
