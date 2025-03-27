import cupy as cp
import numpy as np
import pypty
print("PyPty version: ", pypty.__version__)


## files and metadata""
path="/home/anton/data/toma_susi/"
name="data136_200_unnorm"
path_h5=path+ name+".h5"
path_json=""
path_raw=path+name+".npy"
output_folder="/home/anton/data/toma_susi/11/"

## create an h5 file
pypty.initialize.create_pypty_data(path_raw, path_h5, swap_axes=False,
                    flip_ky=0,flip_kx=0, comcalc_len=200*200,
                    comx=0, comy=0, bin=1, crop_left=None,
                    crop_right=None, crop_top=None, crop_bottom=None, normalize=False, exist_ok=1)


## append exp params to a preset
experimental_params={
    'output_folder': output_folder,
    'path_json': path_json,
    'data_path': path_h5,
    'rez_pixel_size_A': None,
    'conv_semiangle_mrad': 33.8,
    'scan_size': [256,256],
    'scan_step_A': 0.18,
    'fov_nm': 0.018*255,
    'acc_voltage': 60, ## kV
    'upsample_pattern': 1,
    'aberrations': [200,0,0],## Angstrom # 200A
    'PLRotation_deg': 4.65, # deg
    'bright_threshold': 0.2,
    'total_thickness': 6, ## Angstrom
    'num_slices': 1,
    'sequence': None,
    'plot': True,
    'print_flag': 3,
    'data_pad': 116
    }

pypty_params={
    'update_obj':  1,
    'update_probe':  1,
    'update_probe_pos': "lambda x: x>=10",
    'fast_axis_reg_weight_positions': 1e-1,

    'hist_length': 30,
    'hist_length': 30,
    'update_step_bfgs': "lambda x: 8.100e-09 if x==0 else 1",
    
    'defocus_spread_modes' np.linspace(-50,50,5), ## creates 5 probe modes by defocussing
    
    'compute_batch': 30,
    'min_step': 1e-30,
    'max_count': 20,
    'optimism': 3,
    'reduce_factor': 1/10,
    'wolfe_c1_constant': 1e-5,
    'wolfe_c2_constant': 0.9999,
    'epoch_max': 1000,

    'dynamically_resize_yx_object': 1,
    'extra_space_on_side_px': 100,
    
    'damping_cutoff_multislice': 0.66,
    'lazy_clean': False,
    'update_extra_cut': 0.05,
    
        
    'save_inter_checkpoints': True,
    'save_checkpoints_every_epoch': False,
    'algorithm':  'lsq_sqrt',
    'use_full_FOV': True,
    
    
    'load_one_by_one': False,
    'preload_to_cpu': False,
    'smart_memory': False,
    'force_pad': False,
    'default_dtype': 'double'}


### append experimental params to preset
pypty_params=pypty.initialize.append_exp_params(experimental_params, pypty_params)


pypty.iterative.run(pypty_params)
