""""
This file describes the reconstruction procedure for the scans we acquiered on 30.01.2025 and for the majority of scans acquired on 31.01.2025 (expect 51-64 that require upsampling)
"""

import cupy as cp
import numpy as np
import pypty
print("PyPty version: ", pypty.__version__)


## files and metadata""
path="/home/anton/data/300125/"
path_output="/home/anton/data/300125/"
name="scan_apo_52"
output_folder=path_output+"52_m3/"
path_h5=path+ name+".h5"
path_json=""
path_raw=path+name+".npy"
vac_pattern=np.load("/home/anton/data/300125/vac_probe_6p4s.npy")

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
    'rez_pixel_size_mrad': 0.0556,
    'conv_semiangle_mrad': 6,
    'scan_size': [200,200],
    'scan_step_A': 3530/199,
    'fov_nm': 353,
    'aberrations': [-1.3e4,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # only initial guess
    'acc_voltage': 200, ## kV
    'upsample_pattern': 1,
    'data_pad': 112,
    'PLRotation_deg': 0.0, # deg # only initial guess
    'bright_threshold': 0.2,
    'total_thickness': 6, ## Angstrom
    'num_slices': 1,
    'print_flag': 1,
    'plot': 1,
    'save_preprocessing_files': 1
    }
    
pypty_params={
    'update_obj':  1,
    'update_probe':  1,
    'update_probe_pos': "lambda x: x>=10",
    'fast_axis_reg_weight_positions': 1e-1,
    
    
    'hist_length': 5,
    'update_step_bfgs': lambda x: 2e-6 if x==0 else 0.5,
    'atv_weight': 1e-6,
    'compute_batch': 50,

    'min_step': 1e-30,
    'max_count': 20,
    'optimism': 3,
    'reduce_factor': 0.1,

    'wolfe_c1_constant': 0.2,
    'wolfe_c2_constant': 0.9999,

    'dynamically_resize_yx_object': 1,
    'extra_space_on_side_px': 100,
    'lazy_clean': False,
    'update_extra_cut': 0.05,
    
    'save_inter_checkpoints': True,
    'save_checkpoints_every_epoch': False,
    'algorithm':  'lsq_sqrt',
    'use_full_FOV': True,

    'epoch_max': 100,

    'load_one_by_one': False,
    'preload_to_cpu': False,
    'smart_memory': False,
    'force_pad': False,
    'default_dtype': 'double',
    'damping_cutoff_multislice': 0.66
    }
    
### append experimental params to preset
pypty_params=pypty.initialize.append_exp_params(experimental_params, pypty_params)
pypty_params["aperture"]=vac_pattern

pypty_params=pypty.tcbf.run_tcbf_alignment(
        pypty_params,
        binning_for_fit=np.tile([5], 30),
        cross_corr_type="abs",
        refine_box_dim=5,
        upsample=10,
        cancel_large_shifts=0.9,
        reference_type="bf",
        tol_ctf=1e-8,
        optimize_angle=True
        )

if pypty_params["aberrations"][0]>0:
    pypty_params=pypty.initialize.rotate_scan_grid(pypty_params, angle_deg=-180)
    pypty_params=pypty.initialize.conjugate_beam(pypty_params)
    
ups=3
tcbf_image,tcbf_px_size = pypty.tcbf.upsampled_tcbf(pypty_params, upsample=ups,
                                    pad=10, default_float=32,round_shifts=True)
                                    
sc=pypty.initialize.get_grid_for_upsampled_image(pypty_params, tcbf_image,tcbf_px_size, ups*10, ups*10)
laplace=((tcbf_image/np.mean(tcbf_image)) -1)/np.abs(pypty_params["aberrations"][0])
tcbf_image_phase=pypty.dpc.iterative_poisson_solver(laplace=laplace,
            px_size=tcbf_px_size,print_flag=1, hpass=1e-2, lpass=0,
            step_size=1e-5, num_iterations=100,
            beta=0.1,use_backtracking=1, pad_width=50, xp=cp)
            
tcbf_image_phase=(tcbf_image_phase-np.min(tcbf_image_phase))
tcbf_image_phase=(tcbf_image_phase/np.max(tcbf_image_phase))-0.5

pypty_params=pypty.initialize.get_ptycho_obj_from_scan(pypty_params,
                                        array_phase=tcbf_image_phase,array_abs=None,
                                        scale_phase=0.75, scale_abs=1, cutoff=10,
                                        scan_array_A=sc, fill_value_type=0)


pypty_params=pypty.initialize.get_focussed_probe_from_vacscan(pypty_params, vac_pattern)
pypty_params=pypty.initialize.tiltbeamtodata(pypty_params, align_type="com")

pypty_params["print_flag"]=3
pypty.iterative_ptychography.run_ptychography(pypty_params)
