import numpy as np
import cupy as cp
import sys
from pypty import *


## What is needed: 4dstem data as npy file, full tcBF metadata in json format

indexxx=12
path="/testpool/ops/antongladyshev/pypty/bruker/24102024/"


name="20241025_Nr_%d_Apo_AuNP_Gr_HUB_0p3msec_5pA_300nm_2p5um"%indexxx
output_folder=path+"%d_m/"%indexxx
path_json=""
path_h5=path+name+"_4DSTEM.h5"
path_raw=path+name+"_4DSTEM.npy"
path_tcbf_full_json=path+name+"_tcBF_full.json"

with open(path_tcbf_full_json, 'r') as file:
    jsondata_tcbf = json.load(file)
jsondata_tcbf=jsondata_tcbf["metadata"]["nion.tilt_corrected_bright_bright_field_full_res.parameters"]["TCBF fit coefficients"]
PL_rot_tcbf=jsondata_tcbf["rotation"]
tcbf_aberrations=[-1e10*jsondata_tcbf["defocus"], -1e10*jsondata_tcbf["astig a"],-1e10*jsondata_tcbf["astig b"]]


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
    'num_slices': 1,
    'sequence': None,
    'plot': False,
    'print_flag': 3,
    'data_pad': None,
    }


pypty_params={
          'obj_step':  1e3,
          'probe_step': 1e-10,
          'update_batch':  "full",
          'randomize': False,
          'probe_pos_step': [1e0, lambda x: (x>=5)],
          'no_ambiguity': True,
          'obj_update_delta_beta': True,
          'save_dict': True,
          'adaptive_steps': True,
          'save_inter_checkpoints': True,
          'save_checkpoints_every_epoch': 10,
          'algorithm':  'lsq_sqrt',
          'obj_update_delta_beta': True,
          'use_full_FOV': False,
          'optimizer': "cg",
          'max_count': None,
          'min_step': 1e-20,
          'aberrations': tcbf_aberrations,
          'propmethod': 'multislice',
          'allow_subPixel_shift': True,
          'epoch_max': 50,
          'coupled_update': True,
          'optimism': 5,
          'reduce_factor': 1/10,
          'load_one_by_one': True,
          'preload_to_cpu': False,
          'smart_memory': lambda x: x%6==0,
          'default_dtype': 'double',
          'wolfe_c1_constants': np.array([0.1]),
          'damping_cutoff_multislice': 0.66,
          'smooth_rolloff': 0.05,
          'n_parallel_meas': 32,
          'dynamically_resize_yx_object': 20,
        'probe_constraint_mask': 0.6,
        'probe_reg_constraint_weight': np.inf, #lambda x: np.inf if x>=2 else 0,
          }

pypty_params=append_exp_params(experimental_params, pypty_params)


pypty_params["print_flag"]=3
tcbf_image, tcbf_px_size=upsampled_tcbf(pypty_params, upsample=10, pad=10,
                              compensate_lowfreq_drift=False,
                              default_float=64,round_shifts=True,xp=cp,save=0,bin_fac=1)
scx=np.arange(-100,2100,1)*tcbf_px_size
scy=np.arange(-100,2100,1)*tcbf_px_size
scx, scy=np.meshgrid(scx, scy, indexing="xy")

rot_ang=pypty_params["PLRotation_deg"]*np.pi/180
sc_prime_x,sc_prime_y=scx * np.cos(rot_ang) - scy * np.sin(rot_ang), scx * np.sin(rot_ang) + scy * np.cos(rot_ang)


ofy, ofx=get_offset(x_range=200, y_range=200,scan_step_A=15, detector_pixel_size_rezA=5.7176870952051917e-05/(12.4/np.sqrt(200*(200+2*511))), patternshape=[384,384], rot_angle_deg=-1*pypty_params["PLRotation_deg"])
sc_prime_x, sc_prime_y=sc_prime_x+ofx, sc_prime_y+ofy
sc=np.swapaxes(np.array([sc_prime_y.flatten(), sc_prime_x.flatten()]),0,1)


laplace=((tcbf_image/np.mean(tcbf_image))  -1)/np.abs(tcbf_aberrations[0])
tcbf_image_phase=iterative_poisson_solver(laplace=laplace,
            px_size=tcbf_px_size,print_flag=1, hpass=1e-2, lpass=0,
            step_size=1e-5, num_iterations=100,
            beta=0.1,use_backtracking=1, pad_width=50)
d=tcbf_image_phase.shape[0]


#tcbf_image_phase-=np.min(tcbf_image_phase)
#tcbf_image_phase-=0.5*np.max(tcbf_image_phase)

pypty_params["use_full_FOV"]=True
pypty_params, _=get_ptycho_obj_from_scan(pypty_params,
        array_phase=tcbf_image_phase,array_abs=None,
        scale_phase=60, scale_abs=1, cutoff=10, scan_array_A=sc, fill_value_type=0)
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()
cp.fft.config.clear_plan_cache()
run_loop(pypty_params)
