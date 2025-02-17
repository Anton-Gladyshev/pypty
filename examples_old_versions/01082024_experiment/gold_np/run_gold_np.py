### discaimer: if you want to run the reconstruction on a subscan, enable it in a line 87 via "if 1:" or disable it via "if 0:". The subscan is much faster and reasonably large. Further, if you want to kick start ptychography object via tcBF image, leave the lines 94-111 as they are. If you want to start with vacuum, delete them.


import numpy as np
import cupy as cp
from pypty import *
from pypty.cmdline import parse_command_line_arguments

path_json, path_numpy, path_h5, output_folder, create_hdf5 = \
    parse_command_line_arguments(
        default_input_folder="/testpool/ops/antongladyshev/pypty/bruker/01082024/",
        default_name="20240801_AuNPs_pos1_1060nm_200_dose_20eA2_4DSTEM_unwarped",
        default_output_subfolder="au20_1")

### basic preparation of the settings:
if create_hdf5:
    create_h5_file_from_numpy(path_numpy, path_h5, swap_axes=False, flip_ky=0,flip_kx=0, comcalc_len=10,  comx=0, comy=0, bin=1, crop_left=None, crop_right=None, crop_top=None, crop_bottom=None)
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
    'total_thickness': 200, ## Angstrom
    'num_slices': 20,
    'sequence': None,
    'plot': False,
    'print_flag': 1,
    'data_pad': None,
    }
pypty_params={'obj_step': 1e2,
              'probe_step':  [1e-10, lambda x:x>=1],
              'probe_pos_step': [1e2, lambda x:x>=40],
              'tilts_step':  [1e-7, lambda x:x>=1],
              'save_dict': True,
              'adaptive_steps': True,
              'save_inter_checkpoints': True,
              'save_checkpoints_every_epoch': False, ### please set to an int value "n" to save every n'th epoch
              'update_batch': 'full',
              'algorithm': "lsq_sqrt",
              'use_full_FOV': False,
              'optimizer': "gd",
              'max_count': None,
              'min_step': 1e-20,
              'propmethod': 'multislice',
              'allow_subPixel_shift': True,
              'epoch_max': 1000,
              'optimism': 2,
              'reduce_factor': 1/2,
              'smart_memory': True,
              'wolfe_c1_constants': np.array([0.9, 0.9, 0.5, 0.5, 0.1]),
              'damping_cutoff_multislice': 0.65,
              'truncate': 0,
              'estimate_aperture_based_on_binary': 2
              }
pypty_params=append_exp_params(experimental_params, pypty_params)
## now we will fit the aberrations:
pypty_params=run_tcbf_alignment(
    pypty_params,
    aberrations=[-1.2e4,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
    binning_for_fit=np.tile([5], 50), ### repeat the fitting 50 times at binning of 5
    tol_ctf=1e-20,
    plot_inter_image=False,
    plot_CTF_shifts=False,
    save_inter_imags=False,
    refine_box_dim=5,
    upsample=20,
    cancel_large_shifts=0.95,
    scan_pad=None,
    reference_type="bf",
    compensate_lowfreq_drift=1.2,
    aperture="none",
    return_full_ctf=False,
    )
## get the tilt map guess
pypty_params=get_approx_beam_tilt(pypty_params, power='inf', make_binary=False)


### now a super important thing: subscan. I did my reconstructions on a 50x50 grid. This is fast and reasonably large. You can do a full grid, but it may take some time
if 1:
    seq=[]
    for i in range(50):
        for j in range(50):
            seq.append(i*200+j)
    pypty_params["sequence"]=seq

## get an upsampled tcBF image and save it
tcbf_image, tcbf_px_size=upsampled_tcbf(pypty_params,upsample=5, pad=5,bin_fac=3,
                    default_float=64,round_shifts=False,compensate_lowfreq_drift=True,
                    xp=cp,save=1,max_parallel_fft=10) ## if you want to do it faster, set the max_parallel_fft to higher value, e.g. 200 or 1000. It may however saturate the GPU!

## get a Phase estimate using TIE and tcBF image
laplace=tcbf_image[25:-25, 25:-25]
laplace=((laplace/np.mean(laplace))  -1)/np.abs(pypty_params["aberrations"][0])
tcbf_image_phase=iterative_poisson_solver(laplace=laplace,
                    px_size=tcbf_px_size,print_flag=1, hpass=1e-6, lpass=0,
                    step_size=1e-5, num_iterations=100,
                    beta=0.1,use_backtracking=1, pad_width=50)
tcbf_image_phase-=np.min(tcbf_image_phase)

### now this phase can be interpolated to get a guess for the object
scx, scy=np.meshgrid(np.arange(0, tcbf_image_phase.shape[1],1)*tcbf_px_size, np.arange(0, tcbf_image_phase.shape[0],1)*tcbf_px_size, indexing="xy") ### this is the upsampled scan grid of the tcBF image
sc=np.swapaxes(np.array([scy.flatten(), scx.flatten()]),0,1)
pypty_params, _=get_ptycho_obj_from_scan(pypty_params, num_slices="auto", array_phase=tcbf_image_phase,array_abs=None, scale_phase=1, scale_abs=1, cutoff=1, scan_array_A=sc)



run_loop(pypty_params)
