import numpy as np

from pypty import *

""" This script assumes the following data set is in the current directory

A link to the following file on either the A100 VM or LimmatPC can be made:
/media/data/original-data/hu-berlin/20240314_16_Hemo_cryo_200keV/
20240315_Hemo_200keV/20240315_hemo_pos10_865nm_200pos_d2200nm_4DSTEM.npy
"""
name = "hemocyanin_test_data"
path_numpy = f"{name}.npy"
path_json = ""
path_h5 = f"{name}.h5"
output_folder = "."

""" This crashes with an illegal memory access that makes no sense:
    when it happens, only about 20 GB are allocated on the card,
    far from the total of 80 GB available on the A100.
"""
#create_h5_file_from_numpy(path_numpy, path_h5,
                          #swap_axes=False, flip_y=False,flip_x=False,
                          #comcalc_len=1000,
                          #comx=None, comy=None, bin=1,
                          #crop_left=None, crop_right=None, crop_top=None,
                          #crop_bottom=None)

""" This works """
create_h5_file_from_numpy(path_numpy, path_h5,
                          swap_axes=False, flip_y=False,flip_x=False,
                          comcalc_len=1000,
                          comx=None, comy=None, bin=1,
                          crop_left=None, crop_right=32, crop_top=None,
                          crop_bottom=32)

""" This crashes with an illegal memory access """
#create_h5_file_from_numpy(path_numpy, path_h5,
                          #swap_axes=False, flip_y=False,flip_x=False,
                          #comcalc_len=1000,
                          #comx=None, comy=None, bin=1,
                          #crop_left=None, crop_right=16, crop_top=None,
                          #crop_bottom=16)

""" This crashes because deep down some arrays need to be square """
#create_h5_file_from_numpy(path_numpy, path_h5,
                          #swap_axes=False, flip_y=False,flip_x=False,
                          #comcalc_len=1000,
                          #comx=None, comy=None, bin=1,
                          #crop_left=None, crop_right=None, crop_top=None,
                          #crop_bottom=128)

experimental_params={
    'output_folder': output_folder,
    'path_json': path_json,
    'data_path': path_h5,
    'rez_pixel_size_A': 0.002192762548990178,
    'conv_semiangle_mrad': 4,
    'scan_size': [200,200],
    'scan_step_A': 43.25,
    'fov_nm': 865,
    'acc_voltage': 200, ## kV
    'defocus': 0, ## Angstrom
    'PLRotation_deg': -90, # deg
    'bright_threshold': 0.4,
    'total_thickness': 1000, ## Angstrom
    'num_slices': 1,
    'sequence': None,
    'plot': True,
    'print_flag': 1,
    'data_pad': None,
    }

pypty_params={'obj_step': 1e1,
              'probe_step': 1e-10,
              'probe_pos_step': [1e4, lambda x:x>=1],
              'save_dict': True,
              'adaptive_steps': True,
              'save_inter_checkpoints':True,
              'save_checkpoints_every_epoch':False,
              'update_batch': "full",
              'randomize': True,
              'algorithm': "lsq_sqrt",
              'use_full_FOV': False,
              'max_count': None,
              'min_step': 1e-20,
              'masked_fourier_obj_constraint_weight': 0,#1e-9,
              'propmethod': 'multislice',
              'phase_only_obj':True,  #lambda x: x<4, # True,
             #'window': [0.3, 0.5],   #get_window(1158, 0.2, 1)**5,
              'optimizepos1by1': True,
              'atv_q': 1,
              'atv_p': 2,
              'atv_weight': 0,
              'allow_subPixel_shift': True,
              'epoch_max':5, # << number of final optimisation cycles
              'optimism': 5,
              'reduce_factor': 1/3,
              'wolfe_c1_constants': np.array([0.5, 1e-1, 0.5, 0.5]),
              'damping_cutoff_multislice': 0.65
        }

pypty_params=append_exp_params(experimental_params, pypty_params)

# tcBF
# ----

im, p, pypty_params=get_tcbf_image_probe(
     pypty_params,
     aberrations=[2.2e4,0,0,0,0,0,0,0],
     binning_for_fit=np.tile([4], 2),
     tol_ctf=1e-10,
     get_image=False,
     plot_inter_image=False,
     plot_CTF_shifts=False,
     save_inter_imags=False,
     refine_box_dim=20,
     upsample=10,
     reference_type="bf"
     )

# DPC
# ---

pot, pypty_params, comx, comy=getdpcpot(pypty_params, hpass=0.011, lpass=0,
                                        save=False)

# Something that we know
# ----------------------

pypty_params['aberrations']=[2.24e+04, 1.15e+03, 3.19e+02, -2.79e+05, 4.37e+04,
                             1.22e+05, -1.38e+05, 3.98e+08]
pypty_params['extra_probe_defocus']=0

# WDD
# ---
o, probe = simple_wdd(pypty_params, eps_wiener=1e-3)

def asnumpy(a):
    if isinstance(a, np.ndarray):
        return a
    elif isinstance(a, cp.ndarray):
        return a.get()
    else:
        raise TypeError(f'Unsupported type: {a}')

fig, ax=plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(np.angle(asnumpy(o)/np.exp(2j)), cmap="gray")
ax[0].set_title("WDD Phase")
ax[0].axis("off")
ax[1].imshow(np.abs(asnumpy(o)), cmap="gray")
ax[1].set_title("WDD Abs")
ax[1].axis("off")
plt.show()

# Some final reconstruction parameters
# ------------------------------------

pos=pypty_params["positions"]
pos+=4*np.random.rand(pos.shape[0], pos.shape[1])-2
pypty_params["positions"]=pos
pypty_params['print_flag']=1
pypty_params['probe']=None

seq=[]
for i in range(40,50):
    for j in range(30,40):
        seq.append(200*i+j)

pypty_params['sequence']=seq

# Run
# ---

run_loop(pypty_params)

# Plot
# ----

o=np.load(pypty_params["output_folder"]+"/co.npy")[:,:,0,0]

plt.figure(figsize=(10,10))
a=np.angle(o)
plt.imshow(a, cmap="gray")#, vmax=1e-1, vmin=-1e-1)
#plt.title("recon")
plt.colorbar()
plt.axis("off")
#plt.savefig("/Users/anton/Desktop/ptychography_datasets/experiment_14032024/15032024/10_4/recon.png", dpi=200)
plt.show()

plt.imshow(np.abs(shift_fft2(o))**0.1)

p=np.load(pypty_params["output_folder"]+"/cp.npy")
plot_modes(p)

