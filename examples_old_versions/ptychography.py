import numpy as np
import sys
path_scipt="/testpool/ops/antongladyshev/pypty"
sys.path.append(path_scipt)
from pypty_gpu_full import *


main_path="/testpool/ops/antongladyshev/pypty/mos2_300b/"

pos_path=main_path+"positions.npy"
probe_path=main_path+"iniprobe.npy"
bright_field_path=main_path+"aperture.npy"


params={
        'data_path':  main_path+"data_mos2_300x300px_bin2.h5", ## path to the h5 file with a dataset ["data"]. The shape should be [N_measurements,N_pix_y, N_pix_y]
        'masks': None, # if the data is compressed, provide the masks as an array [N_masks, Probe_y, Probe_x]
        'load_one_by_one': True, ## loading setting for the diffraction patterns, set to True!
        'data_multiplier': 1, # rescaling of the data
        'data_pad': 0, # int, if larger than 0, than N pixels will be added to each side of the diffraction pattern
        'sequence': None, ## if sequence if None, all measurements will be selected as they are stored in the h5 file. You can provide either a specific sequence e.g. update everything in "squares", do "circles" etc. As an alrernative you can select only specific measurements with this argument.
        'use_full_FOV': True, ## only relevant if you select certain measurements. If True, the shae of object will correspond to the initial range of positions, if False, the object will contain only the selected positons

        'output_folder': main_path+"37/", # output folder
        'save_checkpoints_every_epoch': True, # if True, the checkpoints after every epoch will be saved, atenatively it can be an integer!
        'save_inter_checkpoints': True, # Save intermediate checkpoints?
        'print_flag': 3, ## 0 is no text output, 1 only the parameters, 2 and higher will give info about the linesearch

        'positions': np.load(pos_path),  # array with positions [[y_0,x_0],[y_1, x_1],....]
        'tilts': np.array([[0.0,0.0]]),  # array with beam tan of beam tilts [[tan(ay_0),tan(ax_0)],[tan(ay_1),tan(ax_1)],....] 
        'acc_voltage': 60, ## kV
        'recon_type': 'far_field', # 'far_field' or 'near_field'
        'Cs': 0, # Angastrom!!! only relevant for 'near_field'
        'alpha_near_field': 0, # flux-preserving constant.  only relevant for 'near_field'
        'defocus_array': np.array([0.0]),  # Angastrom!!! only relevant for 'near_field'
        'phase_plate_in_h5': None, # or string, path to the phase plate in h5 file
        'obj': np.exp(1j*(3e-2*np.random.rand(1, 1,1,1))), 
        'probe': np.load(probe_path),  # array shape [y,x,modes]

        'slice_distances': np.array([6]), # Angstrom, either one value [d] or [d1,d2,..d_N_sclies]
        'pixel_size_x_A': 0.06900297360463607, # Angstrom
        'pixel_size_y_A': 0.06900297360463607, # Angstrom
        'damping_cutoff_multislice': 0.63,
        'allow_subPixel_shift': True,
        'extra_probe_defocus': 0, # extra defocus in Angstrom
        'window': None, # or numpy array that would damp the tails of the probe in real-space!
        'propmethod': "multislice", # or "better_multislice" or 'yoshida'


        'coupled_update': False, # if True, all quantities will be updated with the same step
        'algorithm': 'lsq_sqrt', # "lsq", "ml", "lsq_sqrt", "lsq_ml", "lsq_compressed", "epie", "epie_compressed"
        'optimizer': "cg",       # "cg" for cojugate gradient. "gd" for simple gradient descent.
        'update_batch': 'full',  # or integer, number of measurements per one update
        'epoch_max': 1000, # integer, number of cycles
        'epoch_prev': 0,  # integer, from with index to start (useful for restarting a reconsturction)
        'randomize': False, # if True, measurements will be selected in random order
        'wolfe_c1_constants': np.array([0.5,0.5,0.5,0.5 ]), ## object, probe, positions, tilts #this should be strictly between 0 and 1. The larger the constant, the smaller the step size (and vice-versa). Better set it to 0.5 and don't touch this parameter.
        'max_count': None, # or integer, max number of loss evaluations during the linesearch
        'adaptive_steps': True, # if True, the initial update step at the next iteration will be initialized using the one computed during the previous iteration. If False, the step at the next iteration will be conpouted based on the curvature of the loss during the previous iteration.
        ### order of multiple measurements and updates
        'reduce_factor': 0.5, # Update step reduce factor for the linesearch
        'optimism': 2, # overshooting parameter for the linesearch
        'min_step': 1e-20,  ## lowest allowed update step
        'optimizepos1by1': False, # if True, positions will be updated one by one ignoring the update batch size
        'optimizetilts1by1': False, # if True, tilts will be updated one by one ignoring the update batch size
        
        'probe_step': [1.3e-5, lambda epoch: epoch>=5], #  Initial learning rate for the probe! Int or a list of [int, function] where int is the initial step and the function should return a binary value that indicates whether the probe should be updated or not
        'obj_step': 5e5, ## initial learning rate for the object. Alternbatively the same list as for the probe!
        'probe_pos_step': 3.9e-7, # initial learning rate for the positions. Alternbatively the same list as for the probe!
        'tilts_step': 0, # initial learning rate for the tilts. Alternbatively the same list as for the probe!


        ##### Constraints 
        'phase_only_obj': True, # constant object amplitude 
        'force_fix': True, ## fix phase wraps?
        'make_abs_prop_to_phase': False, # if True, absorption potential will be projected on the electrostatic potential
        'no_ambiguity': False, # if True, maximum of the objects amplitude will be set to one and the minimum of its phase will be shifted to zero
        'keep_probe_states_orthogonal': True, # if True, probe modes will be orthogonalized after every iteration
        'keep_obj_states_orthogonal': False, # if True, object modes will be orthogonalized after every iteration (i don't know if it is a useful option)
        'tune_only_probe_aberrations': False, # self explaining

        ## Note, every following parameter can be specified a 
        'loss_weight': 1, ## Rescaling weight applied to the loss function. Alternatively it can be 
        #'loss_weight': lambda epoch: epoch%2,

        'do_charge_flip': False, #charge flipping !!!!!or a binary function of epoch!!!!
        'cf_delta_phase': 0.05,
        'cf_delta_abs': 0.05,
        'cf_beta_phase': -0.99,
        'cf_beta_abs': -0.99,


        
        'atv_weight': 0, ## adaptive total variation w*([dO/dx]^p+[dO/dy]^p)^(q/p)! Note an alternative
        #'atv_weight': lambda epoch: 1e-5*epoch%10, 
        'atv_q': 1, ## power of sum 
        'atv_p': 1, ## power of the object grad

        'histogram_constraint': False, ## or integer, only relevant for a mixed object!!! Also can be specified as a function of epoch

        'beta_wedge': 0,        # David Muller constraint.  Also can be specified as a function of epoch

        'tilt_punish_reg': 1e10, # weight applied to to the tilt corrections outside of the allowed range.  Also can be specified as a function of epoch
        'max_tilt_deviation': 0.01, # allowed range for tilt correction
        'pos_punish_reg': 1e10, # weight applied to to the position corrections outside of the allowed range.  Also can be specified as a function of epoch
        'max_pos_deviation': 1.5, # allowed range for position correction

        'z_pediod': 1,  ## Autocorrelation-based constraint for producing same slices with given periodicity
        'zp_same_weight': 0, ## weight applied to produce same slices.  Also can be specified as a function of epoch
        'zp_different_weight': 0, # weight applied to enforce changes in the slices.  Also can be specified as a function of epoch
        
        'phase_norm_weight': 0, # l1 weight applied to the phase of the object (electrostatic potential).  Also can be specified as a function of epoch
        'abs_norm_weight': 0,    # l1 weight applied to the absorption potential.  Also can be specified as a function of epoch
        
        'probe_reg_constraint_weight': 1e-6 # float or np.inf, if np.inf the cpnstraint is realized by just setting to zero the dark field intensity.  Also can be specified as a function of epoch

        'aperture_mask': np.load(bright_field_path), # None OR array with shape [Probe_y, Probe_x] OR float describing the radius of the bright field. Radius must be a fraction of the largest frequency, i.e. if your aperute has a diameter of 30 pixel and a full diffraction pattern is 128x128, the fraction must be 30/128.
       
        
        }
run_loop(params)
