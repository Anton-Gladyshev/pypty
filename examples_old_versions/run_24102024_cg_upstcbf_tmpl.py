import numpy as np
import cupy as cp
import sys
from pypty import *

import argparse
from pathlib import Path
import numpy as np
import cupy as cp
from pypty import *

p = argparse.ArgumentParser()
p.add_argument(
    '--scans', type=Path, metavar='FILE', required=True,
    help='the file containing the scans to reconstruct')
p.add_argument(
    '--gpu', type=int, metavar='GPU_ID', default=0,
    help='the ID of the GPU on which to run the reconstruction')
p.add_argument(
    '--skip-h5-generation', action='store_true',
    help='Whether to skip generating the .h5 file if we know it is up-to-date')
args = p.parse_args()

cp.cuda.Device(args.gpu).use()

path_numpy:Path = args.scans
path_h5:Path = path_numpy.with_suffix('.h5')
numpy_stem = path_numpy.stem
i = numpy_stem.find('4DSTEM')
haadf_stem = numpy_stem[:i] + 'HAADF'
haadf_path = path_numpy.with_stem(haadf_stem)
tcBF_stem = numpy_stem[:i] + 'tcBF_full'
path_json_full_image = path_numpy.with_stem(tcBF_stem).with_suffix('.json')
path_tcbf_full_image = path_numpy.with_stem(tcBF_stem).with_suffix('.npy')
this = Path(__file__)
recreate_h5 = (False if args.skip_h5_generation else
               not path_h5.exists()
               or this.stat().st_mtime > path_h5.stat().st_mtime
               or path_numpy.stat().st_mtime > path_h5.stat().st_mtime)
path_numpy, path_h5, haadf_path, path_json_full_image = map(
    str, (path_numpy, path_h5, haadf_path, path_json_full_image))
output_folder = './'

with open(path_json_full_image, 'r') as f:
    jsondata_tcbf = json.load(f)
    jsondata_tcbf = jsondata_tcbf[
        "metadata"][
            "nion.tilt_corrected_bright_bright_field_full_res.parameters"][
                "TCBF fit coefficients"]
    PL_rot_tcbf = jsondata_tcbf["rotation"]
    tcbf_aberrations=[-1e10*jsondata_tcbf["defocus"],
                      -1e10*jsondata_tcbf["astig a"],
                      -1e10*jsondata_tcbf["astig b"]]

if recreate_h5:
    create_h5_file_from_numpy(
        path_numpy, path_h5, swap_axes=False,
        flip_ky=0,flip_kx=0, comcalc_len=200*200,
        comx=None, comy=None, bin=1, crop_left=None,
        crop_right=None, crop_top=None, crop_bottom=None, normalize=True)

if this.stat().st_mtime < Path('params.py').stat().st_mtime:
    print("*** File params.py has changed: regenerate run.py ***")
    sys.exit(1)

{{ define_params }}

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
