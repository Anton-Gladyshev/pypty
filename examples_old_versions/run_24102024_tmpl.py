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
json_stem = numpy_stem[:i] + 'tcBF_full'
path_json = path_numpy.with_stem(json_stem).with_suffix('.json')
this = Path(__file__)
recreate_h5 = (False if args.skip_h5_generation else
               not path_h5.exists()
               or this.stat().st_mtime > path_h5.stat().st_mtime
               or path_numpy.stat().st_mtime > path_h5.stat().st_mtime)
path_numpy, path_h5, haadf_path, path_json = map(
    str, (path_numpy, path_h5, haadf_path, path_json))
output_folder = './'

with open(path_json, 'r') as f:
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
pypty_params=create_probe_marker_chunks(pypty_params,50)

## just to be safe: remove all the junk
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()
cp.fft.config.clear_plan_cache()

run_loop(pypty_params)
