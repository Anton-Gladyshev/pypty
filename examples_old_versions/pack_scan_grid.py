import argparse
from pathlib import Path
import numpy as np
import pickle
import re
import h5py

# command-line arguments
p = argparse.ArgumentParser(
    description=('Recursively find files cg.py in a directory and pack them all '
                 'in a HDF5 file.\n\nScan positions are in an arbitrary unit.'))
p.add_argument('--root', type=Path, metavar='FOLDER', required=True,
               help='the root of the recursive search')
p.add_argument('-o', '--output', type=Path, metavar='FILENAME',
               help='the name of the HDF5 file (absolute or relative to root)')
cmdargs = p.parse_args()
if not cmdargs.output.is_absolute():
    cmdargs.output = cmdargs.root / cmdargs.output

# Recursive search
with h5py.File(cmdargs.output, 'w') as top:
    for cg_fn in cmdargs.root.rglob("cg.npy"):
        params_fn = cg_fn.parent / 'params.pkl'
        with open(params_fn, 'rb') as f:
            params = pickle.load(f)
        cg = np.load(cg_fn)
        scan_step = params['scan_step_A']
        # Save to HDF5
        data_fn = Path(params['data_path'])
        if (m := re.search(r'(?:nr|pos)_(\d+)', data_fn.stem, re.I)) is None:
            print(f'Dataset ID could not be found in:\n\t{data_fn}\nSkipping...')
            continue
        idx = int(m[1])
        posg = top.create_group(f'position_{idx:02}')
        posg.attrs['idx'] = idx
        posg.attrs['rotation'] = params['PLRotation_deg']
        posg.attrs['rotation_unit'] = 'deg'
        posg.attrs['scan_step'] = scan_step
        posg.attrs['scan_step_unit'] = 'Å'
        ds = posg.create_dataset('scan_positions', data=cg)
