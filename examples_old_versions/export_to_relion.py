from pathlib import Path
import argparse
import pickle
import re
import sys
import numpy as np
from PIL import Image

yes_values = {'yes', 'y', 'true', 't', '1'}
no_values = {'no', 'n', 'false', 'f', '0'}

def yes_no(value):
    if value.lower() in yes_values:
        return True
    elif value.lower() in no_values:
        return False
    else:
        values = ",".join(yes_values.union(no_values))
        raise argparse.ArgumentTypeError(
            f"Invalid boolean value: '{value}' is not one of {values}")

p = argparse.ArgumentParser('Export PyPty reconstructions')
p.add_argument(
    '--source', type=Path, required=True, metavar='FILE',
    help='The Numpy file containing the reconstruction')
p.add_argument(
    '--destination', type=Path, metavar='FOLDER',
    help="A directory in which to write the exported TIFF of reconstructions")
p.add_argument(
    '--invert', type=yes_no, default="yes", metavar="YES/NO",
    help="Whether to invert the contrast or not")
p.add_argument(
    '--prefix',
    help='The exported file name will starts with the specified string')
p.add_argument('-v', '--verbose', action='store_true',
               help='Informations about the export will be printed')
args = p.parse_args()
params = pickle.load((args.source.parent / 'params.pkl').open('rb'))
original_data_fn = Path(params['data_path']).with_suffix('.npy')
m = re.search(r'(pos|nr_)(?P<n>\d+)', str(original_data_fn.stem), flags=re.I)
if m is None:
    print(f'Cannot deduce dataset number from path to original data:\n'
          f'{original_data_fn}')
    sys.exit(1)
pos = m['n']
semiang = f"{params['pixel_size_x_A']:.1f}A"
pixsz = f"{params['conv_semiangle_mrad']:.2f}mrad"
m = re.search(r'^(?P<yyyy>\d{4})(?P<mm>\d{2})(?P<dd>\d{2})',
              str(original_data_fn.stem))
if m is None:
    print(f'Cannot deduce date from path to original data:\n'
          f'{original_data_fn}')
    sys.exit(1)
date = f"{m['yyyy']}_{m['mm']}_{m['dd']}"
output_stem = f"{date}_{pixsz}_{semiang}_pos{pos}"
if args.prefix:
    output_stem = f"{args.prefix}_{output_stem}"

co = np.load(args.source)
if args.verbose:
    print(f'Position #{pos}')
    print(f'\tSemi-angle (mrad): {params['conv_semiangle_mrad']:.2f}')
    print(f'\tPixel size (A): {params['pixel_size_x_A']:.3f}')
    print(f'\tImage size (px): {co.shape[0]} x {co.shape[1]}')
phases = np.angle(co)
projection = np.mean(phases, axis=(2,3))
mini, maxi = np.min(projection), np.max(projection)
rescaled = (projection - mini)/(maxi - mini)
converted = (rescaled*2**16).astype(np.uint16)
if args.invert:
    converted = np.iinfo(converted.dtype).max - converted
tiff = Image.fromarray(converted)
tifffn = Path(args.destination) / f'{output_stem}.tiff'
tiff.save(tifffn)
