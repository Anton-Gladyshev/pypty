from pathlib import Path
import glob
import pickle
import re
import datetime
import numpy as np
from PIL import Image
from pprint import pp
import pypty
import sys

PIXSZ = "pixel_size_x_A"
SEMIANG = "conv_semiangle_mrad"

LBL_PAT = re.compile(r'_([^_]+)$')
def unit_of(label):
    m = LBL_PAT.search(label)
    if not m:
        raise ValueError(f"{label} is not in the format xxx_<UNIT>")
    return m.group(1)

DATE_PAT = re.compile(r'^(\d{4})(\d{2})(\d{2})_')
def date_of(npyfn):
    m = DATE_PAT.search(str(npyfn.parent.stem))
    if not m:
        raise ValueError(f"{npyfn} is not in the format YYYYMMDD_xxxx")
    return datetime.date(*map(int, m.groups()))

POS_PAT = re.compile(r'pos(\d+)')
def position_of(npyfn):
    m = POS_PAT.search(str(npyfn.parent.stem))
    if not m:
        raise ValueError(f"{npyfn} is not in the format xxx_pos<ddd>_yyy")
    return m.group(1)

VERSION_PAT = re.compile(r'_(\d+)$')
def version_of(npyfn):
    m = VERSION_PAT.search(str(npyfn.parent.stem))
    if m:
        return int(m.group(1))
    else:
        return 0

def collect(folder:Path, epoch=None):
    co_pat = f'checkpoint_obj_epoch_{epoch}.npy' if epoch is not None else 'co.npy'
    co_pat = str(folder / '**' / co_pat)
    for cofn in glob.glob(co_pat, recursive=True):
        cofn = Path(cofn)
        paramsfn = cofn.parent / "params.pkl"
        with open(paramsfn, "rb") as f:
            params = pickle.load(f)
        ctfn = cofn.parent / "ct.npy"
        yield params, cofn, ctfn

def sum_up(co):
    return np.average(np.angle(co), axis=2)[:,:,0]

def slice_it(co, i):
    return np.angle(co)[:,:,i,0]

def compute_phase_correction(params, co, ct):
    low_freq = extract_low_freq(ct)
    params,o = pypty.get_ptycho_obj_from_scan(params, n_slices=1,
                                              array_phase=low_freq,
                                              array_abs=None, scale_phase=1,
                                              scale_abs=1, cutoff=None)
    o = o[:,:,0,0]
    return np.angle(o)

def extract_low_freq(t):
    rsc=256*1.7814286374960782*np.sqrt(200*(200+2*511))/12.4
    sh=int((t.shape[0])**0.5)
    dy_p=-rsc*t[:,4].reshape(sh,sh)
    dx_p=-rsc*t[:,5].reshape(sh,sh)
    angle=np.pi/2
    dx= np.cos(angle)*dx_p+np.sin(angle)*dy_p
    dy=-np.sin(angle)*dx_p+np.cos(angle)*dy_p
    qx,qy=np.meshgrid(pypty.fftshift(pypty.fftfreq(dx.shape[0], 1)),
                      pypty.fftshift(pypty.fftfreq(dy.shape[1], 1)))
    q=(qx**2+qy**2)**0.5
    dx=pypty.shift_fft2(dx)
    dy=pypty.shift_fft2(dy)
    dx=dx*(qx!=0)*(qy!=0)
    dy=dy*(qy!=0)*(qx!=0)
    dx=np.real(pypty.ifft2_ishift(dx))
    dy=np.real(pypty.ifft2_ishift(dy))
    return pypty.iterative_dpc(dx, dy,
                               step_size=1e-5, hpass=1e1, lpass=1e4,
                               num_iterations=100, pad_width=500, phase=None
                               )/(2*np.pi)

def crop(phases, m, n):
    m0, n0 = phases.shape
    # compute edge
    def slicing(k, k0):
        if k < k0:
            i = (k0 - k)//2
            return slice(i, -i)
        else:
            return slice(None)
    si = slicing(m, m0)
    sj = slicing(n, n0)
    return phases[si, sj]

if __name__ == '__main__':
    import argparse
    cmdline = argparse.ArgumentParser("Export PyPty reconstruction")
    cmdline.add_argument(
        '--source', type=Path, required=True,
        help="A directory to recursively explore for reconstructions")
    cmdline.add_argument(
        '--epoch', type=int,
        help="Use the given epoch checkpoint instead of the final co.npy")
    cmdline.add_argument(
        '--skip', nargs='*', default=[],
        help='Skip the dataset matching a Python boolean expression featuring any of'
        ' `pos` (the position), `vers` (the version); '
        ' operators `and`, `or`, and parentheses are obviously allowed')
    cmdline.add_argument(
        '--destination', type=Path,
        help="A directory in which to write the exported TIFF of reconstructions")
    cmdline.add_argument('--low-frequencies-correction', action='store_true',
                         help='Whether to postprocess the reconstruction by '
                         'moving low spatial frequencies from the tilt map to '
                         'the object')
    size_group = cmdline.add_mutually_exclusive_group()
    size_group.add_argument('--crop', metavar='MxN',
                            help='Crop the image to the given size, removing the'
                            ' same margins left and right on one hand, and top and'
                            ' bottom on the other hand')
    size_group.add_argument('--shave',
                            metavar='ALL|HORIZONTAL,VERTICAL|LEFT,TOP,RIGHT,BOTTOM',
                            help='Remove band(s) of pixels at the edges')
    cmdargs = cmdline.parse_args()
    if cmdargs.crop is not None:
        cmdargs.crop = tuple(map(int, cmdargs.crop.split('x')))
    if cmdargs.shave is not None:
        edges = tuple(map(int, cmdargs.shave.split(',')))
        match edges:
            case (all_,):
                edges = (all_,)*4
            case (horizontal, vertical):
                edges = (horizontal, vertical, horizontal, vertical)
            case (left, top, right, bottom):
                pass
            case _:
                print("Error: --shave has an illegal argument")
                sys.exit(1)
        cmdargs.shave = edges
    if cmdargs.destination is not None:
        cmdargs.destination.mkdir(exist_ok=True, parents=True)
    all_skipped = [eval(f'lambda pos, vers, date:{selector}')
                   for selector in cmdargs.skip]
    for params, cofn, ctfn in collect(cmdargs.source, epoch=cmdargs.epoch):
        try:
            pos = position_of(cofn)
            vers = version_of(cofn)
            date = date_of(cofn).strftime("%Y_%m_%d")
        except ValueError:
            print(f">>> Skip {cofn}: unknown file name format")
            continue
        if any(skipped(pos, vers, date) for skipped in all_skipped):
            continue
        pixsz = f"{params[PIXSZ]:.2f}{unit_of(PIXSZ)}"
        semiang = f"{params[SEMIANG]:.0f}{unit_of(SEMIANG)}"
        destination = cmdargs.destination or cofn.parent
        stem = f"{date}_{pixsz}_{semiang}_pos{pos}"
        ct = np.load(ctfn)
        co = np.load(cofn)
        if cmdargs.low_frequencies_correction:
            print(">>> Compute low frequencies correction from tilt map")
            phase_correction = compute_phase_correction(params, co, ct)
            print(">>> Done")
        else:
            phase_correction = None
        slices = co.shape[2]
        print(f"{cofn.parent.stem}/{cofn.stem}")
        actions = []
        print(f"* {slices} slice(s)")
        tifffn = destination / f"{stem}.tiff"
        m, n = cmdargs.crop if cmdargs.crop is not None else co.shape[:2]
        print(f"* output: {tifffn} ({m}x{n})")
        if slices > 1:
            actions.append([(sum_up,), tifffn])
            for i in range(slices):
                tifffn = destination / f"{stem}_slice{i}.tiff"
                print(f"* output: {tifffn}")
                actions.append([(slice_it, i), tifffn])
        else:
            actions.append([(slice_it, 0), tifffn])
        for (f, *args), tifffn in actions:
            phases = f(co, *args)
            if phase_correction is not None:
                phases += phase_correction
            if cmdargs.crop is not None:
                phases = crop(phases, *cmdargs.crop)
            elif cmdargs.shave is not None:
                left, top, right, bottom = cmdargs.shave
                phases = phases[left:-right, top:-bottom]
            mini, maxi = np.min(phases), np.max(phases)
            rescaled = (phases - mini)/(maxi - mini)
            converted = (rescaled*2**16).astype(np.uint16)
            tiff = Image.fromarray(converted)
            tiff.save(tifffn)
