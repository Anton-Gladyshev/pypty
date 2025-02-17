import argparse
from pathlib import Path
import sys
from math import pi, sqrt, cos, sin
import numpy as np
from scipy import linalg
import h5py
import matplotlib.pyplot as plt

# command-line arguments
cmdline = argparse.ArgumentParser(
    description=('Analyse the scan grid exported by script "pack_scan_grid.py"'))
cmdline.add_argument(
    "--scan-grid", type=Path, metavar='HDF5_FILE', required=True,
    help='the HDF5 file containging the scan grid and relevant metadata')
cmdline.add_argument(
    '--title', required=True,
    help='to identify the analysis (outputs will go into a directory '
    'with that name next to the scan grid HDF5 file)')
cmdline.add_argument(
    '--plot-positions', action='store_true',
    help='Whether to plot the fitted positions vs the regular grid')
cmdline.add_argument(
    '--plot-displacements', action='store_true',
    help='Whether to plot the scan position displacements')
cmdargs = cmdline.parse_args()
output_dir:Path = cmdargs.scan_grid.parent / cmdargs.title
output_dir.mkdir(parents=True, exist_ok=True)

# Read input file
data = []
with h5py.File(cmdargs.scan_grid) as top:
    for posg in top.values():
        idx = posg.attrs['idx']
        if posg.attrs['rotation_unit'] != 'deg':
            print(f'Error for scan #{idx}: '
                  'rotation angle shall be in degrees.')
            continue
        theta = posg.attrs['rotation']*pi/180 # rad
        positions = posg['scan_positions']
        if len(positions.shape) != 2 or positions.shape[1] != 2:
            print(f'Error for scan #{idx}: '
                  'scan grid is not shaped as N x 2 array')
            continue
        step = np.median
        n =  sqrt(positions.shape[0])
        if n != int(n):
            print(f'Error for scan #{idx}: scan grid is not squared')
            continue
        step = posg.attrs['scan_step']
        unit = posg.attrs['scan_step_unit']
        data.append((idx, theta, step, unit, int(n), positions[:]))

# Process each dataset
for idx, theta, exp_step, unit, grid_size, positions in data:
    header = f"Dataset #{idx}"
    print(header)
    print('-'*len(header))

    # Rotate positions back
    o = np.mean(positions)
    cs, sn = cos(theta), sin(theta)
    rot = np.array([[cs, sn],
                    [-sn, cs]])
    p:np.ndarray = o + (positions - o).dot(rot)

    # Analyse scan steps
    row_xs = np.mean([p[i:i+grid_size,0] for i in range(0, p.shape[0], grid_size)],
                       axis=1)
    col_xs = np.mean([p[i::grid_size,1] for i in range(grid_size)],
                       axis=1)
    step_xs = np.diff(row_xs)
    step_ys = np.diff(col_xs)
    step_x, step_y = np.mean(step_xs), np.mean(step_ys)
    delta_x, delta_y = np.std(step_xs), np.std(step_ys)
    if abs(step_x - step_y)/np.sqrt(delta_x**2 + delta_y**2) > 1e-2:
        print(f'Error for scan #{idx}: scan grid does not look square')
        continue
    print(f"Step: X = {step_x} ± {delta_x}, Y = {step_y} ± {delta_y}")
    step = np.mean([step_xs, step_ys])
    o = np.array(row_xs[0], col_xs[0])

    # Construct regular grid
    indices = np.arange(grid_size)
    i,j = np.meshgrid(indices, indices)
    q = np.array([i.ravel(), j.ravel()]).T
    q = o + step*q

    # Rescale
    p *= exp_step/step # fitted positions
    q *= exp_step/step # regular grid
    delta = p - q # displacement
    lengths = linalg.norm(delta, axis=1)
    angles = np.atan2(delta[:,1], delta[:,0])*180/pi
    if cmdargs.plot_positions:
        fig1, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.set_aspect('equal')
        pxsz = step/10
        ax.scatter(q[:,0], q[:,1], s=pxsz, color='blue')
        ax.scatter(p[:,0], p[:,1], s=pxsz, color='red')
        ax.set_title('Scan position: fit (red) vs regular grid (blue)')
        fig1.suptitle(f'Dataset #{idx}')
        fig1.show()
    if cmdargs.plot_displacements:
        fig2, axs = plt.subplots(2, 2, figsize=(20, 10))
        im_ang = axs[0,0].matshow(angles.reshape((grid_size, grid_size)),
                                  cmap='viridis')
        axs[0,0].set_title('Angle of scan position displacements')
        plt.colorbar(im_ang, ax=axs[0,0])
        im_len = axs[0,1].matshow(lengths.reshape((grid_size, grid_size)),
                                  cmap='viridis')
        axs[0,1].set_title('Length of scan position displacements')
        plt.colorbar(im_len, ax=axs[0,1])
        im_x = axs[1,0].matshow(delta[:,0].reshape((grid_size, grid_size)),
                                cmap='viridis')
        axs[1,0].set_title('x-coordinate of scan position displacements')
        plt.colorbar(im_x, ax=axs[1,0])
        im_y = axs[1,1].matshow(delta[:,1].reshape((grid_size, grid_size)),
                                cmap='viridis')
        axs[1,1].set_title('y-coordinate of scan position displacements')
        plt.colorbar(im_y, ax=axs[1,1])
        fig2.suptitle(f'Dataset #{idx}')
        fig2.show()
    plt.show()
