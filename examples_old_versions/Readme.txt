WARNING: This folder contains OLD examples written by Luc Bourhis (Bruker AXS). To access examples combatible with the new version of PyPty, please refer to the examples folder. This folder is kept as legacy and may server as a reference for future development.




Instructions to reconstruct datasets with 'run_exp.py'

First you need to build a conda environment with all the necessary modules:
install conda, mamba, or micromamba. I'll use micromamba here.
Then:

% micromamba env create -f pypty.yml

where pypty.yml is the file at the root of the repository. You only need to do that
once, obviously.

Then open a terminal and run

% path/to/jupyter-ptychography/examples/run_exp.bash --help

to see the command-line options of the reconstruction script. For terseness, I will
assume that we are in the directory jupyter-ptychography/examples.

To reconstruct a single scan, do

% run_exp.bash -- scan=path/to/xxx_4DSTEM.npy --device=1 --output-dir=...

To reconstruct many scans it is easier to use

% run_exp.bash --instructions=path/datasets.txt --output-dir=...

where datasets.txt contains a list of .npy file path. One per line. Lines starting
with the character '#' are skipped as well as blank lines. This will run several
reconstructions in parallel and distribute the load to all the GPUs: you need to
change the variables BATCH and DEVICES in run_exp.py to fit your machine (those
should be command-line options too but I did not take the time to do that).

In either case, if the dataset is named YYY/XXX.npy, then a directory XXX will be
created in the directory specified with --output-dir, and all the reconstruction
outputs will go into XXX. In particular, there will be a file co.npy that contains
the reconstructed object whereas cp.npy contains the reconstructed probe.

The reconstruction algorithm is fairly slow. Even on A100, the main loop at the
end (run_loop in the Python script) takes about 15 to 30 mins per epoch of
optimisation, and it will do 40 of them. You can reduce the number of epochs by
changing `epoch_max` in the dictionary `pypty_params` near the top of the
script. But the preliminary phases will still take between 30 and 60 mins. It
would be nice to be able to crop the images: `run_exp.py` has an option --margin
for that but currently this crashes the reconstruction.
