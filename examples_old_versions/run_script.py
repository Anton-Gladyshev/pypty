from pathlib import Path
import itertools
import subprocess
import sys
import os
import re
import textwrap
import time
from pprint import pp
if 'WING_PRO_DEBUG' in os.environ:
    import wingdbstub
import yaml

PROCESSED_DATA_DIR = Path('/media/data/processed-data')
ORIGINAL_DATA_DIR = Path('/media/data/original-data')
MICROMAMBA = Path("~/.local/bin/micromamba").expanduser()
PYTHONPATH = Path(__file__).parent.parent

CMD_REGX = re.compile(r'^>\s+(.*)$')

def read_input_file(input_file:Path) -> ([Path], [Path]):
    if not input_file.is_absolute():
        input_file = PROCESSED_DATA_DIR / input_file
    with input_file.open() as f:
        job_descs = yaml.safe_load(f)
    for job_desc in job_descs:
        fn = Path(job_desc['scan'])
        if not fn.suffix:
            fn = fn.with_suffix('.npy')
        if not fn.is_absolute():
            branch = input_file.parent.relative_to(PROCESSED_DATA_DIR)
            root = ORIGINAL_DATA_DIR / branch
            fn = root / fn
        job_desc['scan'] = fn
        job_desc['missing'] = not fn.is_file()
        job_desc.setdefault('commands', [])
    return job_descs

def next_available_directory(directory:Path):
    if not directory.exists():
        return directory
    for i in itertools.count(1):
        alt_dir = Path(f'{directory}_{i}')
        if not alt_dir.exists():
            return alt_dir

def monitor(jobs):
    proc, scan = jobs.pop()
    if (ret := proc.poll()) is not None:
        if ret == 0:
            print(f"* Finished {scan}\n")
        else:
            print(f"* Failed {scan} ({ret})\n")
    else:
        jobs.insert(0, (proc, scan))
        time.sleep(2)

def process_job_desc_commands(commands:[list], scan:Path, working_dir_base:Path):
    cmdline = []
    for name, *args in commands:
        match name:
            case 'continue':
                job_nb = args[0]
                marker = f'_{job_nb}' if job_nb else ''
                job_dir = (working_dir_base
                           .with_suffix('')
                           .with_stem(working_dir_base.stem + marker))
                cmdline.extend(['--continue', str(job_dir)])
            case _:
                raise ValueError(f'Unknown command: {name}')
    return cmdline

def process(script:Path, script_args: [str], input_file:Path, output_dir:Path,
            gpus:int, batches:int):
    # Process input file
    job_descs = read_input_file(input_file)
    missing = [job_desc['scan'] for job_desc in job_descs if job_desc['missing']]
    if missing:
        print('ERROR: these files could not be found:')
        for fn in missing:
            print(f'\t{fn}')
        sys.exit(1)

    # Assign GPU devices
    for i, job_desc in enumerate(job_descs):
        job_desc['cuda_device'] = i % gpus

    # Look up script
    if not script.is_absolute():
        script = Path(__file__).parent / script

    # Run!
    jobs = []
    root = output_dir or PROCESSED_DATA_DIR
    for i, job_desc in enumerate(job_descs):
        scan:Path = job_desc['scan']
        working_dir_base = \
            (root / scan.relative_to(ORIGINAL_DATA_DIR)).with_suffix('')
        working_dir = next_available_directory(working_dir_base)
        working_dir.mkdir(parents=True, exist_ok=True)
        env = {}
        env['CUDA_VISIBLE_DEVICES'] = str(job_desc['cuda_device'])
        if 'WING_PRO_DEBUG' in os.environ:
            env['WING_PRO_DEBUG'] = os.environ['WING_PRO_DEBUG']
        cmdline = [
            str(MICROMAMBA),
            'run', '-n', 'pypty0', '-e', f'PYTHONPATH={PYTHONPATH}',
            'python', str(script),
            '--scans', str(scan)]
        extra_args = process_job_desc_commands(job_desc['commands'],
                                               scan, working_dir_base)
        cmdline.extend(extra_args)
        cmdline.extend(script_args)
        proc = subprocess.Popen(
            cmdline,
            cwd=working_dir,
            stdout=(working_dir / 'run.log').open('w'), stderr=subprocess.STDOUT,
            env=env)
        cmdline_str = ' '.join(cmdline)
        print(f"* Started {scan} with:\n{cmdline_str}\n")
        jobs.append((proc, scan))
        while len(jobs) == batches:
            monitor(jobs)
    while len(jobs):
        monitor(jobs)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Execute a script on a series of inputs, balancing GPU's

            The script shall take the following command-line options:
              --scans FILE: the file containing the scans to reconstruct
            It shall write the reconstructions in the current working directory.
        """))
    p.add_argument(
        '--from', type=Path, metavar='FILE', required=True, dest='input_file',
        help='A text file listing the .npy files to process, one per line')
    p.add_argument(
        '--to', type=Path, metavar='DIRECTORY', dest='output_dir',
        help='The directory where to write reconstructions')
    p.add_argument('--run', type=Path, metavar='FILE', required=True, dest='script',
                   help='A Python script to run')
    p.add_argument('--gpus', type=int,  required=True,
                   help='Number of GPU cards')
    p.add_argument('--batches', type=int,  required=True,
                   help='Number of jobs to run in parallel, balancing GPU cards')
    args, others = p.parse_known_args()
    process(script_args=others, **vars(args))



