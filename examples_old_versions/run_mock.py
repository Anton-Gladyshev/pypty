import argparse
from pathlib import Path
import os
from pprint import pp

p = argparse.ArgumentParser(description="Mock script for debugging `run_script.py`")
p.add_argument(
    '--scans', type=Path, metavar='FILE', required=True, dest='scans',
    help='the file containing the scans to reconstruct')
args, others = p.parse_known_args()
print(f"Reconstruct: {args.scans}")
print(f"Reconstructions go into: {Path.cwd()}")
print(f"Run on GPU #{os.environ['CUDA_VISIBLE_DEVICES']}")
print("Other arguments:")
pp(others)
try:
    import pypty
except ImportError:
    print("PYTHONPATH is not correctly set")
else:
    print("PYTHONPATH is correctly set")
