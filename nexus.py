import numpy as np
import sys
import os
import csv
import time
import pickle
import types
import copy
import inspect
from pypty.fft import *

import h5py
import datetime


def convert_to_nxs(folder_path, output_file):
    """
    Convert saved PyPty reconstruction data to NeXus (.nxs) format.

    Parameters
    ----------
    folder_path : str
        Directory containing saved reconstruction files.
    output_file : str
        Path where the NeXus file will be saved.

    Returns
    -------
    None
    """
    co_path = os.path.join(folder_path, "co.npy") ## this is my object
    cp_path = os.path.join(folder_path, "cp.npy") ## this is my probe
    cg_path = os.path.join(folder_path, "cg.npy") # positions
    pkl_path = os.path.join(folder_path, "params.pkl") # parameters
    if not all(os.path.exists(p) for p in [co_path, cp_path, cg_path, pkl_path]):
        raise FileNotFoundError("Missing required files.")
    co = np.load(co_path)
    cpr = np.load(cp_path)
    cg = np.load(cg_path)
    creation_time = datetime.datetime.fromtimestamp(os.path.getmtime(co_path)).isoformat()
    with open(pkl_path, "rb") as f:
        metadata = pickle.load(f)
    pixel_size_y = metadata["pixel_size_y_A"]
    pixel_size_x = metadata["pixel_size_x_A"]
    slice_spacing = metadata.get("slice_distances", [1])[0]
    chemical_formula = metadata.get("chemical_formula", "")
    sample_name = metadata.get("sample_name", None)
    if sample_name is None:
        sample_name= (metadata.get("data_path", "").split("/")[-1]).split(".")[0]
    cg[:, 0] *= pixel_size_y
    cg[:, 1] *= pixel_size_x

    probe_shape = cpr.shape
    is_probe_4d = len(probe_shape) == 4
    
    co = co[::-1, :, :, :].transpose(3, 2, 0, 1)

    # Flip y-axis and reorder axes for probe (modes, scenarios?, y, x)
    if is_probe_4d:
        cpr = cpr[::-1, :, :, :].transpose(2, 3, 0, 1)
    else:
        cpr = cpr[::-1, :, :].transpose(2, 0, 1)

    with h5py.File(output_file, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "object" # open object by default
        
        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        sample.create_dataset("name", data=sample_name)
        if chemical_formula != "":
            sample.create_dataset("chemical_formula", data=chemical_formula)
        # Object data
        obj_grp = entry.create_group("object")
        obj_grp.attrs["NX_class"] = "NXdata"
        obj_grp.create_dataset("data", data=co)
        obj_grp.attrs["axes"] = ["modes", "z", "y", "x"]
        obj_grp.create_dataset("modes", data=np.arange(co.shape[0]))
        obj_grp["modes"].attrs["units"] = "mode index"
        obj_grp.create_dataset("z", data=np.arange(co.shape[1]) * slice_spacing)
        obj_grp["z"].attrs["units"] = "angstrom"
        obj_grp.create_dataset("y", data=np.arange(co.shape[2]) * pixel_size_y)
        obj_grp["y"].attrs["units"] = "angstrom"
        obj_grp.create_dataset("x", data=np.arange(co.shape[3]) * pixel_size_x)
        obj_grp["x"].attrs["units"] = "angstrom"

        # Instrument
        instr_grp = entry.create_group("instrument")
        instr_grp.attrs["NX_class"] = "NXinstrument"

        # Probe data
        probe_grp = instr_grp.create_group("probe")
        probe_grp.attrs["NX_class"] = "NXbeam"
        probe_grp.create_dataset("data", data=cpr)
        probe_axes = ["modes"] + (["scenarios"] if is_probe_4d else []) + ["y", "x"]
        probe_grp.attrs["axes"] = probe_axes
        probe_grp.create_dataset("modes", data=np.arange(cpr.shape[0]))
        probe_grp["modes"].attrs["units"] = "mode index"
        offset = 1
        if is_probe_4d:
            probe_grp.create_dataset("scenarios", data=np.arange(cpr.shape[1]))
            probe_grp["scenarios"].attrs["units"] = "scenario index"
            offset += 1
        probe_grp.create_dataset("y", data=np.arange(cpr.shape[offset]) * pixel_size_y)
        probe_grp["y"].attrs["units"] = "angstrom"
        probe_grp.create_dataset("x", data=np.arange(cpr.shape[offset + 1]) * pixel_size_x)
        probe_grp["x"].attrs["units"] = "angstrom"

        # Scan positions
        scan_grp = entry["instrument"].create_group("scan")
        scan_grp.attrs["NX_class"] = "NXpositioner"
        scan_grp.create_dataset("positions", data=cg)
        scan_grp["positions"].attrs["units"] = "angstrom"
        scan_grp.attrs["axes"] = ["positions", "coordinates"]

        # Reconstruction parameters
        recon_grp = entry.create_group("reconstruction")
        recon_grp.attrs["NX_class"] = "NXprocess"
        recon_grp.create_dataset("software", data="PyPty")
        recon_grp.create_dataset("version", data="v2.0")
        recon_grp.create_dataset("date", data=creation_time)
        recon_grp.create_dataset("folder",  data=metadata.get("output_folder", ""))
        recon_grp.create_dataset("dataset", data=metadata.get("data_path", "").split("/")[-1])
        p_grp=recon_grp.create_group("reconstruction parameters")
        p_grp.attrs["NX_class"] = "NXcollection"
        p_grp.create_dataset("software", data="PyPty")
        for key, value in metadata.items():
            if value is None: value="None";
            p_grp.create_dataset(key, data=value)
    print(f"NeXus file saved as: {output_file}")



