import numpy as np
from collections import OrderedDict
import h5py
import jax

from diff_sm import(
    compute_weight_and_jac
)
from diffsmhm.galhalo_models.merging import (
    _calculate_indx_to_deposit
)

# first we want to load data
particle_file = "/Users/josephwick/Documents/Argonne23/data/hlist_1.00231.particles.halotools_v0p4.hdf5"
halo_file = "/Users/josephwick/Documents/Argonne23/data/value_added_orphan_complete_bpl_1.002310.h5"

# load_and_chop requires multiple mpi ranks, let's make a basic one here for loading halos
def load_testing(halo_file, host_mpeak_cut=0.0):
    important_keys = ["upid", "halo_id", "mpeak", "host_mpeak", "vmax_frac"]

    halos =  OrderedDict()

    with h5py.File(halo_file, "r") as hdf:
        _host_mpeak_mask = np.log10(hdf["host_mpeak"][...]) >= host_mpeak_cut
        for key in hdf.keys():
            # only keep columns we want
            if key not in important_keys:
                continue

            # integer dtypes
            if key in ("halo_id", "upid", "mmhid"):
                dt = "i8"
            else:
                dt = "f4"
            halos[key] = hdf[key][...][_host_mpeak_mask].astype(dt)

    # compute some logs once and for all
    halos["logmpeak"] = np.log10(halos["mpeak"])
    halos["loghost_mpeak"] = np.log10(halos["host_mpeak"])
    halos["logvmax_frac"] = np.log10(halos["vmax_frac"])

    return halos

halos = load_testing(halo_file, host_mpeak_cut=14.0)
idx_to_deposit = _calculate_indx_to_deposit(halos["upid"], halos["halo_id"])

print(idx_to_deposit)
print(halos["halo_id"])
print(halos["upid"], flush=True)

mass_bin_edges = np.array([10.0, 12.0], dtype=np.float64)

theta_default = np.array([
    11.35, -1.65, 1.58489319, 2.5, 0.5, 
    0.3, 0.2, 12.0, 0.1,
    13.5, 3.16227766, -1., -0.7, 3.16227766,
    12.65, 1.58489319, 0.15, 0.9, 13., 1.58489319, 0.3, 2., 3.16227766 ])

#breakpoint()

w_data, dw_data = compute_weight_and_jac(
                    halos["logmpeak"],
                    halos["loghost_mpeak"],
                    halos["logvmax_frac"],
                    halos["upid"],
                    idx_to_deposit,
                    mass_bin_edges[0], mass_bin_edges[1],
                    theta_default
)
                    
print("dw min:", np.min(dw_data, axis=1)[9:14])
print("dw max:", np.max(dw_data, axis=1)[9:14])
print("dw mean:", np.mean(dw_data, axis=1)[9:14])

