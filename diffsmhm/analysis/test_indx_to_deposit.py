import numpy as np
from collections import OrderedDict
import h5py

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

# let's check idx_to_deposit

# 1) do all halos with upid == -1 point to themselves?
host_idxs = np.where(halos["upid"]==-1)[0]

print(np.all(idx_to_deposit[host_idxs] == host_idxs))

# 2) do all halos with upid != -1 point to their upid?
upids = np.copy(halos["upid"])
upids[halos["upid"]==-1] = halos["halo_id"][halos["upid"]==-1]
print(np.all(halos["halo_id"][idx_to_deposit] == upids))

