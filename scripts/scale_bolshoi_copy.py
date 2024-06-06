from collections import OrderedDict
import numpy as np
import cupy as cp
import time
import sys

import jax
import jax.numpy as jnp

import h5py

import mpipartition

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1

from diffsmhm.loader import wrap_to_local_volume_inplace
from diffsmhm.analysis.tools.diff_sm import compute_weight_and_jac
from diffsmhm.galhalo_models.merging import _calculate_indx_to_deposit

from diffsmhm.diff_stats.mpi.wprp import wprp_mpi_comp_and_reduce
from diffsmhm.diff_stats.cuda.wprp import wprp_mpi_kernel_cuda

from diffsmhm.galhalo_models.sigmoid_smhm import (
    DEFAULT_PARAM_VALUES as smhm_params
)
from diffsmhm.galhalo_models.sigmoid_smhm_sigma import (
    DEFAULT_PARAM_VALUES as smhm_sigma_params
)
from diffsmhm.galhalo_models.sigmoid_disruption import (
    DEFAULT_PARAM_VALUES as disruption_params
)
from diffsmhm.galhalo_models.sigmoid_quenching import (
    DEFAULT_PARAM_VALUES as quenching_params
)


halo_file = "/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"
single_length = 250.0 # Mpc

overload_length = 20.0 # Mpc

mass_bin_edges = np.array([9.8, 50.0], dtype=np.float64)

rpbins = np.logspace(-1, 1.3010, 16, dtype=np.float64)
zmax = 20.0

theta = np.array(list(smhm_params.values()) +
                 list(smhm_sigma_params.values()) +
                 list(disruption_params.values()) +
                 list(quenching_params.values()), dtype=np.float64)

# 0) command line args

# usage: python weak_scale.py [N_COPIES_PER_DIM]
if len(sys.argv) != 2 or int(sys.argv[1]) < 1:
    print("Usage: mpiexec -np [N_RANKS] python weak_scale.py [N_BOLSHOIS_PER_DIM]")
    print("N_BOLSHOIS_PER_DIM must be a positive number")
    exit(1)

n_copies = int(sys.argv[1])
# check that we have enough ranks for that amount of copies
if N_RANKS < n_copies**3:
    print("Weak scaling error: Not enough ranks for specified number of copies")
    exit(1)

# 1) load data; going to need a modified version of load and chop

# define shift amounts
shift_amounts_per_rank = np.zeros((n_copies**3, 3), dtype="i")
shift_idx = 0
for i in range(n_copies):
    for j in range(n_copies):
        for k in range(n_copies):
            shift_amounts_per_rank[shift_idx] = [i, j, k]
            shift_idx += 1

# relevant ranks load and shift
# note that the ordering of these is particular if we want mpipartition to cooperate
# the keys need to be in the same order across all ranks
important_keys = [
    "halo_id", "host_mpeak", "mpeak", "time_since_infall", "upid", "vmax_frac",
    "logmpeak", "loghost_mpeak", "logvmax_frac", "halo_x", "halo_y", "halo_z",
    "x", "y", "z"
]
host_mpeak_cut = 0.0 
if RANK < n_copies**3:
    # do load 
    halos = OrderedDict()
    with h5py.File(halo_file, "r") as hdf:
        _host_mpeak_mask = np.log10(hdf["host_mpeak"][...]) >= host_mpeak_cut
        for key in hdf.keys():
            if key not in important_keys:
                continue

            if key in ("halo_id", "upid"):
                dt = "i8"
            else:
                dt = "f4"
            halos[key] = hdf[key][...][_host_mpeak_mask].astype(dt)

    # compute logs
    halos["logmpeak"] = np.log10(halos["mpeak"])
    halos["loghost_mpeak"] = np.log10(halos["host_mpeak"])
    halos["logvmax_frac"] = np.log10(halos["vmax_frac"])

    # change "x"/"y"/"z" to "halo_x"/etc for clarity
    halos["halo_x"] = halos["x"].copy()
    halos["halo_y"] = halos["y"].copy()
    halos["halo_z"] = halos["z"].copy()
    del halos["x"]
    del halos["y"]
    del halos["z"]

    # fix "out of bounds" halos using periodicity
    for pos in ["halo_x", "halo_y", "halo_z"]:
        halos[pos][halos[pos] < 0] += single_length
        halos[pos][halos[pos] > single_length] -= single_length

    # do shift
    halos["halo_x"] += single_length*shift_amounts_per_rank[RANK, 0]
    halos["halo_y"] += single_length*shift_amounts_per_rank[RANK, 1]
    halos["halo_z"] += single_length*shift_amounts_per_rank[RANK, 2]

# halos not told to load need empty arrays for mpipartition
else:
    halos = OrderedDict()

    for key in important_keys:
        if key in ("halo_id", "upid"):
            dt = "i8"
        else:
            dt = "f4"
        halos[key] = np.array([1], dtype=dt)
    del halos["x"]
    del halos["y"]
    del halos["z"]

# distribute
partition = mpipartition.Partition()

halos = mpipartition.distribute(partition, single_length*n_copies, data=halos,
                                coord_keys=["halo_x", "halo_y", "halo_z"])
halos["rank"] = np.zeros_like(halos["halo_x"], dtype=np.int32) + RANK

halos = mpipartition.overload(partition, single_length*n_copies, halos,
                              overload_length,
                              ["halo_x", "halo_y", "halo_z"])

halos["_inside_subvol"] = halos["rank"] == RANK

# wrap to volume
center = single_length * n_copies * (
    np.array(partition.extent) / 2.0 + 
    np.array(partition.origin)
)

wrap_to_local_volume_inplace(halos["halo_x"], center[0], single_length*n_copies)
wrap_to_local_volume_inplace(halos["halo_y"], center[1], single_length*n_copies)
wrap_to_local_volume_inplace(halos["halo_z"], center[2], single_length*n_copies)

# create jax and cupy copies
n_devices = jax.local_device_count()
halos_jax = OrderedDict()
halos_cp = OrderedDict()
for k in halos.keys():
    halos_jax[k] = jax.device_put(jnp.array(halos[k]), jax.devices()[0])

    halos_cp[k] = []
    for d in range(n_devices):
        cp.cuda.Device(d).use()
        halos_cp[k].append(cp.array(halos[k]))

halos_cp["rpbins_squared"] = []
for d in range(n_devices):
    cp.cuda.Device(d).use()
    halos_cp["rpbins_squared"].append(cp.array(rpbins**2))

# 2) time a repeated measurement

# copying of halos messes up the crossmatch
idx_to_deposit = np.random.randint(0, 1000, len(halos["halo_x"]))
idx_to_deposit[:1000] = np.arange(1000)

# weights
w, dw = compute_weight_and_jac(
            logmpeak=halos_jax["logmpeak"],
            loghost_mpeak=halos_jax["loghost_mpeak"],
            log_vmax_by_vmpeak=halos_jax["logvmax_frac"],
            upid=halos_jax["upid"],
            idx_to_deposit=idx_to_deposit,
            mass_bin_low=mass_bin_edges[0],
            mass_bin_high=mass_bin_edges[1],
            theta=theta
)

mask_wgt = w > 0.0
mask_dwgt = cp.sum(cp.abs(dw), axis=0) > 0.0
full_mask = mask_wgt & mask_dwgt

# need our device lists too
w_list = []
dw_list = []
full_mask_list = []
for d in range(n_devices):
    cp.cuda.Device(d).use()
    w_list.append(cp.array(w))
    dw_list.append(cp.array(dw))
    full_mask_list.append(cp.array(full_mask))

# warmup wprp
_, _ = wprp_mpi_comp_and_reduce(
        x1=halos_cp["halo_x"],
        y1=halos_cp["halo_y"],
        z1=halos_cp["halo_z"],
        w1=w_list,
        w1_jac=dw_list,
        mask=full_mask_list,
        inside_subvol=halos_cp["_inside_subvol"],
        rpbins_squared=halos_cp["rpbins_squared"],
        zmax=zmax,
        boxsize=single_length*n_copies,
        kernel_func=wprp_mpi_kernel_cuda
)

# rpwp
n_rep = 10
tsum = 0.0
t0 = time.time()
for _ in range(n_rep):
    _, _ = wprp_mpi_comp_and_reduce(
            x1=halos_cp["halo_x"],
            y1=halos_cp["halo_y"],
            z1=halos_cp["halo_z"],
            w1=w_list,
            w1_jac=dw_list,
            mask=full_mask_list,
            inside_subvol=halos_cp["_inside_subvol"],
            rpbins_squared=halos_cp["rpbins_squared"],
            zmax=zmax,
            boxsize=single_length*n_copies,
            kernel_func=wprp_mpi_kernel_cuda
    )

t1 = time.time()
tavg = (t1 - t0)/ n_rep

if RANK == 0:
    print("N_RANKS:", N_RANKS,
          "N_DEV:", n_devices,
          "N_COPIES:", n_copies,
          "TIME:", tavg, flush=True)
