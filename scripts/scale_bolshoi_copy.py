try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1

import argparse

from collections import OrderedDict
import numpy as np
import cupy as cp

import time

import jax
import jax.numpy as jnp

import h5py

import mpipartition

from diffsmhm.loader import wrap_to_local_volume_inplace
from diffsmhm.analysis.diff_sm import compute_weight_and_jac

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="scale_bolshoi_copy.py",
        description="Time wprp computations over one or more Bolshoi volumes."
    )
    parser.add_argument(
        "--halo-file",
        type=str,
        default="/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"
    )
    parser.add_argument(
        "--hmcut",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--n-copies",
        type=int,
        default=1
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=5
    )
    args = parser.parse_args()

    # 1) process command line args
    halo_file = args.halo_file
    hmcut = args.hmcut

    # note that this assumes bolshoi
    single_length = 250.0
    overload_length = 20.0

    mass_bin_edges = np.array([9.8, 50.0], dtype=np.float64)

    rpbins = np.logspace(-1, 1.3010, 16, dtype=np.float64)
    rpbins = np.concatenate([np.array([0.0]), rpbins])
    zmax = 20.0

    n_rep = args.n_iter

    theta = np.array(list(smhm_params.values()) +
                     list(smhm_sigma_params.values()) +
                     list(disruption_params.values()), dtype=np.float64)

    # check for reasonable number of copies
    n_copies = args.n_copies
    if n_copies < 1:
        if RANK == 0:
            print("Error: `--n_copies` must be a positive number")
        exit(1)

    # check that we have enough ranks for that amount of copies
    # this has to do with how the data is loaded below; with the current algorithm
    # each rank can only load and shift one volume
    if N_RANKS < n_copies**3:
        if RANK == 0:
            print("Error: `--n_copies` must be > n_ranks^3")
        exit(1)

    # 2) load data; a modified version of load and chop

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
    if RANK < n_copies**3:
        # do load
        halos = OrderedDict()
        with h5py.File(halo_file, "r") as hdf:
            _host_mpeak_mask = np.log10(hdf["host_mpeak"][...]) >= hmcut
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

    # ranks not told to load need empty arrays for mpipartition
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

    # 3) time a repeated measurement
    # no need to divide the ranks here, unlike HMC/optimization all ranks only compute

    # copying of halos messes up the crossmatch so we fake it
    idx_to_deposit = np.random.randint(0, 1000, len(halos["halo_x"]))
    idx_to_deposit[:1000] = np.arange(1000)

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

    # device lists
    w_list = []
    dw_list = []
    for d in range(n_devices):
        cp.cuda.Device(d).use()
        w_list.append(cp.array(w))
        dw_list.append(cp.array(dw))

    # warmup wprp
    _, _ = wprp_mpi_comp_and_reduce(
            x1=halos_cp["halo_x"],
            y1=halos_cp["halo_y"],
            z1=halos_cp["halo_z"],
            w1=w_list,
            w1_jac=dw_list,
            inside_subvol=halos_cp["_inside_subvol"],
            rpbins_squared=halos_cp["rpbins_squared"],
            zmax=zmax,
            boxsize=single_length*n_copies,
            kernel_func=wprp_mpi_kernel_cuda
    )

    # timed iterations
    t0 = time.time()
    for i in range(n_rep):
        _, _ = wprp_mpi_comp_and_reduce(
                x1=halos_cp["halo_x"],
                y1=halos_cp["halo_y"],
                z1=halos_cp["halo_z"],
                w1=w_list,
                w1_jac=dw_list,
                inside_subvol=halos_cp["_inside_subvol"],
                rpbins_squared=halos_cp["rpbins_squared"],
                zmax=zmax,
                boxsize=single_length*n_copies,
                kernel_func=wprp_mpi_kernel_cuda
        )
    t1 = time.time()
    tavg = (t1 - t0) / n_rep

    if RANK == 0:
        print("N_RANKS:", N_RANKS,
              "N_DEVICE:", n_devices,
              "N_COPIES:", n_copies,
              "TIME:", tavg, flush=True)
