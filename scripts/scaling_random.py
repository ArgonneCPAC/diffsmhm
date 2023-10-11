import time
import argparse
import sys

import numpy as np

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1

if COMM is not None:
    import mpipartition

from diffsmhm.diff_stats.mpi.wprp import wprp_mpi_comp_and_reduce
from diffsmhm.loader import wrap_to_local_volume_inplace
from diffsmhm.utils import time_step


def _distribute_data(data, lbox, partition):
    partition = mpipartition.Partition()
    data = mpipartition.distribute(partition, lbox, data, ["x", "y", "z"])
    data["rank"] = np.zeros_like(data["x"], dtype=np.int32) + RANK

    data = mpipartition.overload(partition, lbox, data, 25, ["x", "y", "z"])
    data["_inside_subvol"] = data["rank"] == RANK

    center = lbox * (
        np.array(partition.extent) / 2.0
        + np.array(partition.origin)
    )

    wrap_to_local_volume_inplace(data["x"], center[0], lbox)
    wrap_to_local_volume_inplace(data["y"], center[1], lbox)
    wrap_to_local_volume_inplace(data["z"], center[2], lbox)

    return data


def _gen_data(npts, lbox, partition):
    rng = np.random.RandomState(seed=42+RANK)
    halo_catalog = dict()
    halo_catalog["x"] = (
        (rng.uniform(size=npts) * partition.extent[0] + partition.origin[0]) * lbox
    )
    halo_catalog["y"] = (
        (rng.uniform(size=npts) * partition.extent[1] + partition.origin[1]) * lbox
    )
    halo_catalog["z"] = (
        (rng.uniform(size=npts) * partition.extent[2] + partition.origin[2]) * lbox
    )
    halo_catalog["w"] = rng.uniform(size=npts)
    halo_catalog["dw0"] = rng.uniform(size=npts)
    halo_catalog["dw1"] = rng.uniform(size=npts)
    halo_catalog["dw2"] = rng.uniform(size=npts)

    halo_catalog = _distribute_data(halo_catalog, lbox, partition)
    halo_catalog["dw"] = np.stack(
        [halo_catalog["dw0"], halo_catalog["dw1"], halo_catalog["dw2"]],
        axis=0,
    )
    return halo_catalog


def _run_cuda(halo_catalog, lbox, zmax, rbins_sqaured):
    from diffsmhm.diff_stats.cuda.wprp import (
        wprp_mpi_kernel_cuda,
    )

    wprp, wprp_grad = wprp_mpi_comp_and_reduce(
        x1=np.asarray(halo_catalog["x"].astype(np.float64)),
        y1=np.asarray(halo_catalog["y"].astype(np.float64)),
        z1=np.asarray(halo_catalog["z"].astype(np.float64)),
        w1=np.asarray(halo_catalog["w"].astype(np.float64)),
        w1_jac=np.asarray(halo_catalog["dw"].astype(np.float64)),
        inside_subvol=np.asarray(halo_catalog["_inside_subvol"]),
        rpbins_squared=rbins_squared,
        zmax=zmax,
        boxsize=lbox,
        kernel_func=wprp_mpi_kernel_cuda,
    )


def _run_cpu(halo_catalog, lbox, zmax, rbins_sqaured):
    from diffsmhm.diff_stats.cpu.wprp import (
        wprp_mpi_kernel_cpu,
    )
    wprp, wprp_grad = wprp_mpi_comp_and_reduce(
        x1=halo_catalog["x"].astype(np.float64),
        y1=halo_catalog["y"].astype(np.float64),
        z1=halo_catalog["z"].astype(np.float64),
        w1=halo_catalog["w"].astype(np.float64),
        w1_jac=halo_catalog["dw"].astype(np.float64),
        inside_subvol=halo_catalog["_inside_subvol"],
        rpbins_squared=rbins_squared,
        zmax=zmax,
        boxsize=lbox,
        kernel_func=wprp_mpi_kernel_cpu,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='scaling_random.py',
        description='run a scaling test',
    )
    parser.add_argument(
        '-b', '--backend',
        choices=["cpu", "cuda"],
        required=True,
    )
    parser.add_argument(
        '-n', '--npts',
        type=int,
        default=10_000_000,
    )
    parser.add_argument(
        '-l', '--lbox',
        type=float,
        default=250.0,
    )
    parser.add_argument(
        '--print-header',
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if RANK == 0 and args.print_header:
        print(",".join(["backend", "nranks", "boxsize", "npts", "time"]), flush=True)
        sys.exit(0)

    partition = mpipartition.Partition()
    ntot = args.npts // N_RANKS
    # add last little bit to last rank
    if RANK == N_RANKS - 1:
        ntot += args.npts - (ntot * N_RANKS)
    lbox = args.lbox
    zmax = 20
    nbins = 10
    rbins_squared = np.logspace(-1, np.log10(25), nbins + 1) ** 2
    with time_step("generate & distribute data"):
        halo_catalog = _gen_data(ntot, lbox, partition)

    ntest = 3
    res = []
    for i in range(ntest):
        t0 = time.time()
        if args.backend == "cuda":
            _run_cuda(halo_catalog, lbox, zmax, rbins_squared)
        elif args.backend == "cpu":
            _run_cpu(halo_catalog, lbox, zmax, rbins_squared)
        t0 = time.time() - t0
        res.append(t0)
    avg = np.median(res)

    if RANK == 0:
        print(
            ",".join(
                "%r" % item if not isinstance(item, str) else item
                for item in [args.backend, N_RANKS, args.lbox, args.npts, avg]
            ),
            flush=True,
        )
