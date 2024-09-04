import argparse

import numpy as np
import cupy as cp
from collections import OrderedDict

import jax
import jax.numpy as jnp

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1

from diffsmhm.loader import load_and_chop_data_bolshoi_planck
from diffsmhm.galhalo_models.merging import _calculate_indx_to_deposit

from diffsmhm.analysis.diff_sm import compute_weight_and_jac

from diffsmhm.diff_stats.mpi.wprp import wprp_mpi_comp_and_reduce
from diffsmhm.diff_stats.cuda.wprp import wprp_mpi_kernel_cuda

from diffsmhm.analysis.util import get_default_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="compute_one_wprp.py",
        description="Compute a single wprp with specified parameters"
    )
    parser.add_argument(
        "--halo_file",
        type=str,
        default="/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"
    )
    parser.add_argument(
        "--particle_file",
        type=str,
        default="/home/jwick/data/hlist_1.00231.particles.halotools_v0p4.hdf5"
    )
    parser.add_argument(
        "--mass_bin_low",
        type=float,
        default=10.6
    )
    parser.add_argument(
        "-mass_bin_high",
        type=float,
        default=100.0
    )
    parser.add_argument(
        "-r", "--rpbins",
        type=str,
        default=None
    )
    parser.add_argument(
        "-t", "--theta",
        type=str,
        default=None
    )
    parser.add_argument(
        "-p", "--perturbation_limit",
        type=float,
        default=0.00
    )
    parser.add_argument(
        "-s", "--perturbation_seed",
        type=int,
        default=999
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default="./"
    )
    parser.add_argument(
        "--hmcut",
        type=float,
        default=0.0
    )
    args = parser.parse_args()

    outdir = args.outdir
    if outdir[-1] != "/":
        outdir.append("/")

    # load data; note that we assume bolshoi
    box_length = 250.0  # Mpc
    buff_wprp = 20.0  # Mpc

    halos, _ = load_and_chop_data_bolshoi_planck(
                args.particle_file,
                args.halo_file,
                box_length,
                buff_wprp,
                host_mpeak_cut=args.hmcut
    )

    # other parameters
    # jax wants an array for these when we do weights
    mass_bin_edges = np.array([args.mass_bin_low, args.mass_bin_high], dtype=np.float64)

    rpbins = cp.logspace(-1, 1.3, 16, dtype=np.float64)
    if args.rpbins is not None:
        rpbins = np.load(args.rpbins)
    if rpbins[0] > 0:
        rpbins = cp.concatenate([cp.array([0]), rpbins])

    zmax = 20.0

    theta_default = get_default_params()

    n_params = len(theta_default)
    n_rpbins = len(rpbins)-1
    n_devices = jax.local_device_count()

    # perturb theta unless file to load is specified
    np.random.seed(args.perturbation_seed)
    parameter_perturbations = np.random.uniform(1-args.perturbation_limit,
                                                1+args.perturbation_limit,
                                                n_params)
    theta = theta_default * parameter_perturbations
    if args.theta is not None:
        theta = np.load(args.theta)
    if RANK == 0:
        print("theta:", theta, flush=True)

    idx_to_deposit = _calculate_indx_to_deposit(halos["upid"], halos["halo_id"])
    idx_to_deposit = jnp.copy(idx_to_deposit)

    # make a jax and a cupy version of the catalog
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

    w, dw = compute_weight_and_jac(
                logmpeak=halos_jax["logmpeak"],
                loghost_mpeak=halos_jax["loghost_mpeak"],
                log_vmax_by_vmpeak=halos_jax["logvmax_frac"],
                upid=halos_jax["upid"],
                idx_to_deposit=idx_to_deposit,
                mass_bin_low=mass_bin_edges[0],
                mass_bin_high=mass_bin_edges[1],
                theta=jax.device_put(theta, jax.devices()[0])
    )

    w_list = []
    dw_list = []
    for d in range(n_devices):
        cp.cuda.Device(d).use()
        w_list.append(cp.array(w))
        dw_list.append(cp.array(dw))

    wprp, wprp_grad = wprp_mpi_comp_and_reduce(
                        x1=halos_cp["halo_x"],
                        y1=halos_cp["halo_y"],
                        z1=halos_cp["halo_z"],
                        w1=w_list,
                        w1_jac=dw_list,
                        inside_subvol=halos_cp["_inside_subvol"],
                        rpbins_squared=halos_cp["rpbins_squared"],
                        zmax=zmax,
                        boxsize=box_length,
                        kernel_func=wprp_mpi_kernel_cuda
    )

    if RANK == 0:
        print("WPRP:", wprp)

        # save to file
        fpath_wprp = outdir+"wprp_single.npy"
        fpath_rpbins = outdir+"rpbins_single.npy"

        np.save(fpath_wprp, wprp)
        np.save(fpath_rpbins, rpbins)
