import argparse

from collections import OrderedDict

import jax
import jax.numpy as jnp

import numpy as np
import cupy as cp

import matplotlib.pyplot as plt

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1

from diffsmhm.galhalo_models.merging import _calculate_indx_to_deposit
from diffsmhm.loader import load_and_chop_data_bolshoi_planck

from diffsmhm.analysis.diff_sm import compute_weight_and_jac
from diffsmhm.analysis.hmc_bounding import (
    model_pos_to_hmc_pos,
    hmc_pos_to_model_pos
)
from diffsmhm.analysis.adam import adam
from diffsmhm.analysis.util import (
    get_default_params,
    get_param_bounds
)

from diffsmhm.diff_stats.mpi.wprp import wprp_mpi_comp_and_reduce
from diffsmhm.diff_stats.cuda.wprp import wprp_mpi_kernel_cuda


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="fit_wprp_all.py",
        description="Fit model to provided wprp with Adam"
    )
    parser.add_argument(
        "-w", "--wprp",
        type=str,
        required=True
    )
    parser.add_argument(
        "-e", "--wprp-error",
        type=str,
        default=None
    )
    parser.add_argument(
        "-r", "--rpbins",
        type=str,
        default=None
    )
    parser.add_argument(
        "--halo-file",
        type=str,
        default="/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"
    )
    parser.add_argument(
        "--particle-file",
        type=str,
        default="/home/jwick/data/hlist_1.00231.particles.halotools_v0p4.hdf5"
    )
    parser.add_argument(
        "--hmcut",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--mass-bin-low",
        type=float,
        default=10.6
    )
    parser.add_argument(
        "--mass-bin-high",
        type=float,
        default=100.0
    )
    parser.add_argument(
        "-t", "--theta-init",
        type=str,
        default=None
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default="./"
    )
    parser.add_argument(
        "--adam-a",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--adam-b1",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "--adam-b2",
        type=float,
        default=0.999
    )
    parser.add_argument(
        "--adam-tmax",
        type=float,
        default=50
    )
    args = parser.parse_args()

    # 1) setup
    # command line args
    outdir = args.outdir
    if outdir[-1] != "/":
        outdir.append("/")

    wprp_info = {}
    if RANK == 0:
        wprp_goal = np.load(args.wprp)
        print("wprp goal:", wprp_goal)

        # this is left optional for the demo, really you should provide this
        wprp_err = 0.1 * wprp_goal
        if args.wprp_error is not None:
            wprp_err = np.load(args.wprp_error)

        # default rpbins is roughly watson rpbins
        rpbins = np.logspace(-1, 1.3, 16, dtype=np.float64)
        if args.rpbins is not None:
            rpbins = np.load(args.rpbins)
        if rpbins[0] != 0:
            rpbins = np.concatenate([np.array([0.0]), rpbins], dtype=np.float64)

        assert len(wprp_goal) == len(rpbins) - 2
        wprp_info = {
                        "wprp": wprp_goal,
                        "wprp_err": wprp_err,
                        "rpbins": rpbins
        }
    wprp_info = COMM.bcast(wprp_info, root=0)
    wprp_goal = wprp_info["wprp"]
    wprp_err = wprp_info["wprp_err"]
    rpbins = wprp_info["rpbins"]

    theta_default = get_default_params()
    lower_bounds, upper_bounds = get_param_bounds()

    # let's load the catalog; note this assumes Bolshoi
    halo_file = args.halo_file
    particle_file = args.particle_file
    box_length = 250.0  # Mpc
    buff_wprp = max(rpbins)+1  # Mpc
    zmax = 20.0

    # jax weights prefer this as an array
    mass_bin_edges = np.array([args.mass_bin_low, args.mass_bin_high], dtype=np.float64)

    theta_init = np.copy(theta_default)
    if args.theta_init is not None:
        theta_init = np.load(args.theta_init)

    n_params = len(theta_init)
    n_rpbins = len(rpbins) - 2
    n_devices = jax.local_device_count()

    hmcut = args.hmcut
    halos, _ = load_and_chop_data_bolshoi_planck(
                particle_file,
                halo_file,
                box_length,
                buff_wprp,
                host_mpeak_cut=hmcut
    )

    idx_to_deposit = _calculate_indx_to_deposit(halos["upid"], halos["halo_id"])

    # jax copy (weights) and cupy copies (wprp) for each GPU
    halos_jax = OrderedDict()
    halos_cp = OrderedDict()
    for k in halos.keys():
        halos_jax[k] = jnp.array(halos[k], dtype=halos[k].dtype)

        halos_cp[k] = []
        for d in range(n_devices):
            cp.cuda.Device(d).use()
            halos_cp[k].append(cp.array(halos[k], dtype=halos[k].dtype))

    halos_cp["rpbins_squared"] = []
    for d in range(n_devices):
        cp.cuda.Device(d).use()
        halos_cp["rpbins_squared"].append(cp.array(rpbins**2, dtype=cp.float64))

    n_calls = np.zeros(1, dtype="i")

    # 2) optimization
    np.set_printoptions(precision=1, floatmode="maxprec_equal")

    # this is declared inside the __main__ so we can only have one input argument
    # for use with Adam
    def mse_wprp_all(theta_unbounded):
        theta_model = hmc_pos_to_model_pos(theta_unbounded, lower_bounds, upper_bounds)
        theta_model = np.array(theta_model, dtype=np.float64)  # no jax for Bcast
        COMM.Bcast([theta_model, MPI.DOUBLE], root=0)

        w, dw = compute_weight_and_jac(
                    logmpeak=halos_jax["logmpeak"],
                    loghost_mpeak=halos_jax["loghost_mpeak"],
                    log_vmax_by_vmpeak=halos_jax["logvmax_frac"],
                    upid=halos_jax["upid"],
                    idx_to_deposit=idx_to_deposit,
                    mass_bin_low=mass_bin_edges[0],
                    mass_bin_high=mass_bin_edges[1],
                    theta=theta_model
        )

        # dlpack is crucial here otherwise copying may not work
        # also note explicit copy bc we're moving between GPUs
        w_list = []
        dw_list = []
        for d in range(n_devices):
            cp.cuda.Device(d).use()
            w_list.append(cp.from_dlpack(jax.dlpack.to_dlpack(w, copy=True)))
            dw_list.append(cp.from_dlpack(jax.dlpack.to_dlpack(dw, copy=True)))

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

        error = 0.5 * np.sum(((wprp - wprp_goal) / wprp_err)**2)
        error_grad = 0.5 * np.sum((2 * wprp_grad * (wprp - wprp_goal)) / (wprp_err**2),
                                  axis=1)

        percent_error = 100 * (wprp - wprp_goal) / wprp_goal

        if n_calls % 100 == 0:
            print(f"{n_calls[0]:04}", ":", f"{error:.4f}", percent_error, flush=True)

        n_calls[0] += 1
        return error, error_grad

    # rank 0 drives optimization, others just compute
    theta_opt = np.zeros_like(theta_init)
    if RANK == 0:
        theta_init_unbounded = model_pos_to_hmc_pos(theta_init, lower_bounds,
                                                    upper_bounds)
        theta_opt, error_history = adam(
                                    a=args.adam_a,
                                    b1=args.adam_b1,
                                    b2=args.adam_b2,
                                    opt_params=theta_init_unbounded,
                                    err_func=mse_wprp_all,
                                    maxiter=10000,
                                    tmax=args.adam_tmax*60
        )
        theta_opt = np.array(
                        hmc_pos_to_model_pos(theta_opt, lower_bounds, upper_bounds),
                        dtype=np.float64
        )
        print("theta opt:", theta_opt)

        # stop the other ranks
        stop = -1 * np.ones(n_params, dtype=np.float64)
        COMM.Bcast([stop, MPI.DOUBLE], root=0)

    else:
        while True:
            # receive
            theta_model = np.empty(n_params, dtype=np.float64)
            COMM.Bcast([theta_model, MPI.DOUBLE], root=0)

            if theta_model[0] < 0:
                break

            w, dw = compute_weight_and_jac(
                        logmpeak=halos_jax["logmpeak"],
                        loghost_mpeak=halos_jax["loghost_mpeak"],
                        log_vmax_by_vmpeak=halos_jax["logvmax_frac"],
                        upid=halos_jax["upid"],
                        idx_to_deposit=idx_to_deposit,
                        mass_bin_low=mass_bin_edges[0],
                        mass_bin_high=mass_bin_edges[1],
                        theta=theta_model
            )

            # again, dlpack and explicit copy is important
            w_list = []
            dw_list = []
            for d in range(n_devices):
                cp.cuda.Device(d).use()
                w_list.append(cp.from_dlpack(jax.dlpack.to_dlpack(w, copy=True)))
                dw_list.append(cp.from_dlpack(jax.dlpack.to_dlpack(dw, copy=True)))

            _, _ = wprp_mpi_comp_and_reduce(
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

    # 3) make a figure
    np.set_printoptions(precision=5, floatmode="maxprec_equal")
    if RANK == 0:
        print("opt done")
        print("theta_opt:", theta_opt)
        print("error change:", error_history[0], error_history[-1], flush=True)

    # figure for initial and final wprp
    # compute initial wprp
    w, dw = compute_weight_and_jac(
                logmpeak=halos_jax["logmpeak"],
                loghost_mpeak=halos_jax["loghost_mpeak"],
                log_vmax_by_vmpeak=halos_jax["logvmax_frac"],
                upid=halos_jax["upid"],
                idx_to_deposit=idx_to_deposit,
                mass_bin_low=mass_bin_edges[0],
                mass_bin_high=mass_bin_edges[1],
                theta=theta_init
    )

    w_list = []
    dw_list = []
    for d in range(n_devices):
        cp.cuda.Device(d).use()
        w_list.append(cp.from_dlpack(jax.dlpack.to_dlpack(w, copy=True)))
        dw_list.append(cp.from_dlpack(jax.dlpack.to_dlpack(dw, copy=True)))

    wprp_init, _ = wprp_mpi_comp_and_reduce(
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

    # final wprp
    COMM.Bcast([theta_opt, MPI.DOUBLE], root=0)
    w, dw = compute_weight_and_jac(
                logmpeak=halos_jax["logmpeak"],
                loghost_mpeak=halos_jax["loghost_mpeak"],
                log_vmax_by_vmpeak=halos_jax["logvmax_frac"],
                upid=halos_jax["upid"],
                idx_to_deposit=idx_to_deposit,
                mass_bin_low=mass_bin_edges[0],
                mass_bin_high=mass_bin_edges[1],
                theta=theta_opt
    )

    w_list = []
    dw_list = []
    for d in range(n_devices):
        cp.cuda.Device(d).use()
        w_list.append(cp.from_dlpack(jax.dlpack.to_dlpack(w, copy=True)))
        dw_list.append(cp.from_dlpack(jax.dlpack.to_dlpack(dw, copy=True)))

    wprp_final, _ = wprp_mpi_comp_and_reduce(
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

    # rank 0 makes the figure
    if RANK == 0:
        print("wprp final:", wprp_final, flush=True)
        rpbins = cp.asnumpy(rpbins)[1:-1]

        # figure for wprp optimization
        fig = plt.figure(figsize=(10, 8), facecolor="w")

        plt.plot(rpbins, wprp_init * rpbins, linewidth=3, c="tab:blue")
        plt.plot(rpbins, wprp_final * rpbins, linewidth=2, c="tab:orange")
        plt.plot(rpbins, wprp_goal * rpbins, linewidth=1, c="k")

        plt.xlabel("rp", fontsize=16)
        plt.ylabel("rp wp(rp)", fontsize=16)

        plt.legend(["start params", "opt params", "goal"])

        plt.xscale("log")

        plt.savefig(outdir+"fit_wprp_all_wprp.png")

        # figure for error history
        fig = plt.figure(figsize=(10, 8), facecolor="w")
        plt.plot(error_history)

        plt.yscale("log")

        plt.xlabel("iteration number")
        plt.ylabel("log error")
        plt.savefig(outdir+"fit_wprp_all_error.png")

        # also save the final theta
        np.save(outdir+"theta_opt.npy", theta_opt)
