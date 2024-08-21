from collections import OrderedDict

import sys

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

from diffsmhm.galhalo_models.sigmoid_smhm import (
    DEFAULT_PARAM_VALUES as smhm_params,
    PARAM_BOUNDS as smhm_bounds
)
from diffsmhm.galhalo_models.sigmoid_smhm_sigma import (
    DEFAULT_PARAM_VALUES as smhm_sigma_params,
    PARAM_BOUNDS as smhm_sigma_bounds
)
from diffsmhm.galhalo_models.sigmoid_disruption import (
    DEFAULT_PARAM_VALUES as disruption_params,
    PARAM_BOUNDS as disruption_bounds
)

from diffsmhm.galhalo_models.merging import _calculate_indx_to_deposit
from diffsmhm.loader import load_and_chop_data_bolshoi_planck

from diffsmhm.analysis.diff_sm import compute_weight_and_jac
from diffsmhm.analysis.hmc_bounding import (
    model_pos_to_hmc_pos,
    hmc_pos_to_model_pos
)
from diffsmhm.analysis.adam import adam

from diffsmhm.diff_stats.mpi.wprp import wprp_mpi_comp_and_reduce
from diffsmhm.diff_stats.cuda.wprp import wprp_mpi_kernel_cuda


# 1) setup
# command line args
if len(sys.argv) != 2:
    if RANK == 0:
        print("Usage: python fit_watson_adam.py [DATAFILE]")
    exit(1)

wprp_file = sys.argv[1]
mass_bin_low = 9.8
if "10.2" in wprp_file:
    mass_bin_low = 10.2
elif "10.6" in wprp_file:
    mass_bin_low = 10.6


def read_watson_wp(fname, h=0.7):
    dw = np.genfromtxt(fname, usecols=[1, 2, 3], names=["r", "wp", "wperr"])
    dw["r"] /= h
    dw["wp"] /= h
    dw["wperr"] /= h
    return dw


# TODO: check h for bolshoi
watson_data = {}
if RANK == 0:
    watson_data = read_watson_wp(wprp_file, h=1.0)
watson_data = COMM.bcast(watson_data, root=0)

rpbins = cp.array(watson_data["r"], dtype=cp.float64)
rpbins = cp.concatenate([cp.array([0.0]), rpbins, cp.array([20.0])], dtype=cp.float64)
wprp_goal = np.array(watson_data["wp"], dtype=np.float64)
wprp_err = np.array(watson_data["wperr"], dtype=np.float64)
if RANK == 0:
    print(wprp_goal, flush=True)

assert rpbins[0] == 0.0
assert len(wprp_goal) == len(rpbins) - 2

# load param bounds
lower_bounds = np.array([
    smhm_bounds["smhm_logm_crit"][0],
    smhm_bounds["smhm_ratio_logm_crit"][0],
    smhm_bounds["smhm_k_logm"][0],
    smhm_bounds["smhm_lowm_index"][0],
    smhm_bounds["smhm_highm_index"][0],
    smhm_sigma_bounds["smhm_sigma_low"][0],
    smhm_sigma_bounds["smhm_sigma_high"][0],
    smhm_sigma_bounds["smhm_sigma_logm_pivot"][0],
    smhm_sigma_bounds["smhm_sigma_logm_width"][0],
    disruption_bounds["satmerg_logmhost_crit"][0],
    disruption_bounds["satmerg_logmhost_k"][0],
    disruption_bounds["satmerg_logvr_crit_dwarfs"][0],
    disruption_bounds["satmerg_logvr_crit_clusters"][0],
    disruption_bounds["satmerg_logvr_k"][0],
], dtype=np.float64)
upper_bounds = np.array([
    smhm_bounds["smhm_logm_crit"][1],
    smhm_bounds["smhm_ratio_logm_crit"][1],
    smhm_bounds["smhm_k_logm"][1],
    smhm_bounds["smhm_lowm_index"][1],
    smhm_bounds["smhm_highm_index"][1],
    smhm_sigma_bounds["smhm_sigma_low"][1],
    smhm_sigma_bounds["smhm_sigma_high"][1],
    smhm_sigma_bounds["smhm_sigma_logm_pivot"][1],
    smhm_sigma_bounds["smhm_sigma_logm_width"][1],
    disruption_bounds["satmerg_logmhost_crit"][1],
    disruption_bounds["satmerg_logmhost_k"][1],
    disruption_bounds["satmerg_logvr_crit_dwarfs"][1],
    disruption_bounds["satmerg_logvr_crit_clusters"][1],
    disruption_bounds["satmerg_logvr_k"][1],
], dtype=np.float64)


# let's load the catalog
halo_file = "/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"
particle_file = "/home/jwick/data/hlist_1.00231.particles.halotools_v0p4.hdf5"
box_length = 250.0  # Mpc
buff_wprp = 20.0  # Mpc
zmax = 20.0

# jax weights prefer this as an array; watson bins are all only lower bounded
mass_bin_edges = np.array([mass_bin_low, 100.0], dtype=np.float64)

theta_init = np.array(list(smhm_params.values()) +
                      list(smhm_sigma_params.values()) +
                      list(disruption_params.values()), dtype=np.float64)
"""
theta_init = np.array([
    10.50000, -2.20148, 2.00000, 1.50000, 0.78571,
    0.10000, 0.10000, 11.00260, 0.19934,
    14.91979, 6.06973, 0.00000, -2.00000, 10.00000
], dtype=np.float64)
"""

n_params = len(theta_init)
n_rpbins = len(rpbins) - 2
n_devices = jax.local_device_count()

hmcut = 0.0
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


def mse_wprp_all(theta_unbounded):
    theta_model = hmc_pos_to_model_pos(theta_unbounded, lower_bounds, upper_bounds)
    theta_model = np.array(theta_model, dtype=np.float64)  # ensure not jax for Bcast
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

    if n_calls % 10 == 0:
        print(f"{n_calls[0]:04}", ":", f"{error:.4f}", percent_error, flush=True)

    n_calls[0] += 1
    return error, error_grad


# rank 0 drives optimization, others just compute
theta_opt = np.zeros_like(theta_init)
if RANK == 0:
    theta_init_unbounded = model_pos_to_hmc_pos(theta_init, lower_bounds, upper_bounds)
    theta_opt, error_history = adam(
                                a=0.01,
                                b1=0.99,
                                b2=0.9999,
                                static_params=None,
                                opt_params=theta_init_unbounded,
                                err_func=mse_wprp_all,
                                maxiter=10000,
                                tmax=50*60
    )
    theta_opt = hmc_pos_to_model_pos(theta_opt, lower_bounds, upper_bounds)

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

# 3) plotting
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
COMM.Bcast([theta_init, MPI.DOUBLE], root=0)
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

    plt.savefig("figures/fit_watson_all_wprp.png")

    # figure for error history
    fig = plt.figure(figsize=(10, 8), facecolor="w")
    plt.plot(error_history)

    plt.yscale("log")

    plt.xlabel("iteration number")
    plt.ylabel("log error")
    plt.savefig("figures/fit_watson_all_error.png")
