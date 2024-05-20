import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import custom_vjp

import blackjax

import pandas as pd
import pickle

from collections import OrderedDict

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1

import time

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

from diffsmhm.loader import load_and_chop_data_bolshoi_planck
from diffsmhm.galhalo_models.merging import _calculate_indx_to_deposit

from diffsmhm.analysis.tools.diff_sm import compute_weight_and_jac

from diffsmhm.diff_stats.mpi.wprp import wprp_mpi_comp_and_reduce
from diffsmhm.diff_stats.cuda.wprp import wprp_mpi_kernel_cuda

from diffsmhm.analysis.scripts.hmc_bounding import (
    hmc_pos_to_model_pos,
    model_pos_to_hmc_pos,
    logdens_model_to_logdens_hmc
)

# data files and rpwp params
halo_file="/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"
particle_file="/home/jwick/data/hlist_1.00231.particles.halotools_v0p4.hdf5"
box_length = 250.0 # Mpc
buff_wprp = 20.0 # Mpc

mass_bin_edges = np.array([10.0, 11.0], dtype=np.float64)

rpbins = np.logspace(-1, 1.2, 5, dtype=np.float64)
zmax = 20.0 # Mpc

theta = np.array(list(smhm_params.values()) + 
                 list(smhm_sigma_params.values()) + 
                 list(disruption_params.values()), dtype=np.float64
)
theta_init = np.copy(theta)

n_params = len(theta)
n_rpbins = len(rpbins) - 1

# hmc params
n_warmup_steps = 10
n_iter = 10
#
hmcut = 14.6

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
    disruption_bounds["satmerg_logvr_k"][0]
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
    disruption_bounds["satmerg_logvr_k"][1]
], dtype=np.float64)

# 1) load data
halos, _ = load_and_chop_data_bolshoi_planck(
                particle_file,
                halo_file,
                box_length,
                buff_wprp,
                host_mpeak_cut=hmcut
)

idx_to_deposit = _calculate_indx_to_deposit(halos["upid"], halos["halo_id"])

print(RANK, "data loaded", len(idx_to_deposit), flush=True)

# 2) obtain "goal" measurement
np.random.seed(999)
parameter_perturbations = np.random.uniform(low=0.98, high=1.02, size=n_params)

theta_goal = theta * parameter_perturbations
if RANK == 0:
    print("theta goal:", theta_goal)

# rpwp, quenched and unquenched
t0 = time.time()
w, dw = compute_weight_and_jac(
            logmpeak=halos["logmpeak"],
            loghost_mpeak=halos["loghost_mpeak"],
            log_vmax_by_vmpeak=halos["logvmax_frac"],
            upid=halos["upid"],
            idx_to_deposit=idx_to_deposit,
            mass_bin_low=mass_bin_edges[0],
            mass_bin_high=mass_bin_edges[1],
            theta=theta_goal
)
t1 = time.time()
print("weights done:", t1-t0)

# mask out gals with zero weight and zero weight grad
mask_wgt = w != 0.0
mask_dwgt = np.sum(np.abs(dw),axis=0) != 0.0
full_mask = mask_wgt & mask_dwgt

print(RANK, "n gals:", len(w[full_mask]), flush=True)

t0 = time.time()
wprp_goal, _ = wprp_mpi_comp_and_reduce(
                x1=np.asarray(halos["halo_x"][full_mask].astype(np.float64)),
                y1=np.asarray(halos["halo_y"][full_mask].astype(np.float64)),
                z1=np.asarray(halos["halo_z"][full_mask].astype(np.float64)),
                w1=np.asarray(w[full_mask].astype(np.float64)),
                w1_jac=np.asarray(dw[:, full_mask].astype(np.float64)),
                inside_subvol=np.asarray(halos["_inside_subvol"][full_mask]),
                rpbins_squared=rpbins**2,
                zmax=zmax,
                boxsize=box_length,
                kernel_func=wprp_mpi_kernel_cuda
)
t1 = time.time()

if RANK == 0:
    print("goal wprp done in:", t1-t0, flush=False)
    print("goal wprp:", wprp_goal, flush=True)

# 3) do optimization
ncalls = np.array([0], dtype="i")

# define functions

# transform functions; following STAN manual for bounded scalar

# logdensity function set upi
# note that only rank zero will deal with this func
def logdensity(
    theta_hmc,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    wprp_goal=wprp_goal
):

    # transform HMC params into model params
    theta_model = hmc_pos_to_model_pos(theta_hmc, lower_bounds, upper_bounds)

    # call func for error/potential
    U = get_potential(theta_model)

    # convert potential to logdensity
    log_density_model = -1.0 * U 

    # transform logdensity of model into hmc logdensity 
    log_density_hmc = logdens_model_to_logdens_hmc(
                        log_density_model,
                        theta_hmc,
                        lower_bounds,
                        upper_bounds
    )

    return log_density_hmc


# method to calculate quenched and unquenched rpwp 
# this is what we pure_callback to; it returns both value and gradient
def potential(
    theta,
    wprp_goal=wprp_goal,
    log_halomass=halos["logmpeak"],
    log_host_halomass=halos["loghost_mpeak"],
    log_vmax_by_vmpeak=halos["logvmax_frac"],
    halo_x=halos["halo_x"], halo_y=halos["halo_y"], halo_z=halos["halo_z"],
    upid=halos["upid"],
    idx_to_deposit=idx_to_deposit,
    inside_subvol=halos["_inside_subvol"],
    mass_bin_low=mass_bin_edges[0], mass_bin_high=mass_bin_edges[1],
    rpbins=rpbins,
    zmax=zmax,
    boxsize=box_length
):
    ncalls[0] = ncalls[0]+1

    # broadcast theta
    COMM.Bcast([theta, MPI.DOUBLE], root=0)

    w, dw = compute_weight_and_jac(
                logmpeak=log_halomass,
                loghost_mpeak=log_host_halomass,
                log_vmax_by_vmpeak=log_vmax_by_vmpeak,
                upid=upid,
                idx_to_deposit=idx_to_deposit,
                mass_bin_low=mass_bin_edges[0],
                mass_bin_high=mass_bin_edges[1],
                theta=theta
    )

    # mask out gals with zero weight and zero weight grad
    mask_wgt = w != 0.0
    mask_dwgt = np.sum(np.abs(dw), axis=0) != 0.0
    full_mask = mask_wgt & mask_dwgt

    if len(halo_x[full_mask]) == 0:
        print(RANK, "zero condition", flush=True)

    try:
        wprp, wprp_grad = wprp_mpi_comp_and_reduce(
                            x1=np.asarray(halo_x[full_mask].astype(np.float64)),
                            y1=np.asarray(halo_y[full_mask].astype(np.float64)),
                            z1=np.asarray(halo_z[full_mask].astype(np.float64)),
                            w1=np.asarray(w[full_mask].astype(np.float64)),
                            w1_jac=np.asarray(dw[:, full_mask].astype(np.float64)),
                            inside_subvol=np.asarray(inside_subvol[full_mask]),
                            rpbins_squared=rpbins**2,
                            zmax=zmax,
                            boxsize=boxsize,
                            kernel_func=wprp_mpi_kernel_cuda
        )
    except:
        print("ZERO WEIGHT CONDITION")
        print(theta, flush=True)
        COMM.Bcast([-1*np.ones_like(theta, dtype=np.float64), MPI.DOUBLE], root=0)
        print("EXITING")
        exit(1)

    cov = 0.1 * wprp_goal

    error = 0.5 * np.sum(((wprp - wprp_goal) / cov)**2)
    error_grad = 0.5 * np.sum((2 * wprp_grad * (wprp-wprp_goal)) / (cov**2), axis=1)

    return error, error_grad


# this returns only val
@custom_vjp
def get_potential(
    theta
):
    val, grad = jax.pure_callback(
                    potential,
                    (np.array(1.0, dtype=np.float64), np.ones(14, dtype=np.float64)),
                    theta
    )

    return val 

def vjp_fwd(
    theta
):
    # not elegant, but inputs here need to match potential hence literal length
    val, grad = jax.pure_callback(
                    potential,
                    (np.array(1.0, dtype=np.float64), np.ones(14, dtype=np.float64)),
                    theta
    )

    return val, grad

def vjp_bwd(grad, tan):
    # note that blackjax expects a tuple here
    return (grad * tan,)

get_potential.defvjp(vjp_fwd, vjp_bwd)

# this is where we split up the ranks

# sampler set up 

# non 0 mpi ranks just loop until told to stop
if RANK > 0:
    while True:

        # receive current theta
        theta = np.empty(n_params, dtype=np.float64)
        COMM.Bcast([theta, MPI.DOUBLE], root=0)

        if theta[0] < 0:
            break

        # do computation with jax
        w, dw = compute_weight_and_jac(
                    logmpeak=halos["logmpeak"],
                    loghost_mpeak=halos["loghost_mpeak"],
                    log_vmax_by_vmpeak=halos["logvmax_frac"],
                    upid=halos["upid"],
                    idx_to_deposit=idx_to_deposit,
                    mass_bin_low=mass_bin_edges[0],
                    mass_bin_high=mass_bin_edges[1],
                    theta=theta
        )

        # mask out gals with zero weight and zero weight grad
        mask_wgt = w != 0.0
        mask_dwgt = np.sum(np.abs(dw), axis=0) != 0.0
        full_mask = mask_wgt & mask_dwgt

        if len(halos["halo_x"][full_mask]) == 0:
            print(RANK, "zero condition", flush=True)

        # do computation with CUDA
        _, _ = wprp_mpi_comp_and_reduce(
                    x1=np.asarray(halos["halo_x"][full_mask].astype(np.float64)),
                    y1=np.asarray(halos["halo_y"][full_mask].astype(np.float64)),
                    z1=np.asarray(halos["halo_z"][full_mask].astype(np.float64)),
                    w1=np.asarray(w[full_mask].astype(np.float64)),
                    w1_jac=np.asarray(dw[:, full_mask].astype(np.float64)),
                    inside_subvol=np.asarray(halos["_inside_subvol"][full_mask]),
                    rpbins_squared=rpbins**2,
                    zmax=zmax,
                    boxsize=box_length,
                    kernel_func=wprp_mpi_kernel_cuda
        )

    if RANK == 1:
        print("while exited")

# RANK 0 continues with the HMC
else:
    rng_key = jax.random.PRNGKey(42)

    # transform theta for the initial position
    initial_position = model_pos_to_hmc_pos(theta_init, lower_bounds, upper_bounds)

    # split the key
    rng_key, sample_key, warmup_key = jax.random.split(rng_key, 3)

    print("Begin warmup", flush=True)
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity,
                                        initial_step_size=1e-4,
                                        is_mass_matrix_diagonal=True,
                                        progress_bar=True)
    t0 = time.time()
    (init_state, tuned_params), _ = warmup.run(warmup_key, initial_position,
                                               num_steps=n_warmup_steps)
    t1 = time.time()

    ncalls_w = ncalls[0]
    print("Warm up done", "t:", t1-t0, "n:", ncalls_w, flush=False)
    print(init_state)
    print(tuned_params, flush=True)

    hmc = blackjax.nuts(logdensity, **tuned_params)

    init_pos = model_pos_to_hmc_pos(theta_goal, lower_bounds, upper_bounds)
    init_state_d = hmc.init(init_pos)

    # build kernel and inference loop
    hmc_kernel = jax.jit(hmc.step)

    # run inference loop
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    t0 = time.time()
    states = inference_loop(sample_key, hmc_kernel, init_state_d, n_iter)
    t1 = time.time()

    ncalls_h = ncalls[0] - ncalls_w
    print("hmc done", "t:", t1-t0, "n:", ncalls_h, flush=True)

    # we're done iterating, broadcast FALSE to stop the other ranks
    cont = -1*np.ones_like(theta)
    COMM.Bcast(cont, root=0)
    print("opt done", flush=True)

