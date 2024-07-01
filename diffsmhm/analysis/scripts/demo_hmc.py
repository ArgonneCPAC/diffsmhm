import numpy as np
import cupy as cp

import jax
import jax.numpy as jnp
from jax import custom_vjp

import blackjax

import pandas as pd

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

from diffsmhm.analysis.tools.fim import (
    make_FIM
)

# data files and rpwp params
halo_file = "/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"
particle_file = "/home/jwick/data/hlist_1.00231.particles.halotools_v0p4.hdf5"
box_length = 250.0
buff_wprp = 20.0

mass_bin_edges = np.array([10.0, 11.0], dtype=np.float64)

rpbins = cp.logspace(-1, 1.3, 16, dtype=np.float64)
zmax = 20.0

theta = np.array(list(smhm_params.values()) +
                 list(smhm_sigma_params.values()) +
                 list(disruption_params.values()), dtype=np.float64)
theta_init = np.copy(theta)

n_params = len(theta)
n_rpbins = len(rpbins) - 1

# hmc params
load_warmup = True
load_hmc = False
n_iter = 500
hmc_step_size = 2.5e-1
#
hmcut = 13.3
# 14.6, 14.0, 13.3, 9.0, 0.0

# output files
suffix = str(hmcut)+"_"+str(mass_bin_edges[0])+"_"+str(mass_bin_edges[1])
fpath_fim = "output/fim_"+suffix+".npy"
fpath_positions = "output/positions_"+suffix+".csv"

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
idx_to_deposit = jnp.copy(idx_to_deposit)

# make a jax version of the catalog for the weights
# make a cupy version of the catalog for the kernels?
halos_jax = OrderedDict()
halos_cp = OrderedDict()
for k in halos.keys():
    halos_jax[k] = jnp.copy(halos[k])
    halos_cp[k] = cp.array(halos[k])

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
            logmpeak=halos_jax["logmpeak"],
            loghost_mpeak=halos_jax["loghost_mpeak"],
            log_vmax_by_vmpeak=halos_jax["logvmax_frac"],
            upid=halos_jax["upid"],
            idx_to_deposit=idx_to_deposit,
            mass_bin_low=mass_bin_edges[0],
            mass_bin_high=mass_bin_edges[1],
            theta=theta_goal
)
t1 = time.time()
print("weights done:", t1-t0)

# mask out gals with zero weight and zero weight grad
mask_wgt = w != 0.0
mask_dwgt = cp.sum(cp.abs(dw), axis=0) != 0.0
full_mask = mask_wgt & mask_dwgt

print(RANK, "n gals:", len(w[full_mask]), flush=True)

t0 = time.time()
wprp_goal, _ = wprp_mpi_comp_and_reduce(
                x1=cp.asarray(halos_cp["halo_x"][full_mask]),
                y1=cp.asarray(halos_cp["halo_y"][full_mask]),
                z1=cp.asarray(halos_cp["halo_z"][full_mask]),
                w1=cp.asarray(w[full_mask]),
                w1_jac=cp.asarray(dw[:, full_mask]),
                inside_subvol=cp.asarray(halos_cp["_inside_subvol"][full_mask]),
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
    log_halomass=halos_jax["logmpeak"],
    log_host_halomass=halos_jax["loghost_mpeak"],
    log_vmax_by_vmpeak=halos_jax["logvmax_frac"],
    halo_x=halos_jax["halo_x"],
    halo_y=halos_jax["halo_y"],
    halo_z=halos_jax["halo_z"],
    upid=halos_jax["upid"],
    idx_to_deposit=idx_to_deposit,
    inside_subvol=halos_jax["_inside_subvol"],
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
                mass_bin_low=mass_bin_low,
                mass_bin_high=mass_bin_high,
                theta=jax.device_put(theta, jax.devices()[0])
    )

    # mask out gals with zero weight and zero weight grad
    mask_wgt = w != 0.0
    mask_dwgt = cp.sum(cp.abs(dw), axis=0) != 0.0
    full_mask = mask_wgt & mask_dwgt

    if len(w[full_mask]) == 0:
        print("ZERO WEIGHT CONDITION")
        print(theta, flush=True)
        COMM.Bcast([-1*np.ones_like(theta, dtype=np.float64), MPI.DOUBLE], root=0)
        print("EXITING")
        exit(1)

    wprp, wprp_grad = wprp_mpi_comp_and_reduce(
                        x1=cp.asarray(halos_cp["halo_x"][full_mask]),
                        y1=cp.asarray(halos_cp["halo_y"][full_mask]),
                        z1=cp.asarray(halos_cp["halo_z"][full_mask]),
                        w1=cp.asarray(w[full_mask]),
                        w1_jac=cp.asarray(dw[:, full_mask]),
                        inside_subvol=cp.asarray(halos_cp["_inside_subvol"][full_mask]),
                        rpbins_squared=rpbins**2,
                        zmax=zmax,
                        boxsize=boxsize,
                        kernel_func=wprp_mpi_kernel_cuda
    )

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
                    logmpeak=halos_jax["logmpeak"],
                    loghost_mpeak=halos_jax["loghost_mpeak"],
                    log_vmax_by_vmpeak=halos_jax["logvmax_frac"],
                    upid=halos_jax["upid"],
                    idx_to_deposit=idx_to_deposit,
                    mass_bin_low=mass_bin_edges[0],
                    mass_bin_high=mass_bin_edges[1],
                    theta=theta
        )

        # mask out gals with zero weight and zero weight grad
        mask_wgt = w != 0.0
        mask_dwgt = cp.sum(cp.abs(dw), axis=0) != 0.0
        full_mask = mask_wgt & mask_dwgt
        full_mask = full_mask.get()

        if len(halos_cp["halo_x"][full_mask]) == 0:
            print(RANK, "zero condition", flush=True)

        # do computation with CUDA
        _, _ = wprp_mpi_comp_and_reduce(
                    x1=cp.asarray(halos_cp["halo_x"][full_mask]),
                    y1=cp.asarray(halos_cp["halo_y"][full_mask]),
                    z1=cp.asarray(halos_cp["halo_z"][full_mask]),
                    w1=cp.asarray(w[full_mask]),
                    w1_jac=cp.asarray(dw[:, full_mask]),
                    inside_subvol=halos_cp["_inside_subvol"][full_mask],
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
    rng_key, sample_key = jax.random.split(rng_key, 2)

    # make FIM as mass matrix or load
    if not load_warmup:
        ld_grad = jax.jacrev(logdensity)
        theta_fim = model_pos_to_hmc_pos(
                        theta_goal, lower_bounds, upper_bounds
        )
        FIM = make_FIM(ld_grad, theta_fim)
        np.save(fpath_fim, FIM)
    else:
        # load if specified
        FIM = np.load(fpath_fim)

    # load prior chain if specified
    if load_hmc:
        mpos = np.array(pd.read_csv(fpath_positions).iloc[-1:].values[0],
                        dtype=np.float64)
        init_pos = model_pos_to_hmc_pos(mpos, lower_bounds, upper_bounds)
    else:
        init_pos = model_pos_to_hmc_pos(theta_goal, lower_bounds, upper_bounds)

    hmc = blackjax.nuts(logdensity, hmc_step_size, FIM)
    init_state = hmc.init(init_pos)

    # build kernel and inference loop
    hmc_kernel = jax.jit(hmc.step)

    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state, info)

        keys = jax.random.split(rng_key, num_samples)
        _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

        infos_tuple = (
                        infos.acceptance_rate,
                        infos.is_divergent,
                        infos.num_integration_steps,
        )

        return states, infos_tuple

    t0 = time.time()
    states, infos = inference_loop(sample_key, hmc_kernel, init_state, n_iter)
    t1 = time.time()

    print("hmc done", "t:", t1-t0, "n:", ncalls[0])
    print("acceptance rate:", np.mean(infos[0]), flush=True)

    # we're done iterating, broadcast FALSE to stop the other ranks
    cont = -1*np.ones_like(theta)
    COMM.Bcast(cont, root=0)
    print("opt done", flush=True)

    # save output
    positions = states.position
    positions_df = pd.DataFrame.from_dict(positions)
    # transform from HMC positions to model positions
    for p, key in enumerate(positions_df.keys()):
        positions_df[key] = hmc_pos_to_model_pos(np.array(positions_df[key]),
                                                 lower_bounds[p], upper_bounds[p])

    if load_hmc:
        positions_df.to_csv(fpath_positions, mode="a", header=False, index=False)
    else:
        positions_df.to_csv(fpath_positions, mode="w", header=True, index=False)
