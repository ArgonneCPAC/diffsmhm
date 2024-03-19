import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import custom_vjp

import blackjax

import pandas as pd

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
from diffsmhm.analysis.tools.rpwp import compute_rpwp

from diffsmhm.analysis.tools.error import mse_rpwp


# data files and params
#halo_file="/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"
#particle_file="/home/jwick/data/hlist_1.00231.particles.halotools_v0p4.hdf5"
halo_file="/Users/josephwick/Documents/Argonne23/data/value_added_orphan_complete_bpl_1.002310.h5"
particle_file="/Users/josephwick/Documents/Argonne23/data/hlist_1.00231.particles.halotools_v0p4.hdf5"
box_length = 250.0 # Mpc
buff_wprp = 20.0 # Mpc

outdir = "/home/jwick/diffsmhm/analysis/output/rpwp_demo_nuts"

mass_bin_edges = np.array([10.6, 11.2], dtype=np.float64)

rpbins = np.logspace(-1, 1.2, 13, dtype=np.float64)
zmax = 20.0 # Mpc

theta = np.array(list(smhm_params.values()) + 
                 list(smhm_sigma_params.values()) + 
                 list(disruption_params.values()), dtype=np.float64
)
theta_init = np.copy(theta)

n_params = len(theta)
n_rpbins = len(rpbins) - 1

# hmc params
n_warmup_steps = 1
n_iter = 1
#

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
                host_mpeak_cut=14.7
)

idx_to_deposit = _calculate_indx_to_deposit(halos["upid"], halos["halo_id"])

print(RANK, "data loaded", len(idx_to_deposit), flush=True)

# 2) obtain "goal" measurement
np.random.seed(999)
parameter_perturbations = np.random.uniform(low=0.98, high=1.02, size=n_params)

theta_goal = theta * parameter_perturbations

# rpwp, quenched and unquenched
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

rpwp_goal, _ = compute_rpwp(
                    x1=halos["halo_x"],
                    y1=halos["halo_y"],
                    z1=halos["halo_z"],
                    w1=w,
                    w1_jac=dw,
                    inside_subvol=halos["_inside_subvol"],
                    rpbins=rpbins,
                    zmax=zmax,
                    boxsize=box_length
)

if RANK == 0:
    print("goal rpwp done", flush=True)

# 3) do optimization

# define functions

# transform functions; following STAN manual for bounded scalar

def logit(x):
    return jax.scipy.special.logit(x)

def logit_inv(x):
    return jax.nn.sigmoid(x)

def logit_inv_jac(x):
    sig = jax.nn.sigmoid(x)
    return sig * (1 - sig)

def transform_model_to_hmc(x, a, b):
    return logit((x - a) / (b - a))

def transform_hmc_to_model(y, a, b):
    return a + (b - a) * logit_inv(y)

# logdensity function set upi
# note that only rank zero will deal with this func
def logdensity_fn(
    *,
    smhm_logm_crit, smhm_ratio_logm_crit, smhm_k_logm, smhm_lowm_index,
    smhm_highm_index,

    smhm_sigma_low, smhm_sigma_high, smhm_sigma_logm_pivot, smhm_sigma_logm_width,

    satmerg_logmhost_crit, satmerg_logmhost_k, satmerg_logvr_crit_dwarfs,
    satmerg_logvr_crit_clusters, satmerg_logvr_k,

    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds
):
    # munge params into array
    theta_hmc = jnp.array([
        smhm_logm_crit, smhm_ratio_logm_crit, smhm_k_logm, smhm_lowm_index,
        smhm_highm_index,

        smhm_sigma_low, smhm_sigma_high, smhm_sigma_logm_pivot, smhm_sigma_logm_width,

        satmerg_logmhost_crit, satmerg_logmhost_k, satmerg_logvr_crit_dwarfs,
        satmerg_logvr_crit_clusters, satmerg_logvr_k
    ], dtype=jnp.float64)

    # transform HMC params into model params
    theta = transform_hmc_to_model(theta_hmc, lower_bounds, upper_bounds)

    # call func for error/potential
    U = get_potential(theta)
    
    # convert potential to logdensity
    log_density_model = -0.5 * U 

    # transform logdensity of model into hmc logdensity 
    abs_det_of_transform = jnp.log10(jnp.sum((upper_bounds - lower_bounds) * logit_inv_jac(theta_hmc)))
    log_density_hmc = log_density_model * abs_det_of_transform

    return log_density_hmc

# we wrap in a lambda for blackjax 
logdensity = lambda x: logdensity_fn(**x)

# TODO: probably move these
error_history = np.empty(n_iter+n_warmup_steps, dtype=np.float64)
rpwp_history = np.empty((n_iter+n_warmup_steps, n_rpbins), dtype=np.float64)
step_index = np.zeros(1, dtype="i")

# method to calculate quenched and unquenched rpwp 
# this is what we pure_callback to; it returns both value and gradient
def potential(
    theta,
    rpwp_goal=rpwp_goal,
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
    boxsize=box_length,
    error_history=error_history,
    rpwp_history=rpwp_history
):

    # broadcast continue command to other ranks 
    cont = True
    cont = COMM.bcast(cont, root=0)

    # broadcast theta
    COMM.Bcast([theta, MPI.DOUBLE], root=0)

    error, error_grad, rpwp = mse_rpwp(
            rpwp_goal,
            log_halomass, log_host_halomass, log_vmax_by_vmpeak,
            halo_x, halo_y, halo_z,
            upid, inside_subvol,
            idx_to_deposit,
            rpbins, mass_bin_low, mass_bin_high, zmax, box_length,
            theta
    )

    # rank zero update our arrays
    error_history[step_index[0]] = error
    rpwp_history[step_index[0]] = rpwp
    step_index[0] = step_index[0]+1

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

# non 0 mpi ranks
if RANK > 0:
    while True:

        # check condition
        cont = -1
        cont = COMM.bcast(cont, root=0)
        if not cont: 
            break

        # receive current theta
        theta = np.empty(n_params, dtype=np.float64)
        COMM.Bcast([theta, MPI.DOUBLE], root=0)

        # do computation
        _, _, _ = mse_rpwp(
                rpwp_goal,
                halos["logmpeak"], halos["loghost_mpeak"], halos["logvmax_frac"],
                halos["halo_x"], halos["halo_y"], halos["halo_z"],
                halos["upid"], halos["_inside_subvol"],
                idx_to_deposit,
                rpbins, mass_bin_edges[0], mass_bin_edges[1], zmax, box_length,
                theta
        )

    if RANK == 1:
        print("while exited")

# RANK 0 continues with the HMC
else:
    rng_key = jax.random.PRNGKey(42)

    # transform theta for the initial position
    initial_hmc_params = transform_model_to_hmc(theta, lower_bounds, upper_bounds)

    # adapt the mass matrix
    initial_position = {
        "smhm_logm_crit":initial_hmc_params[0],
        "smhm_ratio_logm_crit":initial_hmc_params[1],
        "smhm_k_logm":initial_hmc_params[2],
        "smhm_lowm_index":initial_hmc_params[3],
        "smhm_highm_index":initial_hmc_params[4],
        "smhm_sigma_low":initial_hmc_params[5],
        "smhm_sigma_high":initial_hmc_params[6],
        "smhm_sigma_logm_pivot":initial_hmc_params[7],
        "smhm_sigma_logm_width":initial_hmc_params[8],
        "satmerg_logmhost_crit":initial_hmc_params[9],
        "satmerg_logmhost_k":initial_hmc_params[10],
        "satmerg_logvr_crit_dwarfs":initial_hmc_params[11],
        "satmerg_logvr_crit_clusters":initial_hmc_params[12],
        "satmerg_logvr_k":initial_hmc_params[13]
    }
        
    # split the key
    rng_key, sample_key, warmup_key = jax.random.split(rng_key, 3)

    print("Begin warmup", flush=True)
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity, 
                                        is_mass_matrix_diagonal=False)
    t0 = time.time()
    (init_state, tuned_params), _ = warmup.run(warmup_key, initial_position,
                                               num_steps=n_warmup_steps)
    t1 = time.time()
    print("Warm up done", t1-t0, flush=True)

    hmc = blackjax.nuts(logdensity, **tuned_params)

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

    states = inference_loop(sample_key, hmc_kernel, init_state, n_iter)
    theta_final_hmc = np.array([
                    states.position["smhm_logm_crit"][-1],
                    states.position["smhm_ratio_logm_crit"][-1],
                    states.position["smhm_k_logm"][-1],
                    states.position["smhm_lowm_index"][-1],
                    states.position["smhm_highm_index"][-1],

                    states.position["smhm_sigma_low"][-1],
                    states.position["smhm_sigma_high"][-1],
                    states.position["smhm_sigma_logm_pivot"][-1],
                    states.position["smhm_sigma_logm_width"][-1],

                    states.position["satmerg_logmhost_crit"][-1],
                    states.position["satmerg_logmhost_k"][-1],
                    states.position["satmerg_logvr_crit_dwarfs"][-1],
                    states.position["satmerg_logvr_crit_clusters"][-1],
                    states.position["satmerg_logvr_k"][-1],
    ], dtype=np.float64)
    theta_final = transform_hmc_to_model(theta_final_hmc, lower_bounds, upper_bounds)

    # we're done iterating, broadcast FALSE to stop the other ranks
    cont = False
    COMM.bcast(cont, root=0)
    print("opt done", flush=True)

    samples = states.position
    print(states.logdensity)

# 4) save output and plot results (rank 0 for IO, everyone for calculations)

# save outputs 
if RANK == 0:
    # positions
    positions = states.position
        
    positions_df = pd.DataFrame.from_dict(positions)
    # need to transform from HMC positions to model positions
    for p,key in enumerate(positions_df.keys()):
        positions_df[key] = transform_hmc_to_model(np.array(positions_df[key]),
                                                   lower_bounds[p], upper_bounds[p])

    fpath_positions = outdir+"/positions.csv"
    positions_df.to_csv(fpath_positions, index=False)

    # logdensity and grad
    # note that these are in "HMC units"
    logdensities = np.array(states.logdensity, dtype=np.float64)
    logdensity_grads = states.logdensity_grad

    logdensity_df = pd.DataFrame.from_dict(logdensity_grads)
    logdensity_df.insert(0, column="logdensity", value=logdensities)

    fpath_logdensity = outdir+"/logdensity_grad.csv"
    logdensity_df.to_csv(fpath_logdensity, index=False)    

maybe="""
# rpwp and error at each position; note this involves all ranks
rpwp_history = np.empty((n_iter, n_rpbins), dtype=np.float64)
error_history = np.empty(n_iter, dtype=np.float64)
# loop over positions
for i in range(n_iter):
    # broadcast position / theta
    if RANK == 0:
        theta = np.array(positions_df.iloc[i], dtype=np.float64)
    else:
        theta = np.empty(n_params, dtype=np.float64)
    COMM.Bcast([theta, MPI.DOUBLE], root=0)    

    # calculate rpwp
    w, dw = compute_weight_and_jac(
                halos["logmpeak"],
                halos["loghost_mpeak"],
                halos["logvmax_frac"],
                halos["upid"],
                idx_to_deposit,
                mass_bin_edges[0], mass_bin_edges[1],
                theta
    )

    rpwp, _ = compute_rpwp(
                    x1=halos["halo_x"],
                    y1=halos["halo_y"],
                    z1=halos["halo_z"],
                    w1=w,
                    w1_jac=dw,
                    inside_subvol=halos["_inside_subvol"],
                    rpbins=rpbins,
                    zmax=zmax,
                    boxsize=box_length
    )

    # calculate error and save
    if RANK == 0:
        error = np.sum((rpwp-rpwp_goal)*(rpwp-rpwp_goal)) / len(rpwp)

        rpwp_history[i,:] = rpwp
        error_history[i] = error

# rank 0 write to disk
if RANK == 0:
    rpwp_history.tofile(outdir+"/rpwp_history.csv", sep=",")
    error_history.tofile(outdir+"/error_history.csv", sep=",")
"""

# let's plot rpwp before and after optimization
# first we need initial and final rpwp measurements
if RANK > 0:
    theta_final = None
theta_final = COMM.bcast(theta_final, root=0)
# initial
w, dw, = compute_weight_and_jac(
            logmpeak=halos["logmpeak"],
            loghost_mpeak=halos["loghost_mpeak"],
            log_vmax_by_vmpeak=halos["logvmax_frac"],
            upid=halos["upid"],
            idx_to_deposit=idx_to_deposit,
            mass_bin_low=mass_bin_edges[0],
            mass_bin_high=mass_bin_edges[1],
            theta=theta_init
)

rpwp_init, _ = compute_rpwp(
                x1=halos["halo_x"],
                y1=halos["halo_y"],
                z1=halos["halo_z"],
                w1=w,
                w1_jac=dw,
                inside_subvol=halos["_inside_subvol"],
                rpbins=rpbins,
                zmax=zmax,
                boxsize=box_length
)

# final
w, dw = compute_weight_and_jac(
            logmpeak=halos["logmpeak"],
            loghost_mpeak=halos["loghost_mpeak"],
            log_vmax_by_vmpeak=halos["logvmax_frac"],
            upid=halos["upid"],
            idx_to_deposit=idx_to_deposit,
            mass_bin_low=mass_bin_edges[0],
            mass_bin_high=mass_bin_edges[1],
            theta=theta_final
)

rpwp_final, _ = compute_rpwp(
                x1=halos["halo_x"],
                y1=halos["halo_y"],
                z1=halos["halo_z"],
                w1=w,
                w1_jac=dw,
                inside_subvol=halos["_inside_subvol"],
                rpbins=rpbins,
                zmax=zmax,
                boxsize=box_length
)

# figure
if RANK == 0:
    fig = plt.figure(figsize=(16,8), facecolor="w")

    plt.plot(rpbins[:-1], rpwp_init, c="tab:blue", linewidth=3)
    plt.plot(rpbins[:-1], rpwp_final, c="tab:orange", linewidth=3)
    plt.plot(rpbins[:-1], rpwp_goal, c="k", linewidth=1)

    plt.xlabel("rp")
    plt.ylabel("rp wp")
    plt.title("Quenched wp(rp)")

    plt.legend(["initial guess", "final guess", "ideal"])

    plt.savefig("rpwp_demo_nuts.png")

