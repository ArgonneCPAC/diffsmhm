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
from diffsmhm.galhalo_models.sigmoid_quenching import (
    DEFAULT_PARAM_VALUES as quenching_params,
    PARAM_BOUNDS as quenching_bounds
)

from diffsmhm.loader import load_and_chop_data_bolshoi_planck
from diffsmhm.galhalo_models.merging import _calculate_indx_to_deposit

from diffsmhm.analysis.tools.diff_sm import compute_weight_and_jac_quench
from diffsmhm.analysis.tools.rpwp import compute_rpwp

from diffsmhm.analysis.tools.error import mse_rpwp_quench

from hmc_bounding import (
    hmc_pos_to_model_pos,
    model_pos_to_hmc_pos,
    logdens_model_to_logdens_hmc
)

# data files and params
halo_file = "/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"
particle_file = "/home/jwick/data/hlist_1.00231.particles.halotools_v0p4.hdf5"
box_length = 250.0  # Mpc
buff_wprp = 20.0  # Mpc

outdir = "/home/jwick/diffsmhm/analysis/output/rpwp_quench_demo_nuts"

mass_bin_edges = np.array([10.6, 11.2], dtype=np.float64)

rpbins = np.logspace(-1, 1.2, 13, dtype=np.float64)
zmax = 20.0  # Mpc

theta = np.array(
            list(smhm_params.values()) +
            list(smhm_sigma_params.values()) +
            list(disruption_params.values()) +
            list(quenching_params.values()), dtype=np.float64
)
theta_init = np.copy(theta)

n_params = len(theta)
n_rpbins = len(rpbins) - 1

# hmc params
n_warmup_steps = 250
n_iter = 500

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
    quenching_bounds["fq_cens_logm_crit"][0],
    quenching_bounds["fq_cens_k"][0],
    quenching_bounds["fq_cens_ylo"][0],
    quenching_bounds["fq_cens_yhi"][0],
    quenching_bounds["fq_satboost_logmhost_crit"][0],
    quenching_bounds["fq_satboost_logmhost_k"][0],
    quenching_bounds["fq_satboost_clusters"][0],
    quenching_bounds["fq_sat_delay_time"][0],
    quenching_bounds["fq_sat_tinfall_k"][0]
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
    quenching_bounds["fq_cens_logm_crit"][1],
    quenching_bounds["fq_cens_k"][1],
    quenching_bounds["fq_cens_ylo"][1],
    quenching_bounds["fq_cens_yhi"][1],
    quenching_bounds["fq_satboost_logmhost_crit"][1],
    quenching_bounds["fq_satboost_logmhost_k"][1],
    quenching_bounds["fq_satboost_clusters"][1],
    quenching_bounds["fq_sat_delay_time"][1],
    quenching_bounds["fq_sat_tinfall_k"][1]
], dtype=np.float64)

# 1) load data
halos, _ = load_and_chop_data_bolshoi_planck(
                particle_file,
                halo_file,
                box_length,
                buff_wprp,
                host_mpeak_cut=10.0
)

idx_to_deposit = _calculate_indx_to_deposit(halos["upid"], halos["halo_id"])

# 2) obtain "goal" measurement
np.random.seed(999)
parameter_perturbations = np.random.uniform(low=0.98, high=1.02, size=n_params)

theta_goal = theta * parameter_perturbations

# rpwp, quenched and unquenched
w_q, dw_q, w_nq, dw_nq = compute_weight_and_jac_quench(
                            logmpeak=halos["logmpeak"],
                            loghost_mpeak=halos["loghost_mpeak"],
                            log_vmax_by_vmpeak=halos["logvmax_frac"],
                            upid=halos["upid"],
                            time_since_infall=halos["time_since_infall"],
                            idx_to_deposit=idx_to_deposit,
                            mass_bin_low=mass_bin_edges[0],
                            mass_bin_high=mass_bin_edges[1],
                            theta=theta
)

if RANK == 0:
    print("goal weights done", flush=True)

# goal rpwp computation
rpwp_q_goal, _ = compute_rpwp(
                    x1=halos["halo_x"],
                    y1=halos["halo_y"],
                    z1=halos["halo_z"],
                    w1=w_q,
                    w1_jac=dw_q,
                    inside_subvol=halos["_inside_subvol"],
                    rpbins=rpbins,
                    zmax=zmax,
                    boxsize=box_length
)

rpwp_nq_goal, _ = compute_rpwp(
                    x1=halos["halo_x"],
                    y1=halos["halo_y"],
                    z1=halos["halo_z"],
                    w1=w_nq,
                    w1_jac=dw_nq,
                    inside_subvol=halos["_inside_subvol"],
                    rpbins=rpbins,
                    zmax=zmax,
                    boxsize=box_length
)

if RANK == 0:
    print("goal rpwp done", flush=True)

# 3) do optimization
# define functions


# logdensity function set up; note that only rank zero will deal with this func
def logdensity_fn(
    *,
    smhm_logm_crit, smhm_ratio_logm_crit, smhm_k_logm, smhm_lowm_index,
    smhm_highm_index,

    smhm_sigma_low, smhm_sigma_high, smhm_sigma_logm_pivot, smhm_sigma_logm_width,

    satmerg_logmhost_crit, satmerg_logmhost_k, satmerg_logvr_crit_dwarfs,
    satmerg_logvr_crit_clusters, satmerg_logvr_k,

    fq_cens_logm_crit, fq_cens_k, fq_cens_ylo, fq_cens_yhi, fq_satboost_logmhost_crit,
    fq_satboost_logmhost_k, fq_satboost_clusters, fq_sat_delay_time, fq_sat_tinfall_k,

    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds
):
    # munge params into array
    theta_hmc = jnp.array([
        smhm_logm_crit, smhm_ratio_logm_crit, smhm_k_logm, smhm_lowm_index,
        smhm_highm_index,

        smhm_sigma_low, smhm_sigma_high, smhm_sigma_logm_pivot, smhm_sigma_logm_width,

        satmerg_logmhost_crit, satmerg_logmhost_k, satmerg_logvr_crit_dwarfs,
        satmerg_logvr_crit_clusters, satmerg_logvr_k,

        fq_cens_logm_crit, fq_cens_k, fq_cens_ylo, fq_cens_yhi,
        fq_satboost_logmhost_crit, fq_satboost_logmhost_k, fq_satboost_clusters,
        fq_sat_delay_time, fq_sat_tinfall_k
    ], dtype=jnp.float64)

    # transform HMC params into model params
    theta = hmc_pos_to_model_pos(theta_hmc, lower_bounds, upper_bounds)

    # call func for error/potential
    U = get_potential(theta)

    # convert potential to logdensity
    log_density_model = -0.5 * U

    # transform logdensity of model into hmc logdensity
    log_density_hmc = logdens_model_to_logdens_hmc(
                        log_density_model,
                        theta_hmc,
                        lower_bounds,
                        upper_bounds
    )

    return log_density_hmc


# we wrap to one input blackjax
def logdensity(x):
    return logdensity_fn(**x)


# method to calculate quenched and unquenched rpwp
# this is what we pure_callback to; it returns both value and gradient
def potential(
    theta,
    rpwp_q_goal=rpwp_q_goal, rpwp_nq_goal=rpwp_q_goal,
    log_halomass=halos["logmpeak"],
    log_host_halomass=halos["loghost_mpeak"],
    log_vmax_by_vmpeak=halos["logvmax_frac"],
    halo_x=halos["halo_x"], halo_y=halos["halo_y"], halo_z=halos["halo_z"],
    upid=halos["upid"],
    time_since_infall=halos["time_since_infall"],
    idx_to_deposit=idx_to_deposit,
    inside_subvol=halos["_inside_subvol"],
    mass_bin_low=mass_bin_edges[0], mass_bin_high=mass_bin_edges[1],
    rpbins=rpbins,
    zmax=zmax,
    boxsize=box_length
):

    # broadcast continue command to other ranks
    cont = True
    cont = COMM.bcast(cont, root=0)

    # broadcast theta
    COMM.Bcast([theta, MPI.DOUBLE], root=0)

    error, error_grad = mse_rpwp_quench(
            rpwp_q_goal=rpwp_q_goal,
            rpwp_nq_goal=rpwp_nq_goal,
            logmpeak=log_halomass,
            loghost_mpeak=log_host_halomass,
            log_vmax_by_vmpeak=log_vmax_by_vmpeak,
            halo_x=halo_x, halo_y=halo_y, halo_z=halo_z,
            upid=upid,
            time_since_infall=time_since_infall,
            idx_to_deposit=idx_to_deposit,
            inside_subvol=inside_subvol,
            rpbins=rpbins,
            mass_bin_low=mass_bin_low, mass_bin_high=mass_bin_high,
            zmax=zmax,
            boxsize=box_length,
            theta=theta
    )

    return error, error_grad


# this returns only val
@custom_vjp
def get_potential(
    theta,
):
    val, grad = jax.pure_callback(
                    potential,
                    (np.array(1.0, dtype=np.float64), np.ones(23, dtype=np.float64)),
                    theta
    )

    return val


def vjp_fwd(
    theta,
):
    # hardcoded sizes isn't elegant, but arg length needs to match potential
    val, grad = jax.pure_callback(
                    potential,
                    (np.array(1.0, dtype=np.float64), np.ones(23, dtype=np.float64)),
                    theta
    )

    return val, grad


def vjp_bwd(grad, tan):
    # note that blackjax expects a tuple here
    return (grad * tan,)


get_potential.defvjp(vjp_fwd, vjp_bwd)

# this is where we split up the ranks
# sampler set up

# non 0 mpi ranks only do computation
if RANK > 0:
    while True:

        # check condition
        cont = -1
        cont = COMM.bcast(cont, root=0)
        if not cont:
            break

        # receive current theta
        theta = np.empty(23, dtype=np.float64)
        COMM.Bcast([theta, MPI.DOUBLE], root=0)

        # do computation
        _, _ = mse_rpwp_quench(
                rpwp_q_goal=rpwp_q_goal,
                rpwp_nq_goal=rpwp_nq_goal,
                logmpeak=halos["logmpeak"],
                loghost_mpeak=halos["loghost_mpeak"],
                log_vmax_by_vmpeak=halos["logvmax_frac"],
                halo_x=halos["halo_x"],
                halo_y=halos["halo_y"],
                halo_z=halos["halo_z"],
                upid=halos["upid"],
                time_since_infall=halos["time_since_infall"],
                idx_to_deposit=idx_to_deposit,
                inside_subvol=halos["_inside_subvol"],
                rpbins=rpbins,
                mass_bin_low=mass_bin_edges[0], mass_bin_high=mass_bin_edges[1],
                zmax=zmax,
                boxsize=box_length,
                theta=theta
        )

    if RANK == 1:
        print("while exited")

# RANK 0 continues with the HMC
else:
    rng_key = jax.random.PRNGKey(42)

    # transform theta for the initial position
    initial_hmc_params = model_pos_to_hmc_pos(theta, lower_bounds, upper_bounds)

    # adapt the mass matrix
    initial_position = {
        "smhm_logm_crit" : initial_hmc_params[0],
        "smhm_ratio_logm_crit" : initial_hmc_params[1],
        "smhm_k_logm" : initial_hmc_params[2],
        "smhm_lowm_index" : initial_hmc_params[3],
        "smhm_highm_index" : initial_hmc_params[4],
        "smhm_sigma_low" : initial_hmc_params[5],
        "smhm_sigma_high" : initial_hmc_params[6],
        "smhm_sigma_logm_pivot" : initial_hmc_params[7],
        "smhm_sigma_logm_width" : initial_hmc_params[8],
        "satmerg_logmhost_crit" : initial_hmc_params[9],
        "satmerg_logmhost_k" : initial_hmc_params[10],
        "satmerg_logvr_crit_dwarfs" : initial_hmc_params[11],
        "satmerg_logvr_crit_clusters" : initial_hmc_params[12],
        "satmerg_logvr_k" : initial_hmc_params[13],
        "fq_cens_logm_crit" : initial_hmc_params[14],
        "fq_cens_k" : initial_hmc_params[15],
        "fq_cens_ylo" : initial_hmc_params[16],
        "fq_cens_yhi" : initial_hmc_params[17],
        "fq_satboost_logmhost_crit" : initial_hmc_params[18],
        "fq_satboost_logmhost_k" : initial_hmc_params[19],
        "fq_satboost_clusters" : initial_hmc_params[20],
        "fq_sat_delay_time" : initial_hmc_params[21],
        "fq_sat_tinfall_k" : initial_hmc_params[22]
    }

    # split the key
    rng_key, sample_key, warmup_key = jax.random.split(rng_key, 3)

    print("Begin warmup", flush=True)
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity,
                                        is_mass_matrix_diagonal=False)
    (init_state, tuned_params), _ = warmup.run(warmup_key, initial_position,
                                               num_steps=n_warmup_steps)
    print("Warm up done", flush=True)

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

                    states.position["fq_cens_logm_crit"][-1],
                    states.position["fq_cens_k"][-1],
                    states.position["fq_cens_ylo"][-1],
                    states.position["fq_cens_yhi"][-1],
                    states.position["fq_satboost_logmhost_crit"][-1],
                    states.position["fq_satboost_logmhost_k"][-1],
                    states.position["fq_satboost_clusters"][-1],
                    states.position["fq_sat_delay_time"][-1],
                    states.position["fq_sat_tinfall_k"][-1]
    ], dtype=np.float64)
    theta_final = hmc_pos_to_model_pos(theta_final_hmc, lower_bounds, upper_bounds)

    # we're done iterating, broadcast FALSE to stop the other ranks
    cont = False
    COMM.bcast(cont, root=0)
    print("opt done", flush=True)

    samples = states.position
    print(states.logdensity)

# 4) save output and plot results (rank 0 for IO, all for computations)

# save outputs
if RANK == 0:
    # positions
    positions = states.position

    positions_df = pd.DataFrame.from_dict(positions)
    # need to transform from HMC positions to model positions
    for p, key in enumerate(positions_df.keys()):
        positions_df[key] = hmc_pos_to_model_pos(np.array(positions_df[key]),
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


# let's plot quenched and unquenched before and after optimization
# first we need initial and final rpwp measurements
if RANK > 0:
    theta_final = None
theta_final = COMM.bcast(theta_final, root=0)
# initial
w_q, dw_q, w_nq, dw_nq = compute_weight_and_jac_quench(
                            logmpeak=halos["logmpeak"],
                            loghost_mpeak=halos["loghost_mpeak"],
                            log_vmax_by_vmpeak=halos["logvmax_frac"],
                            upid=halos["upid"],
                            time_since_infall=halos["time_since_infall"],
                            idx_to_deposit=idx_to_deposit,
                            mass_bin_low=mass_bin_edges[0],
                            mass_bin_high=mass_bin_edges[1],
                            theta=theta
)

rpwp_q_init, _ = compute_rpwp(
                x1=halos["halo_x"],
                y1=halos["halo_y"],
                z1=halos["halo_z"],
                w1=w_q,
                w1_jac=dw_q,
                inside_subvol=halos["_inside_subvol"],
                rpbins=rpbins,
                zmax=zmax,
                boxsize=box_length
)

rpwp_nq_init, _ = compute_rpwp(
                x1=halos["halo_x"],
                y1=halos["halo_y"],
                z1=halos["halo_z"],
                w1=w_nq,
                w1_jac=dw_nq,
                inside_subvol=halos["_inside_subvol"],
                rpbins=rpbins,
                zmax=zmax,
                boxsize=box_length
)

# final
w_q, dw_q, w_nq, dw_nq = compute_weight_and_jac_quench(
                            logmeak=halos["logmpeak"],
                            loghost_mpeak=halos["loghost_mpeak"],
                            log_vmax_by_vmpeak=halos["logvmax_frac"],
                            upid=halos["upid"],
                            time_since_infall=halos["time_since_infall"],
                            idx_to_deposit=idx_to_deposit,
                            mass_bin_low=mass_bin_edges[0],
                            mass_bin_high=mass_bin_edges[1],
                            theta=theta_final
)


rpwp_q_final, _ = compute_rpwp(
                x1=halos["halo_x"],
                y1=halos["halo_y"],
                z1=halos["halo_z"],
                w1=w_q,
                w1_jac=dw_q,
                inside_subvol=halos["_inside_subvol"],
                rpbins=rpbins,
                zmax=zmax,
                boxsize=box_length
)

rpwp_nq_final, _ = compute_rpwp(
                x1=halos["halo_x"],
                y1=halos["halo_y"],
                z1=halos["halo_z"],
                w1=w_nq,
                w1_jac=dw_nq,
                inside_subvol=halos["_inside_subvol"],
                rpbins=rpbins,
                zmax=zmax,
                boxsize=box_length
)

# figure
if RANK == 0:
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), facecolor="w")

    # quenched
    axs[0].plot(rpbins[:-1], rpwp_q_init, c="tab:blue", linewidth=3)
    axs[0].plot(rpbins[:-1], rpwp_q_final, c="tab:orange", linewidth=3)
    axs[0].plot(rpbins[:-1], rpwp_q_goal, c="k", linewidth=1)

    axs[0].set_xlabel("rp")
    axs[0].set_ylabel("rp wp")
    axs[0].set_title("Quenched wp(rp)")

    axs[0].legend(["initial guess", "final guess", "ideal"])

    # unquenched
    axs[1].plot(rpbins[:-1], rpwp_nq_init, c="tab:blue", linewidth=3)
    axs[1].plot(rpbins[:-1], rpwp_nq_final, c="tab:orange", linewidth=3)
    axs[1].plot(rpbins[:-1], rpwp_nq_goal, c="k", linewidth=1)

    axs[1].set_xlabel("rp")
    axs[1].set_ylabel("rp wp")
    axs[1].set_title("Unquenched wp(rp)")

    plt.savefig("figures/rpwp_quench_nuts_demo.png")
