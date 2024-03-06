import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import custom_vjp

import blackjax

from functools import lru_cache

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

from diff_sm import compute_weight_and_jac_quench
from rpwp import compute_rpwp

from error import mse_rpwp_quench


# data files and params
halo_file="/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"
particle_file="/home/jwick/data/hlist_1.00231.particles.halotools_v0p4.hdf5"
box_length = 250.0 # Mpc
buff_wprp = 20.0 # Mpc

mass_bin_edges = np.array([10.6, 11.2], dtype=np.float64)

rpbins = np.logspace(-1, 1.2, 13, dtype=np.float64)
zmax = 20.0 # Mpc

theta = [
    smhm_params["smhm_logm_crit"],
    smhm_params["smhm_ratio_logm_crit"],
    smhm_params["smhm_k_logm"],
    smhm_params["smhm_lowm_index"],
    smhm_params["smhm_highm_index"],
    smhm_sigma_params["smhm_sigma_low"],
    smhm_sigma_params["smhm_sigma_high"],
    smhm_sigma_params["smhm_sigma_logm_pivot"],
    smhm_sigma_params["smhm_sigma_logm_width"],
    disruption_params["satmerg_logmhost_crit"],
    disruption_params["satmerg_logmhost_k"],
    disruption_params["satmerg_logvr_crit_dwarfs"],
    disruption_params["satmerg_logvr_crit_clusters"],
    disruption_params["satmerg_logvr_k"],
    quenching_params["fq_cens_logm_crit"],
    quenching_params["fq_cens_k"],
    quenching_params["fq_cens_ylo"],
    quenching_params["fq_cens_yhi"],
    quenching_params["fq_satboost_logmhost_crit"],
    quenching_params["fq_satboost_logmhost_k"],
    quenching_params["fq_satboost_clusters"],
    quenching_params["fq_sat_delay_time"],
    quenching_params["fq_sat_tinfall_k"]
]
theta = np.array(theta, dtype=np.float64)

lower_bounds = [
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
]
upper_bounds = [
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
]
lower_bounds = np.array(lower_bounds, dtype=np.float64)
upper_bounds = np.array(upper_bounds, dtype=np.float64)

n_params = len(theta)
n_rpbins = len(rpbins) - 1


# 1) load data
halos, _ = load_and_chop_data_bolshoi_planck(
                particle_file,
                halo_file,
                box_length,
                buff_wprp,
                host_mpeak_cut=0.0
)

idx_to_deposit = _calculate_indx_to_deposit(halos["upid"], halos["halo_id"])


# 2) obtain "goal" measurement
np.random.seed(999)
parameter_perturbations = np.random.uniform(low=0.95, high=1.05, size=n_params)

theta_goal = theta * parameter_perturbations

# rpwp, quenched and unquenched
w_q, dw_q, w_nq, dw_nq = compute_weight_and_jac_quench(
                        halos["logmpeak"],
                        halos["loghost_mpeak"],
                        halos["logvmax_frac"],
                        halos["upid"],
                        halos["time_since_infall"],
                        idx_to_deposit,
                        mass_bin_edges[0], mass_bin_edges[1],
                        theta
)

wgt_mask_quench = w_q > 0
wgt_mask_no_quench = w_nq > 0

if RANK == 0:
    print("goal weights done", flush=True)

# goal rpwp computation
rpwp_q_goal, _ = compute_rpwp(
                    x1=halos["halo_x"][wgt_mask_quench],
                    y1=halos["halo_y"][wgt_mask_quench],
                    z1=halos["halo_z"][wgt_mask_quench],
                    w1=w_q[wgt_mask_quench],
                    w1_jac=dw_q[:, wgt_mask_quench],
                    inside_subvol=halos["_inside_subvol"][wgt_mask_quench],
                    rpbins=rpbins,
                    zmax=zmax,
                    boxsize=box_length
)

rpwp_nq_goal, _ = compute_rpwp(
                    x1=halos["halo_x"][wgt_mask_no_quench],
                    y1=halos["halo_y"][wgt_mask_no_quench],
                    z1=halos["halo_z"][wgt_mask_no_quench],
                    w1=w_nq[wgt_mask_no_quench],
                    w1_jac=dw_nq[:, wgt_mask_no_quench],
                    inside_subvol=halos["_inside_subvol"][wgt_mask_no_quench],
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
    return np.log(x/(1-x))

def logit_inv(x):
    return jax.nn.sigmoid(x)

def logit_inv_jac(x):
    sig = jax.nn.sigmoid(x)
    return sig * (1 - sig)

def transform_model_to_hmc(x, a, b):
    return logit((x - a) / (b - a))

def transform_hmc_to_model(y, a, b):
    return a + (b - a) * logit_inv(y)

# error function set up
# note this function doesn't need to stick within jax if we use pure_callback
def compute(
    *,
    smhm_logm_crit, smhm_ratio_logm_crit, smhm_k_logm, smhm_lowm_index,
    smhm_highm_index,

    smhm_sigma_low, smhm_sigma_high, smhm_sigma_logm_pivot, smhm_sigma_logm_width,

    satmerg_logmhost_crit, satmerg_logmhost_k, satmerg_logvr_crit_dwarfs,
    satmerg_logvr_crit_clusters, satmerg_logvr_k,

    fq_cens_logm_crit, fq_cens_k, fq_cens_ylo, fq_cens_yhi, fq_satboost_logmhost_crit,
    fq_satboost_logmhost_k, fq_satboost_clusters, fq_sat_delay_time, fq_sat_tinfall_k,
    
    goal_q=rpwp_q_goal,
    goal_nq=rpwp_nq_goal,
    logmass=halos["logmpeak"], log_hostmass=halos["loghost_mpeak"],
    log_vmax_by_vmpeak=halos["logvmax_frac"],
    halo_x=halos["halo_x"], halo_y=halos["halo_y"], halo_z=halos["halo_z"],
    time_since_infall=halos["time_since_infall"],
    upid=halos["upid"],
    inside_subvol=halos["_inside_subvol"],
    idx_to_deposit=idx_to_deposit,
    rpbins=rpbins,
    mass_bin_low=mass_bin_edges[0], mass_bin_high=mass_bin_edges[1],
    zmax=zmax, boxsize=box_length,
    lower_bounds=lower_bounds, upper_bounds=upper_bounds
    
):
    # munge params into array
    theta_hmc = [
        smhm_logm_crit, smhm_ratio_logm_crit, smhm_k_logm, smhm_lowm_index,
        smhm_highm_index,

        smhm_sigma_low, smhm_sigma_high, smhm_sigma_logm_pivot, smhm_sigma_logm_width,

        satmerg_logmhost_crit, satmerg_logmhost_k, satmerg_logvr_crit_dwarfs,
        satmerg_logvr_crit_clusters, satmerg_logvr_k,

        fq_cens_logm_crit, fq_cens_k, fq_cens_ylo, fq_cens_yhi, fq_satboost_logmhost_crit,
        fq_satboost_logmhost_k, fq_satboost_clusters, fq_sat_delay_time, fq_sat_tinfall_k,
    ]
    theta_hmc = np.array(theta_hmc, dtype=np.float64)

    # transform HMC params into model params
    theta = transform_hmc_to_model(theta_hmc, lower_bounds, upper_bounds)
    if RANK == 0:
        print("hmc:", theta_hmc)
        print("mod:", theta, flush=True)

    cont = True
    COMM.bcast(cont, root=0)

    COMM.Bcast([theta, MPI.DOUBLE], root=0)

    # call func for error and gradient
    log_density_model, log_density_model_grad = mse_rpwp_quench(
                goal_q, goal_nq,
                logmass, log_hostmass, log_vmax_by_vmpeak,
                halo_x, halo_y, halo_z,
                upid, inside_subvol, time_since_infall, idx_to_deposit,
                rpbins, mass_bin_low, mass_bin_high, zmax, boxsize,
                theta
    )

    if RANK == 0:
        print("err: ", log_density_model)
        print("grad:", log_density_model_grad)


        # convert potential to logdensity
        log_density_model *= -1
        log_density_model_grad *= -1

        # transform log density of model into hmc logdensity 
        abs_det_of_transform = np.abs(np.prod(
                                (upper_bounds - lower_bounds) * logit_inv_jac(theta_hmc)
                               ))
        log_density_hmc = log_density_model * abs_det_of_transform

        # and also for derivatives
        log_density_hmc_grad = log_density_model_grad * logit_inv_jac(theta_hmc) * abs_det_of_transform + \
                               log_density_model * abs_det_of_transform / theta_hmc

        ummm="""
        log_density_hmc_grad = log_density_model_grad * logit_inv_of_y_dvtv * np.abs(np.prod(
                                (upper_bounds - lower_bounds) * logit_inv_of_y_dvtv * 
                                (1 - logit_inv_of_y) + (upper_bounds - lower_bounds) *
                                logit_inv_of_y * (1 - logit_inv_of_y_dvtv)))"""

        return log_density_hmc, log_density_hmc_grad

    else:
        return None, None

compute_wrapper = lambda x: compute(**x)

@custom_vjp
def error_fn(params):
    val, grad = jax.pure_callback(compute_wrapper,
        (np.array(1.0, dtype=np.float64), np.ones(23, dtype=np.float64)),
        params)

    return val

def vjp_fwd(params):
    val, grad = jax.pure_callback(compute_wrapper,
        (np.array(1.0, dtype=np.float64), np.ones(23, dtype=np.float64)),
        params)

    return val, grad

def vjp_bwd(grad, tan):
    # blackjax wants a dictionary for the gradient :)
    grad_dict = {
        "smhm_logm_crit":grad[0],
        "smhm_ratio_logm_crit":grad[1],
        "smhm_k_logm":grad[2],
        "smhm_lowm_index":grad[3],
        "smhm_highm_index":grad[4],

        "smhm_sigma_low":grad[5],
        "smhm_sigma_high":grad[6],
        "smhm_sigma_logm_pivot":grad[7],
        "smhm_sigma_logm_width":grad[8],

        "satmerg_logmhost_crit":grad[9],
        "satmerg_logmhost_k":grad[10],
        "satmerg_logvr_crit_dwarfs":grad[11],
        "satmerg_logvr_crit_clusters":grad[12],
        "satmerg_logvr_k":grad[13],

        "fq_cens_logm_crit":grad[14],
        "fq_cens_k":grad[15],
        "fq_cens_ylo":grad[16],
        "fq_cens_yhi":grad[17],
        "fq_satboost_logmhost_crit":grad[18],
        "fq_satboost_logmhost_k":grad[19],
        "fq_satboost_clusters":grad[20],
        "fq_sat_delay_time":grad[21],
        "fq_sat_tinfall_k":grad[22]
    }
    
    for key in grad_dict.keys():
        grad_dict[key] = grad_dict[key] * tan

    # also wants tuple return type
    return (grad_dict,)

error_fn.defvjp(vjp_fwd, vjp_bwd)


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
        theta = np.empty(23, dtype=np.float64)
        COMM.Bcast([theta, MPI.DOUBLE], root=0)

        # do computation
        _, _ = mse_rpwp_quench(
                rpwp_q_goal, rpwp_nq_goal,
                halos["logmpeak"], halos["loghost_mpeak"], halos["logvmax_frac"],
                halos["halo_x"], halos["halo_y"], halos["halo_z"],
                halos["upid"], halos["_inside_subvol"],
                halos["time_since_infall"],
                idx_to_deposit,
                rpbins, mass_bin_edges[0], mass_bin_edges[1], zmax, box_length,
                theta
        )

    if RANK == 1:
        print("while exited")

# RANK 0 continues with the HMC
else:
    rng_key = jax.random.PRNGKey(42)

    # inverse transform theta for the initial position
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
        "satmerg_logvr_k":initial_hmc_params[13],
        "fq_cens_logm_crit":initial_hmc_params[14],
        "fq_cens_k":initial_hmc_params[15],
        "fq_cens_ylo":initial_hmc_params[16],
        "fq_cens_yhi":initial_hmc_params[17],
        "fq_satboost_logmhost_crit":initial_hmc_params[18],
        "fq_satboost_logmhost_k":initial_hmc_params[19],
        "fq_satboost_clusters":initial_hmc_params[20],
        "fq_sat_delay_time":initial_hmc_params[21],
        "fq_sat_tinfall_k":initial_hmc_params[22]
    }
        
    # split the key
    rng_key, sample_key, warmup_key = jax.random.split(rng_key, 3)

    print("Begin warmup", flush=True)
    n_warmup_steps = 2
    warmup = blackjax.window_adaptation(blackjax.nuts, error_fn, 
                                        is_mass_matrix_diagonal=False)
    (init_state, tuned_params), _ = warmup.run(warmup_key, initial_position,
                                               num_steps=n_warmup_steps)
    print("Warm up done", flush=True)

    #hmc = blackjax.nuts(error_fn, **tuned_params)

    # tmp
    # init_state = hmc.init(initial_position)

    # build kernel and inference loop
    hmc_kernel = jax.jit(hmc.step)

    # run
    n_iter = 10

    # inference loop
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    states = inference_loop(sample_key, hmc_kernel, init_state, n_iter)
    theta_final_hmc = [
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
    ]
    theta_final_hmc = np.array(theta_final_hmc, dtype=np.float64)
    theta_final = transform_hmc_to_model(theta_final_hmc, lower_bounds, upper_bounds)

    # we're done iterating, broadcast FALSE to stop the other ranks
    cont = False
    COMM.bcast(cont, root=0)
    print("opt done", flush=True)

    samples = states.position
    print(states.logdensity)

# 4) plot results 
# let's plot quenched and unquenched before and after optimization
# first we need initial and final rpwp measurements
if RANK > 0:
    theta_final = None
theta_final = COMM.bcast(theta_final, root=0)
# initial
w_q, dw_q, w_nq, dw_nq = compute_weight_and_jac_quench(
                            halos["logmpeak"],
                            halos["loghost_mpeak"],
                            halos["logvmax_frac"],
                            halos["upid"],
                            halos["time_since_infall"],
                            idx_to_deposit,
                            mass_bin_edges[0], mass_bin_edges[1],
                            theta
)

wgt_mask_quench = w_q > 0
wgt_mask_no_quench = w_nq > 0

rpwp_q_init, _ = compute_rpwp(
                x1=halos["halo_x"][wgt_mask_quench],
                y1=halos["halo_y"][wgt_mask_quench],
                z1=halos["halo_z"][wgt_mask_quench],
                w1=w_q[wgt_mask_quench],
                w1_jac=dw_q[:, wgt_mask_quench],
                inside_subvol=halos["_inside_subvol"][wgt_mask_quench],
                rpbins=rpbins,
                zmax=zmax,
                boxsize=box_length
)

rpwp_nq_init, _ = compute_rpwp(
                x1=halos["halo_x"][wgt_mask_no_quench],
                y1=halos["halo_y"][wgt_mask_no_quench],
                z1=halos["halo_z"][wgt_mask_no_quench],
                w1=w_nq[wgt_mask_no_quench],
                w1_jac=dw_nq[:, wgt_mask_no_quench],
                inside_subvol=halos["_inside_subvol"][wgt_mask_no_quench],
                rpbins=rpbins,
                zmax=zmax,
                boxsize=box_length
)

# final
w_q, dw_q, w_nq, dw_nq = compute_weight_and_jac_quench(
                            halos["logmpeak"],
                            halos["loghost_mpeak"],
                            halos["logvmax_frac"],
                            halos["upid"],
                            halos["time_since_infall"],
                            idx_to_deposit,
                            mass_bin_edges[0], mass_bin_edges[1],
                            theta_final
)

wgt_mask_quench = w_q > 0
wgt_mask_no_quench = w_nq > 0

rpwp_q_final, _ = compute_rpwp(
                x1=halos["halo_x"][wgt_mask_quench],
                y1=halos["halo_y"][wgt_mask_quench],
                z1=halos["halo_z"][wgt_mask_quench],
                w1=w_q[wgt_mask_quench],
                w1_jac=dw_q[:, wgt_mask_quench],
                inside_subvol=halos["_inside_subvol"][wgt_mask_quench],
                rpbins=rpbins,
                zmax=zmax,
                boxsize=box_length
)

rpwp_nq_final, _ = compute_rpwp(
                x1=halos["halo_x"][wgt_mask_no_quench],
                y1=halos["halo_y"][wgt_mask_no_quench],
                z1=halos["halo_z"][wgt_mask_no_quench],
                w1=w_nq[wgt_mask_no_quench],
                w1_jac=dw_nq[:, wgt_mask_no_quench],
                inside_subvol=halos["_inside_subvol"][wgt_mask_no_quench],
                rpbins=rpbins,
                zmax=zmax,
                boxsize=box_length
)
    
# figure
if RANK == 0:
    fig, axs = plt.subplots(1, 2, figsize=(16,8), facecolor="w")

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
   
    plt.savefig("rpwp_nuts_demo.png")

