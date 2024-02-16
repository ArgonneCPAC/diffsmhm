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
    DEFAULT_PARAM_VALUES as smhm_params
)
from diffsmhm.galhalo_models.sigmoid_smhm_sigma import (
    DEFAULT_PARAM_VALUES as smhm_sigma_params
 )
from diffsmhm.galhalo_models.sigmoid_disruption import (
    DEFAULT_PARAM_VALUES as disruption_params
)
from diffsmhm.galhalo_models.sigmoid_quenching import (
    DEFAULT_PARAM_VALUES as quenching_params
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

mass_bin_edges = np.array([10.6, 10.7], dtype=np.float64)

rpbins = np.logspace(-1, 1.5, 13, dtype=np.float64)
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
    smhm_sigma_params["smhm_sigma_logm_pivot"],
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

n_params = len(theta)
n_rpbins = len(rpbins) - 1

# 1) load data
halos, _ = load_and_chop_data_bolshoi_planck(
                particle_file,
                halo_file,
                box_length,
                buff_wprp,
                host_mpeak_cut=12.0
)

# 1) load data
halos, _ = load_and_chop_data_bolshoi_planck(
                particle_file,
                halo_file,
                box_length,
                buff_wprp,
                host_mpeak_cut=14.7
)

idx_to_deposit = _calculate_indx_to_deposit(halos["upid"], halos["halo_id"])


# 2) obtain "goal" measurement
parameter_perturbations = np.random.uniform(low=0.95, high=1.05, size=n_params)

theta_goal = theta * parameter_perturbations

# rpwp, quenched and unquenched
w_q, dw_q, w_nq, dw_nq = compute_weight_and_jac_quench(
                        halos["logmpeak"],
                        halos["loghost_mpeak"],
                        halos["vmax_frac"],
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

# 3) do optimization

# define functions

# error function set up
# note this function doesn't need to stick within if we use pure_callback
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
    zmax=zmax, boxsize=box_length
    
):
    # munge params into array
    theta = [
        smhm_logm_crit, smhm_ratio_logm_crit, smhm_k_logm, smhm_lowm_index,
        smhm_highm_index,

        smhm_sigma_low, smhm_sigma_high, smhm_sigma_logm_pivot, smhm_sigma_logm_width,

        satmerg_logmhost_crit, satmerg_logmhost_k, satmerg_logvr_crit_dwarfs,
        satmerg_logvr_crit_clusters, satmerg_logvr_k,

        fq_cens_logm_crit, fq_cens_k, fq_cens_ylo, fq_cens_yhi, fq_satboost_logmhost_crit,
        fq_satboost_logmhost_k, fq_satboost_clusters, fq_sat_delay_time, fq_sat_tinfall_k,
    ]
    theta = np.array(theta, dtype=np.float64)

    cont = True
    COMM.bcast(cont, root=0)

    COMM.bcast(theta, root=0)

    # call func for error and gradient
    err, err_grad = mse_rpwp_quench(
                theta,
                goal_q, goal_nq,
                logmass, log_hostmass, log_vmax_by_vmpeak,
                halo_x, halo_y, halo_z,
                upid, inside_subvol, time_since_infall, idx_to_deposit,
                rpbins, mass_bin_low, mass_bin_high, zmax, boxsize
    )

    # TMP
    print(err)

    # return that error
    return np.array(err), err_grad

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
        theta = COMM.bcast(theta, root=0)

        # do computation
        _, _ = mse_rpwp_quench(
                theta,
                rpwp_q_goal, rpwp_nq_goal,
                halos["logmpeak"], halos["loghost_mpeak"], halos["logvmax_frac"],
                halos["halo_x"], halos["halo_y"], halos["halo_z"],
                halos["upid"], halos["_inside_subvol"],
                halos["time_since_infall"],
                idx_to_deposit,
                rpbins, mass_bin_edges[0], mass_bin_edges[1], zmax, box_length
        )

    if RANK == 1:
        print("while exited")

# RANK 0 continues with the HMC
else:
    rng_key = jax.random.PRNGKey(42)

    # adapt the mass matrix
    initial_position = {
        "smhm_logm_crit":smhm_params["smhm_logm_crit"],
        "smhm_ratio_logm_crit":smhm_params["smhm_ratio_logm_crit"],
        "smhm_k_logm":smhm_params["smhm_k_logm"],
        "smhm_lowm_index":smhm_params["smhm_lowm_index"],
        "smhm_highm_index":smhm_params["smhm_highm_index"],
        "smhm_sigma_low":smhm_sigma_params["smhm_sigma_low"],
        "smhm_sigma_high":smhm_sigma_params["smhm_sigma_high"],
        "smhm_sigma_logm_pivot":smhm_sigma_params["smhm_sigma_logm_pivot"],
        "smhm_sigma_logm_width":smhm_sigma_params["smhm_sigma_logm_width"],
        "satmerg_logmhost_crit":disruption_params["satmerg_logmhost_crit"],
        "satmerg_logmhost_k":disruption_params["satmerg_logmhost_k"],
        "satmerg_logvr_crit_dwarfs":disruption_params["satmerg_logvr_crit_dwarfs"],
        "satmerg_logvr_crit_clusters":disruption_params["satmerg_logvr_crit_clusters"],
        "satmerg_logvr_k":disruption_params["satmerg_logvr_k"],
        "fq_cens_logm_crit":quenching_params["fq_cens_logm_crit"],
        "fq_cens_k":quenching_params["fq_cens_k"],
        "fq_cens_ylo":quenching_params["fq_cens_ylo"],
        "fq_cens_yhi":quenching_params["fq_cens_yhi"],
        "fq_satboost_logmhost_crit":quenching_params["fq_satboost_logmhost_crit"],
        "fq_satboost_logmhost_k":quenching_params["fq_satboost_logmhost_k"],
        "fq_satboost_clusters":quenching_params["fq_satboost_clusters"],
        "fq_sat_delay_time":quenching_params["fq_sat_delay_time"],
        "fq_sat_tinfall_k":quenching_params["fq_sat_tinfall_k"]
    }
        
    # split the key
    rng_key, sample_key, warmup_key = jax.random.split(rng_key, 3)

    # temporarily not using this while we figure out conda issue
    # warmup = blackjax.window_adaptation(blackjax.nuts, error)
    # (init_state, tuned_params), _ = warmup.run(rng_key, initial_position)

    step_size=1e-5
    inv_mass_matrix = np.ones(len(theta))
    #hmc = blackjax.nuts(error, **tuned_params)
    hmc = blackjax.nuts(error_fn, step_size, inv_mass_matrix)

    # tmp
    init_state = hmc.init(initial_position)

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

    # we're done iterating, broadcast FALSE to stop the other ranks
    cont = False
    COMM.bcast(cont, root=0)

    # some sanity check business
    samples = states.position
    # errors = states.logdensity # this works in my nb; maybe a versioning thing?


