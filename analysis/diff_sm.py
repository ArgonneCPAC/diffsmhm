import numpy as np
import jax
import jax.numpy as jnp

from diffsmhm.galhalo_models.sigmoid_smhm import (
    logsm_from_logmhalo_jax,
)
from diffsmhm.galhalo_models.sigmoid_smhm_sigma import (
    logsm_sigma_from_logmhalo_jax,
)
from diffsmhm.galhalo_models.sigmoid_disruption import (
    disruption_probability_jax,
)
from diffsmhm.galhalo_models.sigmoid_quenching import (
    quenching_prob_jax
)
from diffsmhm.galhalo_models.merging import (
    deposit_stellar_mass,
    _calculate_indx_to_deposit
)
from diffsmhm.diff_stats.cuda.tw_kernels import (
    tw_kern_mstar_bin_weights_and_derivs_cuda
)

# for debugging
from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


# functions for differentiable stellar mass and scatter

# munge theta into parameter areas
def _munge_theta(theta):
    smhm_params = theta[0:5]
    smhm_sigma_params = theta[5:9]
    disruption_params = theta[9:14]
    quenching_params = theta[14:]

    return smhm_params, smhm_sigma_params, disruption_params, quenching_params

# Function to obtain net stellar mass
# note: even though not all params are used, we want deriv wrt all params
def _net_stellar_mass(
    hm,
    host_hm,
    log_vmax_by_vmpeak,
    upid,
    idx_to_deposit,
    theta
):

    # munge params into arrays
    smhm_params, _, disruption_params, _ = _munge_theta(theta)        
    #smhm_params = jnp.array([11.35, -1.65, 1.58489319, 2.5, 0.5])

    # stellar mass
    stellar_mass = logsm_from_logmhalo_jax(hm, smhm_params)

    #print("sm:", jnp.max(stellar_mass))

    # merging probability
    merging_prob = disruption_probability_jax(
                        upid,
                        log_vmax_by_vmpeak,
                        host_hm,
                        disruption_params
    )

    #print("mp:", jnp.max(merging_prob))

    # add/subtract merged mass
    net_stellar_mass = deposit_stellar_mass(stellar_mass, idx_to_deposit, merging_prob)
    log_net_stellar_mass = jnp.log10(net_stellar_mass)
    
    #print("nsm:", jnp.max(net_stellar_mass))

    #print("diff:", max(jnp.log10(net_stellar_mass)-stellar_mass), flush=True)

    return log_net_stellar_mass


# wrapper function for smhm scatter, makes jacobian easier bc we input full theta
def _stellar_mass_sigma_wrapper(
    hm,
    theta
):

    # munge params into arrays
    _, smhm_sigma_params, _, _ = _munge_theta(theta)

    return logsm_sigma_from_logmhalo_jax(hm, smhm_sigma_params)


# gradients for net stellar mass and sigma stellar mass
_net_stellar_mass_jacobian = jax.jacfwd(_net_stellar_mass,
                                   argnums=5)

_stellar_mass_sigma_jacobian = jax.jacfwd(_stellar_mass_sigma_wrapper,
                                    argnums=1) 

# helper function to return sm and grad for vectorized parameter input
def compute_sm_and_jac(
    hm,
    host_hm,
    log_vmax_by_vmpeak,
    upid,
    idx_to_deposit,
    theta
):
    sm = _net_stellar_mass(
        hm,
        host_hm,
        log_vmax_by_vmpeak,
        upid,
        idx_to_deposit,
        theta
    )

    sm_jac = _net_stellar_mass_jacobian(
        hm,
        host_hm,
        upid,
        log_vmax_by_vmpeak,
        idx_to_deposit,
        theta
    )

    sm = np.array(sm, dtype=np.float64)
    sm_jac = np.array(sm_jac, dtype=np.float64).T

    return sm, sm_jac
        

# helper function for sigma and grad for vectorized parameter input
def compute_sm_sigma_and_jac(
    hm, 
    theta
):
    sm_sigma = _stellar_mass_sigma_wrapper(
                hm, 
                theta
    )

    sm_sigma_jac = _stellar_mass_sigma_jacobian(
                hm, 
                theta
    )

    sm_sigma = np.array(sm_sigma, dtype=np.float64)
    sm_sigma_jac = np.array(sm_sigma_jac, dtype=np.float64).T

    return sm_sigma, sm_sigma_jac
                

# gradient of quenching prob
def _quenching_prob_wrapper(
    upid,
    logmpeak,
    logmhost,
    time_since_infall,
    theta
):
        
    _, _, _, quenching_params = _munge_theta(theta)

    q = quenching_prob_jax(upid, logmpeak, logmhost,
                            time_since_infall, quenching_params)
                            
    return q


_quenching_prob_jacobian = jax.jacfwd(_quenching_prob_wrapper,
                                      argnums=4)


def compute_quenching_prob_and_jac(
    upid,
    logmpeak,
    logmhost,
    time_since_infall,
    theta
):
    
    q = _quenching_prob_wrapper(
                        upid,
                        logmpeak, logmhost,
                        time_since_infall,
                        theta
    )

    dq = _quenching_prob_jacobian(
                        upid,
                        logmpeak, logmhost,
                        time_since_infall,
                        theta
    )

    q = np.array(q, dtype=np.float64)
    dq = np.array(dq, dtype=np.float64).T

    return q, dq


# func for weights and weight gradients
def compute_weight_and_jac(
    hm,
    host_hm,
    log_vmax_by_vmpeak,
    upid,
    idx_to_deposit,
    massbinlow,
    massbinhigh,
    theta,
    threads=32,
    blocks=512
):
    # compute weights
    sm, sm_jac = compute_sm_and_jac(hm, host_hm, log_vmax_by_vmpeak,
                                    upid, idx_to_deposit, theta)
    sigma, sigma_jac = compute_sm_sigma_and_jac(hm, theta)

    w = np.zeros(len(hm), dtype=np.float64)
    dw = np.zeros((sm_jac.shape[0], len(hm)), dtype=np.float64)

    tw_kern_mstar_bin_weights_and_derivs_cuda[blocks, threads](
                                        sm,
                                        sm_jac,
                                        sigma,
                                        sigma_jac,
                                        massbinlow, massbinhigh,
                                        w,
                                        dw
    )

    #print(RANK, "dw: ", np.sum(dw, axis=1), flush=True)

    return w, dw


# func for weights and weight gradients
def compute_weight_and_jac_quench(
    hm,
    host_hm,
    log_vmax_by_vmpeak,
    upid,
    time_since_infall,
    idx_to_deposit,
    massbinlow,
    massbinhigh,
    theta,
    threads=32,
    blocks=512
):
    # compute weights
    sm, sm_jac = compute_sm_and_jac(hm, host_hm, log_vmax_by_vmpeak,
                                    upid, idx_to_deposit, theta)
    sigma, sigma_jac = compute_sm_sigma_and_jac(hm, theta)

    w = np.zeros(len(hm), dtype=np.float64)
    dw = np.zeros((sm_jac.shape[0], len(hm)), dtype=np.float64)

    tw_kern_mstar_bin_weights_and_derivs_cuda[blocks, threads](
                                        sm,
                                        sm_jac,
                                        sigma,
                                        sigma_jac,
                                        massbinlow, massbinhigh,
                                        w,
                                        dw
    )

    # quenching probability
    q, dq = compute_quenching_prob_and_jac(
                        upid,
                        hm, host_hm,
                        time_since_infall,
                        theta
    )

    #print("dq:", np.sum(dq, axis=1), flush=False)

    # multiply weights by quenched probability
    w_quench = w * q

    dw_quench = (w * dq) + (dw * q)

    print(RANK, "dwq: ", np.sum(dw_quench, axis=1), flush=True)

    # multiply weights by not-quenched probabilty
    w_no_quench = w * (1-q)

    dw_no_quench = (w * -1 * dq) + (dw * (1-q))

    print(RANK, "dwnq:", np.sum(dw_no_quench, axis=1), flush=True)

    return w_quench, dw_quench, w_no_quench, dw_no_quench
