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
    satellite_disruption_probability_jax,
)
from diffsmhm.galhalo_models.merging import (
    deposit_stellar_mass,
    _calculate_indx_to_deposit
)
from diffsmhm.diff_stats.cuda.tw_kernels import (
    tw_kern_mstar_bin_weights_and_derivs_cuda
)


# functions for differentiable stellar mass and scatter

# Function to obtain net stellar mass
# note: even though not all params are used, we want deriv wrt all params
def _net_stellar_mass(
    hm,
    host_hm,
    log_vmax_by_vmpeak,
    idx_to_deposit,
    smhm_logm_crit,
    smhm_ratio_logm_crit,
    smhm_k_logm,
    smhm_lowm_index,
    smhm_highm_index,
    smhm_sigma_low,
    smhm_sigma_high,
    smhm_sigma_logm_pivot,
    smhm_sigma_logm_width,
    satmerg_logmhost_crit,
    satmerg_logmhost_k,
    satmerg_logvr_crit_dwarfs,
    satmerg_logvr_crit_clusters,
    satmerge_logvr_k
):

    # munge params into arrays
    smhm_params = [
        smhm_logm_crit,
        smhm_ratio_logm_crit,
        smhm_k_logm,
        smhm_lowm_index,
        smhm_highm_index,
    ]
    smhm_sigma_params = [
        smhm_sigma_low,
        smhm_sigma_high,
        smhm_sigma_logm_pivot,
        smhm_sigma_logm_width
    ]
    disruption_params = [
        satmerg_logmhost_crit,
        satmerg_logmhost_k,
        satmerg_logvr_crit_dwarfs,
        satmerg_logvr_crit_clusters,
        satmerge_logvr_k
    ]
        
    # to avoid flake complaining about unused vars :)
    smhm_params *= 1
    smhm_sigma_params *= 1
    disruption_params *= 1
    
    # stellar mass and scatter
    stellar_mass = logsm_from_logmhalo_jax(hm, smhm_params)

    # merging probability
    merging_prob = satellite_disruption_probability_jax(log_vmax_by_vmpeak, host_hm, disruption_params)

    # add/subtract merged mass
    net_stellar_mass = deposit_stellar_mass(stellar_mass, idx_to_deposit, merging_prob)
    
    return jnp.log10(net_stellar_mass)


# wrapper function for smhm scatter, makes jacobian easier bc we separate params
def _stellar_mass_sigma_wrapper(
    hm,
    smhm_logm_crit,
    smhm_ratio_logm_crit,
    smhm_k_logm,
    smhm_lowm_index,
    smhm_highm_index,
    smhm_sigma_low,
    smhm_sigma_high,
    smhm_sigma_logm_pivot,
    smhm_sigma_logm_width,
    satmerg_logmhost_crit,
    satmerg_logmhost_k,
    satmerg_logvr_crit_dwarfs,
    satmerg_logvr_crit_clusters,
    satmerge_logvr_k
):

    # munge params into arrays
    smhm_params = [
        smhm_logm_crit,
        smhm_ratio_logm_crit,
        smhm_k_logm,
        smhm_lowm_index,
        smhm_highm_index,
    ]
    smhm_sigma_params = [
        smhm_sigma_low,
        smhm_sigma_high,
        smhm_sigma_logm_pivot,
        smhm_sigma_logm_width
    ]
    disruption_params = [
        satmerg_logmhost_crit,
        satmerg_logmhost_k,
        satmerg_logvr_crit_dwarfs,
        satmerg_logvr_crit_clusters,
        satmerge_logvr_k
    ]
        
    # to avoid flake complaining about unused vars :)
    smhm_params *= 1
    smhm_sigma_params *= 1
    disruption_params *= 1

    return logsm_sigma_from_logmhalo_jax(hm, smhm_sigma_params)


# gradients for net stellar mass and sigma stellar mass
_net_stellar_mass_jacobian = jax.jacfwd(_net_stellar_mass,
                                   argnums=[4,5,6,7,8,9,10,11,12,13,14,15,16,17])

_stellar_mass_sigma_jacobian = jax.jacfwd(_stellar_mass_sigma_wrapper,
                                    argnums=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]) 

# helper function to return sm and grad for vectorized parameter input
def compute_sm_and_jac(
    hm,
    host_hm,
    log_vmax_by_vmpeak,
    idx_to_deposit,
    theta
):
    sm = _net_stellar_mass(
        hm,
        host_hm,
        log_vmax_by_vmpeak,
        idx_to_deposit,
        theta[0], theta[1], theta[2], theta[3], theta[4],
        theta[5], theta[6], theta[7], theta[8],
        theta[9], theta[10], theta[11], theta[12], theta[13]
    )

    sm_jac = _net_stellar_mass_jacobian(
        hm,
        host_hm,
        log_vmax_by_vmpeak,
        idx_to_deposit,
        theta[0], theta[1], theta[2], theta[3], theta[4],
        theta[5], theta[6], theta[7], theta[8],
        theta[9], theta[10], theta[11], theta[12], theta[13]
    )

    return np.array(sm, dtype=np.float64), np.array(sm_jac, dtype=np.float64)
        
# helper function for sigma and grad for vectorized parameter input
def compute_sigma_and_jac(
    hm, 
    theta
):
    sigma = _stellar_mass_sigma_wrapper(
                hm, 
                theta[0], theta[1], theta[2], theta[3], theta[4],
                theta[5], theta[6], theta[7], theta[8],
                theta[9], theta[10], theta[11], theta[12], theta[13]
    )

    sigma_jac = _stellar_mass_sigma_jacobian(
                hm, 
                theta[0], theta[1], theta[2], theta[3], theta[4],
                theta[5], theta[6], theta[7], theta[8],
                theta[9], theta[10], theta[11], theta[12], theta[13]
    )

    return np.array(sigma, dtype=np.float64), np.array(sigma_jac, dtype=np.float64)
                

# func for weights and weight gradients
def compute_weight_and_jac(
    hm,
    host_hm,
    log_vmax_by_vmpeak,
    idx_to_deposit,
    massbinlow,
    massbinhigh,
    theta,
    threads=32,
    blocks=512
):
    # compute quantities
    sm, sm_jac = compute_sm_and_jac(hm, host_hm, log_vmax_by_vmpeak,
                                    idx_to_deposit, theta)
    sigma, sigma_jac = compute_sigma_and_jac(hm, theta)

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

    return w, dw
