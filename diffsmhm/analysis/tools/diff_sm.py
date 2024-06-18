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
    deposit_stellar_mass
)
from diffsmhm.diff_stats.cpu.tw_kernels import (
    tw_bin_jax_kern
)


# some jax setup
tw_bin_jax_kern_vmapped = jax.vmap(tw_bin_jax_kern, in_axes=[0, 0, None, None])

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
    logmpeak,
    loghost_mpeak,
    log_vmax_by_vmpeak,
    upid,
    idx_to_deposit,
    theta
):
    # munge params into arrays
    smhm_params, _, disruption_params = _munge_theta(theta)

    # stellar mass
    stellar_mass = logsm_from_logmhalo_jax(logmpeak, smhm_params)

    # merging probability
    merging_prob = disruption_probability_jax(
                        upid,
                        log_vmax_by_vmpeak,
                        loghost_mpeak,
                        disruption_params
    )

    # add/subtract merged mass
    net_stellar_mass = deposit_stellar_mass(stellar_mass, idx_to_deposit, merging_prob)

    return jnp.log10(net_stellar_mass)


# wrapper function for smhm scatter, makes jacobian easier bc we input full theta
def _stellar_mass_sigma_wrapper(
    logmpeak,
    theta
):
    # munge params into arrays
    _, smhm_sigma_params, _ = _munge_theta(theta)

    return logsm_sigma_from_logmhalo_jax(logmpeak, smhm_sigma_params)


# gradients for net stellar mass and sigma stellar mass

_stellar_mass_sigma_jacobian = jax.jacfwd(_stellar_mass_sigma_wrapper,
                                          argnums=1)


# gradient of quenching prob
def _quenching_prob_wrapper(
    logmpeak,
    loghost_mpeak,
    upid,
    time_since_infall,
    theta
):
    _, _, _, quenching_params = _munge_theta(theta)

    q = quenching_prob_jax(upid, logmpeak, loghost_mpeak,
                           time_since_infall, quenching_params)

    return q


_quenching_prob_jacobian = jax.jacfwd(_quenching_prob_wrapper,
                                      argnums=4)


def compute_quenching_prob_and_jac(
    *,
    logmpeak,
    loghost_mpeak,
    upid,
    time_since_infall,
    theta
):
    """
    Compute the quenching probability and its gradient.

    Parameters
    ---------
    logmpeak : array_like, shape (n_gals,)
        The array of log10 halo masses
    loghost_mpeak : array_like, shape (n_gals,)
        The array of log10 host halo masses
    upid : array_like, shape (n_gals,)
        The array of IDs for a (sub)halo. Should be -1 to indicate a (sub)halo
        has no parents.
    time_since_infall : array_like, shape (n_gals,)
        Time since infall for satellite halos.
    theta : array_like, shape (n_params,)
        Model parameters.

    Returns
    -------
    qprob : array-like, shape (n_gals,)
        Quenching probability for each halo.
    dqprob : array-like, shape(n_params, n_gals)
        Gradients of quenching probability.
    """

    qprob = _quenching_prob_wrapper(
                        logmpeak, loghost_mpeak,
                        upid,
                        time_since_infall,
                        theta
    )

    dqprob = _quenching_prob_jacobian(
                        logmpeak, loghost_mpeak,
                        upid,
                        time_since_infall,
                        theta
    )

    qprob = np.array(qprob, dtype=np.float64)
    dqprob = np.array(dqprob, dtype=np.float64).T

    return qprob, dqprob


# helper to take jax grads w.r.t theta only
def _compute_weight(
    theta,
    logmpeak,
    loghost_mpeak,
    log_vmax_by_vmpeak,
    upid,
    idx_to_deposit,
    mass_bin_low,
    mass_bin_high
):
    # compute weights
    sm = _net_stellar_mass(
        logmpeak,
        loghost_mpeak,
        log_vmax_by_vmpeak,
        upid,
        idx_to_deposit,
        theta
    )

    sigma = _stellar_mass_sigma_wrapper(logmpeak=logmpeak, theta=theta)

    # Use DLPack to create zero-copy cupy references to Jax arrays
    # don't need these with jax kernels but leaving them here rn for reference
    # sm_cp = cp.from_dlpack(jax.dlpack.to_dlpack(sm, copy=False))
    # sm_jac_cp = cp.from_dlpack(jax.dlpack.to_dlpack(sm_jac, copy=False)).T
    # sigma_cp = cp.from_dlpack(jax.dlpack.to_dlpack(sigma, copy=False))
    # sigma_jac_cp = cp.from_dlpack(jax.dlpack.to_dlpack(sigma_jac, copy=False)).T
    # w = cp.zeros(len(logmpeak), dtype=cp.float64)
    # dw = cp.zeros((sm_jac.shape[1], len(logmpeak)), dtype=cp.float64)

    w = tw_bin_jax_kern_vmapped(sm, sigma, mass_bin_low, mass_bin_high)

    return w


_compute_weight_grad = jax.jacfwd(_compute_weight, argnums=[0])


# func for weights and weight gradients
def compute_weight_and_jac(
    *,
    logmpeak,
    loghost_mpeak,
    log_vmax_by_vmpeak,
    upid,
    idx_to_deposit,
    mass_bin_low,
    mass_bin_high,
    theta,
):
    """
    Compute the bin weight and gradient for a given stellar mass bin

    Parameters
    ---------
    logmpeak : array_like, shape (n_gals,)
        The array of log10 halo masses
    loghost_mpeak : array_like, shape (n_gals,)
        The array of log10 host halo masses
    log_vmax_by_vmpeak : array_like, shape (n_gals,)
        The array of log10 maximum halo velocity divided by halo velocity at mpeak.
    upid : array_like, shape (n_gals,)
        The array of IDs for a (sub)halo. Should be -1 to indicate a (sub)halo
        has no parents.
    idx_to_deposit : array_like, shape (n_gals,)
        Index of each halo's UPID in the above arrays.
    mass_bin_low : float
        Lower limit of the stellar mass bin.
    mass_bin_high : float
        Upper limit of the stellar mass bin.
    theta : array_like, shape (n_params,)
        Model parameters.

    Returns
    -------
    w : array-like, shape (n_gals,)
        Bin weights for each galaxy/halo.
    dw : array-like, shape(n_params, n_gals)
        Gradients of bin weights.
    """
    w = _compute_weight(
            theta,
            logmpeak,
            loghost_mpeak,
            log_vmax_by_vmpeak,
            upid,
            idx_to_deposit,
            mass_bin_low,
            mass_bin_high
        )

    dw = _compute_weight_grad(
            theta,
            logmpeak,
            loghost_mpeak,
            log_vmax_by_vmpeak,
            upid,
            idx_to_deposit,
            mass_bin_low,
            mass_bin_high
    )

    return w, dw


# func for weights and weight gradients
def compute_weight_and_jac_quench(
    *,
    logmpeak,
    loghost_mpeak,
    log_vmax_by_vmpeak,
    upid,
    time_since_infall,
    idx_to_deposit,
    mass_bin_low,
    mass_bin_high,
    theta,
    threads=32,
    blocks=512
):
    """
    Compute stellar mass bin weight for quenched/unquenched galaxies.

    Parameters
    ----------
    logmpeak : array_like, shape (n_gals,)
        The array of log10 halo masses
    loghost_mpeak : array_like, shape (n_gals,)
        The array of log10 host halo masses
    log_vmax_by_vmpeak : array_like, shape (n_gals,)
        The array of log10 maximum halo velocity divided by halo velocity at mpeak.
    upid : array_like, shape (n_gals,)
        The array of parent IDs for a (sub)halo. Should be -1 to indicate a (sub)halo
        has no parents.
    time_since_infall : array_like, shape (n_gals,)
        Time since infall for satellite halos.
    idx_to_deposit : array_like, shape (n_gals,)
        Index of each halo's UPID in the above arrays.
    mass_bin_low : float
        Lower limit of the stellar mass bin.
    mass_bin_high : float
        Upper limit of the stellar mass bin.
    theta : array_like, shape (n_params,)
        Model parameters.

    Returns
    -------
    w_quench : array-like, shape (n_gals,)
        Stellar mass bin weights for quenched galaxies.
    dw_quench : array_like, shape (n_params, n_gals)
        Gradients of stellar mass bin weights for quenched galaxies.
    w_no_quench : array-like, shape (n_gals,)
        Stellar mass bin weights for unquenched galaxies.
    dw_no_quench : array-like, shape (n_params, n_gals)
        Gradients of stellar mass bin weights for unquenched galaxies.
    """
    # compute weights
    w, dw = compute_weight_and_jac(
                logmpeak=logmpeak,
                loghost_mpeak=loghost_mpeak,
                log_vmax_by_vmpeak=log_vmax_by_vmpeak,
                upid=upid,
                idx_to_deposit=idx_to_deposit,
                mass_bin_low=mass_bin_low,
                mass_bin_high=mass_bin_high,
                theta=theta
    )

    # quenching probability
    q, dq = compute_quenching_prob_and_jac(
                        logmpeak=logmpeak,
                        loghost_mpeak=loghost_mpeak,
                        upid=upid,
                        time_since_infall=time_since_infall,
                        theta=theta
    )

    # multiply weights by quenched probability
    w_quench = w * q

    dw_quench = (w * dq) + (dw * q)

    # multiply weights by not-quenched probabilty
    w_no_quench = w * (1-q)

    dw_no_quench = (w * -1 * dq) + (dw * (1-q))

    return w_quench, dw_quench, w_no_quench, dw_no_quench
