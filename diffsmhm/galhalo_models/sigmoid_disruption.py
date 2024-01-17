"""Model for the probability of satellite/subhalo disruption."""
import numpy as np
from collections import OrderedDict
from jax import numpy as jax_np
import jax

from .utils import get_1d_arrays, jax_sigmoid


__all__ = ("satellite_disruption_probability", "disruption_probability")


DEFAULT_PARAM_VALUES = OrderedDict(
    satmerg_logmhost_crit=13.5,
    satmerg_logmhost_k=10 ** 0.5,
    satmerg_logvr_crit_dwarfs=-1.0,
    satmerg_logvr_crit_clusters=-0.7,
    satmerg_logvr_k=10 ** 0.5,
)


PARAM_BOUNDS = OrderedDict(
    satmerg_logmhost_crit=(12, 15),
    satmerg_logmhost_k=(0, 10),
    satmerg_logvr_crit_dwarfs=(-2, 0),
    satmerg_logvr_crit_clusters=(-2, 0),
    satmerg_logvr_k=(0, 10),
)


def disruption_probability(
    upid,
    log_vmax_by_vmpeak,
    logmhost,
    satmerg_logmhost_crit=None,
    satmerg_logmhost_k=None,
    satmerg_logvr_crit_dwarfs=None,
    satmerg_logvr_crit_clusters=None,
    satmerg_logvr_k=None,
):
    """Disruption probability of (sub)halos as a function of halo properties.

    Parameters
    ----------
    upid : float or ndarray of shape (nhalos, )
        The ID of the parent halo of a (sub)halo. Should be -1 to indicate a
        (sub)halo has no parents.
    log_vmax_by_vmpeak : ndarray of shape (nhalos, )
        Base-10 log of (sub)halo Vmax/Vmpeak
    logmhost : ndarray of shape (nhalos, )
        Base-10 log of host halo mass
    satmerg_logmhost_crit : float, optional
        Value of log10(Mhost) of the inflection point of prob_disrupt
    satmerg_logmhost_k : float, optional
        Steepness of the sigmoid in log10(Mhost)
    satmerg_logvr_crit_dwarfs : float, optional
        Disruption cutoff in log10(Vmax/Vmpeak) in dwarf-mass host halos
    satmerg_logvr_crit_clusters : float, optional
        Disruption cutoff in log10(Vmax/Vmpeak) in cluster-mass host halos
    satmerg_logvr_k : float, optional
        Steepness of the sigmoid in log10(Vmax/Vmpeak)

    Returns
    -------
    prob_disrupt : ndarray of shape (nhalos, )
        The probability that a given (sub)halo will be disrupted.

    """
    upid, log_vmax_by_vmpeak, logmhost = get_1d_arrays(
        upid, log_vmax_by_vmpeak, logmhost
    )

    satmerg_logmhost_crit = (
        DEFAULT_PARAM_VALUES["satmerg_logmhost_crit"]
        if satmerg_logmhost_crit is None
        else satmerg_logmhost_crit
    )
    satmerg_logmhost_k = (
        DEFAULT_PARAM_VALUES["satmerg_logmhost_k"]
        if satmerg_logmhost_k is None
        else satmerg_logmhost_k
    )
    satmerg_logvr_crit_dwarfs = (
        DEFAULT_PARAM_VALUES["satmerg_logvr_crit_dwarfs"]
        if satmerg_logvr_crit_dwarfs is None
        else satmerg_logvr_crit_dwarfs
    )
    satmerg_logvr_crit_clusters = (
        DEFAULT_PARAM_VALUES["satmerg_logvr_crit_clusters"]
        if satmerg_logvr_crit_clusters is None
        else satmerg_logvr_crit_clusters
    )
    satmerg_logvr_k = (
        DEFAULT_PARAM_VALUES["satmerg_logvr_k"]
        if satmerg_logvr_k is None
        else satmerg_logvr_k
    )

    params = np.array(
        [
            satmerg_logmhost_crit,
            satmerg_logmhost_k,
            satmerg_logvr_crit_dwarfs,
            satmerg_logvr_crit_clusters,
            satmerg_logvr_k,
        ]
    )

    return np.asarray(
        disruption_probability_jax(upid, log_vmax_by_vmpeak, logmhost, params)
    )


def satellite_disruption_probability(
    log_vmax_by_vmpeak,
    logmhost,
    satmerg_logmhost_crit=None,
    satmerg_logmhost_k=None,
    satmerg_logvr_crit_dwarfs=None,
    satmerg_logvr_crit_clusters=None,
    satmerg_logvr_k=None,
):
    """Disruption probability of satellite (sub)halos as a function of halo properties.

    Parameters
    ----------
    log_vmax_by_vmpeak : ndarray of shape (nhalos, )
        Base-10 log of (sub)halo Vmax/Vmpeak
    logmhost : ndarray of shape (nhalos, )
        Base-10 log of host halo mass
    satmerg_logmhost_crit : float, optional
        Value of log10(Mhost) of the inflection point of prob_disrupt
    satmerg_logmhost_k : float, optional
        Steepness of the sigmoid in log10(Mhost)
    satmerg_logvr_crit_dwarfs : float, optional
        Disruption cutoff in log10(Vmax/Vmpeak) in dwarf-mass host halos
    satmerg_logvr_crit_clusters : float, optional
        Disruption cutoff in log10(Vmax/Vmpeak) in cluster-mass host halos
    satmerg_logvr_k : float, optional
        Steepness of the sigmoid in log10(Vmax/Vmpeak)

    Returns
    -------
    prob_disrupt : ndarray of shape (nhalos, )
        The probability that a given (sub)halo will be disrupted.

    """
    log_vmax_by_vmpeak, logmhost = get_1d_arrays(log_vmax_by_vmpeak, logmhost)

    satmerg_logmhost_crit = (
        DEFAULT_PARAM_VALUES["satmerg_logmhost_crit"]
        if satmerg_logmhost_crit is None
        else satmerg_logmhost_crit
    )
    satmerg_logmhost_k = (
        DEFAULT_PARAM_VALUES["satmerg_logmhost_k"]
        if satmerg_logmhost_k is None
        else satmerg_logmhost_k
    )
    satmerg_logvr_crit_dwarfs = (
        DEFAULT_PARAM_VALUES["satmerg_logvr_crit_dwarfs"]
        if satmerg_logvr_crit_dwarfs is None
        else satmerg_logvr_crit_dwarfs
    )
    satmerg_logvr_crit_clusters = (
        DEFAULT_PARAM_VALUES["satmerg_logvr_crit_clusters"]
        if satmerg_logvr_crit_clusters is None
        else satmerg_logvr_crit_clusters
    )
    satmerg_logvr_k = (
        DEFAULT_PARAM_VALUES["satmerg_logvr_k"]
        if satmerg_logvr_k is None
        else satmerg_logvr_k
    )

    params = np.array(
        [
            satmerg_logmhost_crit,
            satmerg_logmhost_k,
            satmerg_logvr_crit_dwarfs,
            satmerg_logvr_crit_clusters,
            satmerg_logvr_k,
        ]
    )

    return np.asarray(
        satellite_disruption_probability_jax(log_vmax_by_vmpeak, logmhost, params)
    )


def _disruption_probability_jax_kern(upid, log_vmax_by_vmpeak, logmhost, params):
    """Compute the probability of disruption for a (sub)halo.

    Parameters
    ----------
    upid : float or ndarray of shape (nhalos, )
        The ID of the parent halo of a (sub)halo. Should be -1 to indicate a
        (sub)halo has no parents.
    log_vmax_by_vmpeak : ndarray of shape (nhalos, )
        Base-10 log of (sub)halo Vmax/Vmpeak
    logmhost : ndarray of shape (nhalos, )
        Base-10 log of host halo mass
    params : array-like, shape (5,)
        An array with the parameters

            satmerg_logmhost_crit
            satmerg_logmhost_k
            satmerg_logvr_crit_dwarfs
            satmerg_logvr_crit_clusters
            satmerg_logvr_k

        See the documentation of the function `disruption_probability` for
        their definitions.

    Returns
    -------
    prob_disrupt : ndarray of shape (nhalos, )
        The probability that a given (sub)halo has been disrupted.

    """
    return jax_np.where(
        upid == -1,
        # centrals never disrupt
        0,
        # sats might!
        _satellite_disruption_probability_jax_kern(
            log_vmax_by_vmpeak, logmhost, params
        ),
    )


disruption_probability_jax = jax.jit(
    jax.vmap(_disruption_probability_jax_kern, in_axes=(0, 0, 0, None))
)
disruption_probability_jax.__doc__ = _disruption_probability_jax_kern.__doc__


def _satellite_disruption_probability_jax_kern(log_vmax_by_vmpeak, logmhost, params):
    """Compute the probability of disruption for a satellite (sub)halo.

    Parameters
    ----------
    log_vmax_by_vmpeak : ndarray of shape (nhalos, )
        Base-10 log of (sub)halo Vmax/Vmpeak
    logmhost : ndarray of shape (nhalos, )
        Base-10 log of host halo mass
    params : array-like, shape (5,)
        An array with the parameters

            satmerg_logmhost_crit
            satmerg_logmhost_k
            satmerg_logvr_crit_dwarfs
            satmerg_logvr_crit_clusters
            satmerg_logvr_k

        See the documentation of the function `satellite_disruption_probability` for
        their definitions.

    Returns
    -------
    prob_disrupt : ndarray of shape (nhalos, )
        The probability that a given (sub)halo has been disrupted.

    """
    logvr = log_vmax_by_vmpeak
    x0 = _disruption_prob_logvr_crit_mhost_dependence(logmhost, params)
    k = params[4]
    ylo, yhi = 1, 0

    return jax_sigmoid(logvr, x0, k, ylo, yhi)


satellite_disruption_probability_jax = jax.jit(
    jax.vmap(_satellite_disruption_probability_jax_kern, in_axes=(0, 0, None))
)
satellite_disruption_probability_jax.__doc__ = (
    _satellite_disruption_probability_jax_kern.__doc__
)


def _disruption_prob_logvr_crit_mhost_dependence(logmhost, params):
    """Host mass dep of disruption prob for sats. See the doc string
    of `satellite_disruption_probability` for more details on the arguments
    and what should be in `params`.
    """
    x0 = params[0]
    k = params[1]
    ylo = params[2]
    yhi = params[3]

    return jax_sigmoid(logmhost, x0, k, ylo, yhi)
