"""Sigmoid-based models for the stellar-to-halo-mass relation."""
import numpy as np
from collections import OrderedDict
import jax
from jax import numpy as jax_np
from jax.nn import sigmoid as sigmoid


__all__ = [
    "logsm_from_logmhalo",
    "logsm_from_logmhalo_jax",
    "logsm_from_logmhalo_jax_kern",
]


DEFAULT_PARAM_VALUES = OrderedDict(
    smhm_logm_crit=11.35,
    smhm_ratio_logm_crit=-1.65,
    smhm_k_logm=10 ** 0.2,
    smhm_lowm_index=2.5,
    smhm_highm_index=0.5,
)

PARAM_BOUNDS = OrderedDict(
    smhm_logm_crit=(10.5, 12.5),
    smhm_ratio_logm_crit=(-2.5, -0.5),
    smhm_k_logm=(0, 2),
    smhm_lowm_index=(1.5, 3.5),
    smhm_highm_index=(0.1, 2),
)


def logsm_from_logmhalo(
    logmhalo,
    smhm_logm_crit=None,
    smhm_ratio_logm_crit=None,
    smhm_k_logm=None,
    smhm_lowm_index=None,
    smhm_highm_index=None,
):
    """Model for median mstar vs (sub)halo mass.

    Mstar = A*(Mhalo/Mcrit)**alpha

    The function alpha = alpha(logMhalo) is modeled as a sigmoid function.

    Parameters
    ----------
    logmhalo : ndarray
        Numpy array of shape (nhalos, )
    smhm_logm_crit : float, optional
        Value of log10(Mhalo) of the inflection point of the power law index
    smhm_ratio_logm_crit : float, optional
        The SMHM ratio at Mhalo = Mcrit.
    smhm_k_logm : float, optional
        Steepness of the sigmoid governing the power law index alpha(logMhalo)
    smhm_lowm_index : float, optional
        Power law index at low mass: alpha(-infty)
    smhm_highm_index : float, optional
        Power law index at high mass: alpha(+infty)

    Returns
    -------
    Mstar : ndarray
        Numpy array of shape (nhalos, )
    """
    logmhalo = np.atleast_1d(logmhalo).astype("f8")

    smhm_logm_crit = (
        DEFAULT_PARAM_VALUES["smhm_logm_crit"]
        if smhm_logm_crit is None
        else smhm_logm_crit
    )
    smhm_ratio_logm_crit = (
        DEFAULT_PARAM_VALUES["smhm_ratio_logm_crit"]
        if smhm_ratio_logm_crit is None
        else smhm_ratio_logm_crit
    )
    smhm_k_logm = (
        DEFAULT_PARAM_VALUES["smhm_k_logm"] if smhm_k_logm is None else smhm_k_logm
    )
    smhm_lowm_index = (
        DEFAULT_PARAM_VALUES["smhm_lowm_index"]
        if smhm_lowm_index is None
        else smhm_lowm_index
    )
    smhm_highm_index = (
        DEFAULT_PARAM_VALUES["smhm_highm_index"]
        if smhm_highm_index is None
        else smhm_highm_index
    )

    params = np.array(
        [
            smhm_logm_crit,
            smhm_ratio_logm_crit,
            smhm_k_logm,
            smhm_lowm_index,
            smhm_highm_index,
        ]
    )

    return np.asarray(logsm_from_logmhalo_jax(logmhalo, params))


def logsm_from_logmhalo_jax_kern(logm, params):
    """Compute stellar mass from halo mass.

    Parameters
    ----------
    logm : float, array-like
        Base-10 log of halo mass.
    params : float, array-like shape (5,)
        An array of parameters. The ordering is

            smhm_logm_crit
            smhm_ratio_logm_crit
            smhm_k_logm
            smhm_lowm_index
            smhm_highm_index

        See the doc string of `logsm_from_logmhalo` for a description.

    Returns
    -------
    logmstar : array-like
        Base-10 log of stellar mass.
    """
    logm_crit = params[0]
    smhm_ratio_logm_crit = params[1]
    smhm_k_logm = params[2]
    lowm_index = params[3]
    highm_index = params[4]
    logsm_at_logm_crit = logm_crit + smhm_ratio_logm_crit

    sig = sigmoid(smhm_k_logm * (logm - logm_crit))
    powerlaw_index = lowm_index + (highm_index - lowm_index) * sig
    return logsm_at_logm_crit + powerlaw_index * (logm - logm_crit)


logsm_from_logmhalo_jax = jax.jit(
    jax.vmap(logsm_from_logmhalo_jax_kern, in_axes=(0, None))
)
logsm_from_logmhalo_jax.__doc__ = logsm_from_logmhalo_jax_kern.__doc__
