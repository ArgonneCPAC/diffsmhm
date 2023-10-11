"""Sigmoid-based models for the stellar-to-halo-mass relation."""
import numpy as np
from collections import OrderedDict
import jax
from .utils import jax_sigmoid


__all__ = [
    "logsm_sigma_from_logmhalo",
    "logsm_sigma_from_logmhalo_jax",
    "logsm_sigma_from_logmhalo_jax_kern",
]


DEFAULT_PARAM_VALUES = OrderedDict(
    smhm_sigma_low=0.3,
    smhm_sigma_high=0.2,
    smhm_sigma_logm_pivot=12.0,
    smhm_sigma_logm_width=0.1,
)

PARAM_BOUNDS = OrderedDict(
    smhm_sigma_low=(0.1, 0.4),
    smhm_sigma_high=(0.1, 0.4),
    smhm_sigma_logm_pivot=(11.0, 13.0),
    smhm_sigma_logm_width=(0.01, 0.2),
)


def logsm_sigma_from_logmhalo(
    logm,
    smhm_sigma_low=None,
    smhm_sigma_high=None,
    smhm_sigma_logm_pivot=None,
    smhm_sigma_logm_width=None,
):
    """The scatter in the SMHM relation as a function of logm.

    Parameters
    ----------
    logm : float
        Base-10 log of the halo mass, usually Mpeak.
    smhm_sigma_low : float, optional
        The value of sigma in the low mass limit.
    smhm_sigma_high : float, optional
        The value of sigma in the high mass limit.
    smhm_sigma_logm_pivot : float, optional
        The pivot in base-10 logm where the scatter is halfway between
        the low and high values.
    smhm_sigma_logm_width : float, optional
        The width of the transition region between
        the low and high values in base-10 logm

    Returns
    -------
    smhm_sigma : float
        The scatter in the SMHM relation.
    """
    if smhm_sigma_low is None:
        smhm_sigma_low = DEFAULT_PARAM_VALUES["smhm_sigma_low"]
    if smhm_sigma_high is None:
        smhm_sigma_high = DEFAULT_PARAM_VALUES["smhm_sigma_high"]
    if smhm_sigma_logm_pivot is None:
        smhm_sigma_logm_pivot = DEFAULT_PARAM_VALUES["smhm_sigma_logm_pivot"]
    if smhm_sigma_logm_width is None:
        smhm_sigma_logm_width = DEFAULT_PARAM_VALUES["smhm_sigma_logm_width"]

    params = np.array(
        [
            smhm_sigma_low,
            smhm_sigma_high,
            smhm_sigma_logm_pivot,
            smhm_sigma_logm_width,
        ]
    )

    return np.asarray(logsm_sigma_from_logmhalo_jax_kern(logm, params))


def logsm_sigma_from_logmhalo_jax_kern(logm, params):
    """The scatter in the SMHM relation as a function of logm.

    Parameters
    ----------
    logm : float
        Base-10 log of the halo mass, usually Mpeak.
    params : array of length four
        The parameters of the relation between the scatter and logm.
        The order is

            smhm_sigma_low
            smhm_sigma_high
            smhm_sigma_logm_pivot
            smhm_sigma_logm_width

        See the docstring of "logsm_sigma_from_logmhalo" for a description.

    Returns
    -------
    smhm_sigma : float
        The scatter in the SMHM relation.
    """
    smhm_sigma_low = params[0]
    smhm_sigma_high = params[1]
    smhm_sigma_logm_pivot = params[2]
    smhm_sigma_logm_width = params[3]

    return jax_sigmoid(
        logm,
        smhm_sigma_logm_pivot,
        smhm_sigma_logm_width,
        smhm_sigma_low,
        smhm_sigma_high,
    )


logsm_sigma_from_logmhalo_jax = jax.jit(
    jax.vmap(logsm_sigma_from_logmhalo_jax_kern, in_axes=(0, None))
)
logsm_sigma_from_logmhalo_jax.__doc__ = logsm_sigma_from_logmhalo_jax_kern.__doc__
