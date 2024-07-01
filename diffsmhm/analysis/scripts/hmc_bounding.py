from jax.scipy.special import logit as logit
from jax.nn import sigmoid as sigmoid

import jax.numpy as jnp


def model_pos_to_hmc_pos(x, lower_bounds, upper_bounds):
    """
    Transform bounded "model position" to unbounded "HMC position." Note that we
    follow the convention of the STAN reference manual; `x` refers to the bounded
    variables, and `y` to their unbounded transformation (Ch. 10).

    Parameters
    ---------
    x : array_like, shape (n_params,)
        The array of bounded variables to transform.
    lower_bounds : array_like, shape (n_params,)
        The array of lower bounds for `x`.
    upper_bounds : array_like, shape (n_params,)
        The array of upper bounds for `x`.

    Returns
    -------
    y : array_like, shape (n_params,)
        The array of unbounded variables.
    """
    return logit((x - lower_bounds) / (upper_bounds - lower_bounds))


def hmc_pos_to_model_pos(y, lower_bounds, upper_bounds):
    """
    Transform unbounded "HMC position" to bounded "model position". Note that we
    follow the convention of the STAN reference manual; `x` refers to the bounded
    variables, and `y` to their unbounded transformation (Ch. 10).

    Parameters
    ----------
    y : array_like, shape (n_params,)
        The array of unbounded variables to transform.
    lower_bounds : array_like, shape (n_params,)
        The array of lower bounds for `x`.
    upper_bounds : array_like, shape (n_params,)
        The array of upper bounds for `x`.

    Returns
    -------
    x : array_like, shape (n_params,)
        The array of bounded variables.
    """
    return lower_bounds + (upper_bounds - lower_bounds) * sigmoid(y)


def logdens_model_to_logdens_hmc(
    logdens_model,
    y,
    lower_bounds,
    upper_bounds
):
    """
    Convert "model logdensity" to "hmc logdensity," convert P(x) to P(y). Note
    that we follow the convention of the STAN reference manual; `x` refers to
    the bounded variables, and `y` to their unbounded transformation (Ch. 10).

    Parameters
    ----------
    logdens_model : float
        Logdensity using bounded `x` parameters.
    y : array_like, shape (n_params,)
        The array of unbounded variables.
    lower_bounds : array_like, shape (n_params,)
        The array of lower bounds for `x`.
    upper_bounds : array_like, shape (n_params,)
        The array of upper bounds for `x`.

    Returns
    -------
    logdens_hmc : float
        The logdensity in unbounded `y` space.
    """
    sig = sigmoid(y)

    # determinant
    abs_det_of_jacobian = jnp.prod((upper_bounds - lower_bounds) * sig * (1 - sig))
    # log
    log_abs_det_of_jacobian = jnp.log(abs_det_of_jacobian)

    return logdens_model + log_abs_det_of_jacobian
