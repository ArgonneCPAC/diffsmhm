import numpy as np
from jax.nn import sigmoid as sigmoid


def get_1d_arrays(*args):
    """Return a list of ndarrays of the same length.

    Each arg must be either an ndarray of shape (npts, ), or a scalar.

    """
    results = [np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    return [np.zeros(npts).astype(arr.dtype) + arr for arr in results]


def jax_sigmoid_exp(x, x0, k, ylo, yhi):
    """Sigmoid function implemented w/ `jax.numpy.exp`.

    Parameters
    ----------
    x : float or array-like
        Points at which to evaluate the function.
    x0 : float or array-like
        Location of transition.
    k : float or array-like
        Inverse of the width of the transition.
    ylo : float or array-like
        The value as x goes to -infty.
    yhi : float or array-like
        The value as x goes to +infty.

    Returns
    -------
    sigmoid : scalar or array-like, same shape as input
    """
    return ylo + (yhi - ylo) * sigmoid(k * (x - x0))


jax_sigmoid = jax_sigmoid_exp
