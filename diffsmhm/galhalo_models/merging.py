"""Functions transfer stellar mass between galaxies during probabilistic merging."""
import numpy as np
from jax import jit as jax_jit
from jax import numpy as jax_np
from .crossmatch import crossmatch_integers


__all__ = ("deposit_stellar_mass",)


def deposit_stellar_mass(logsm, indx_to_deposit, frac_to_deposit):
    """Exchange stellar mass between merging galaxies.

    Parameters
    ----------
    logsm : float or ndarray
        Base-10 log stellar mass in each galaxy

    indx_to_deposit : float or ndarray
        index of the host halo
        Can be calculated with _calculate_indx_to_deposit

    frac_to_deposit : float or ndarray
        Fraction of mass to deposit

    Returns
    -------
    total_mstar : ndarray
        Total mass in each galaxy after mergers

    """
    logsm, indx_to_deposit, frac_to_deposit = _get_1d_arrays(
        logsm, indx_to_deposit, frac_to_deposit
    )

    total_mstar = _jax_deposit_mstar_jax(logsm, indx_to_deposit, frac_to_deposit)
    return jax_np.asarray(total_mstar)


#@jax_jit
def _jax_deposit_mstar_jax(logsm, indx_to_deposit, frac_to_deposit):
    ngals = logsm.shape[0]

    print("fdep", jax_np.max(jax_np.abs(frac_to_deposit)))
    fkeep = 1 - frac_to_deposit
    print("fkeep", jax_np.mean(jax_np.abs(fkeep)))

    mstar = jax_np.power(10, logsm)
    mstar_to_deposit = frac_to_deposit * mstar
    print("mdep:", jax_np.mean(jax_np.abs(mstar_to_deposit)))
    mstar_to_keep = (1 - frac_to_deposit) * mstar
    print("mkeep:", jax_np.mean(mstar_to_keep), flush=True)
    indx_to_keep = jax_np.arange(ngals).astype("i8")

    total_mstar = jax_np.zeros_like(mstar)
    total_mstar = total_mstar.at[indx_to_deposit].add(mstar_to_deposit)
    total_mstar = total_mstar.at[indx_to_keep].add(mstar_to_keep)
    print("tms:", np.max(total_mstar), flush=True)

    return jax_np.absolute(total_mstar)


def _calculate_indx_to_deposit(upids, halo_ids):
    orig_indices = np.arange(len(halo_ids)).astype("i8")
    indx_to_deposit = np.zeros_like(orig_indices)

    host_ids = np.where(upids == -1, halo_ids, upids)
    idxA, idxB = crossmatch_integers(host_ids, halo_ids)
    indx_to_deposit[idxA] = orig_indices[idxB]
    return indx_to_deposit


def _get_1d_arrays(*args):
    """Return a list of ndarrays of the same length.

    Each arg must be either an ndarray of shape (npts, ), or a scalar.

    """
    results = [jax_np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    return [jax_np.zeros(npts).astype(arr.dtype) + arr for arr in results]
