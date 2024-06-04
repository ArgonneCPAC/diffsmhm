import numpy as np
import Corrfunc
import psutil
import os

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1

from diffsmhm.diff_stats.mpi.types import SigmaMPIData


def sigma_serial_cpu(
    *, xh, yh, zh, wh, wh_jac, mask, xp, yp, zp, rpbins, zmax, boxsize
):
    """
    sigma_cpu_serial(...)
        Compute the 2D (x-y) surface density with derivatives for a periodic volume.

    Parameters
    ----------
    xh, yh, zh, wh : array-like, shape(n_halos,)
        The arrays of positions and weights for the halos.
    wh_jac : array-like, shape(n_grads, n_pts)
        The array of weight gradients for the halos.
    mask : array-like, shape (n_halos,)
        A boolean array with `True` for halos to be included and `False` for halos
        to be ignored. Generally used to mask out zero weights.
    xp, yp, zp : array-like, shape(n_pts,)
        The arrays of positions for the particles.
    rpbins : array-like, shape (n_rpbins+1,)
        Array of bin edges in the `rp` direction. Note that this array is one
        longer than the number of bins in the `rp` direction.
    zmax : float
        Maximum separation in the z direction. Particles with z distance less
        than zmax from a given halo are included in surface density calculations
        for that halo. Note this should be a whole number due to the unit
        binning of Corrfunc.
    boxsize : float
        The size of the total periodic volume.

    Returns
    -------
    sigma : array-like, shape(n_rpbins,)
        The surface density at each radial bin.
    sigma_grad : array-like, shape(n_grads, n_rpbins)
        The gradients of the surface density.
    """

    # ensure the smallest bin is zero
    assert np.allclose(rpbins[0], 0)

    # set up sizes
    n_grads = wh_jac.shape[0]

    n_rpbins = len(rpbins) - 1
    rads = np.array(np.pi * (np.square(rpbins[1:], dtype=np.float64) -
                    np.square(rpbins[:-1], dtype=np.float64)), dtype=np.float64)

    n_threads = int(os.environ.get("OMP_NUM_THREADS", psutil.cpu_count(logical=False)))

    n_parts = len(xp)

    # arrays to return
    sigma_grad = np.zeros((n_grads, n_rpbins), dtype=np.float64)

    # sigma
    res = Corrfunc.theory.DDrppi(
        autocorr=False,
        nthreads=n_threads,
        pimax=zmax,
        binfile=rpbins,
        X1=xh[mask], Y1=yh[mask], Z1=zh[mask], weights1=wh[mask],
        periodic=True, boxsize=boxsize,
        X2=xp, Y2=yp, Z2=zp, weights2=np.ones(n_parts, dtype=np.float64),
        weight_type="pair_product"
    )
    _dd = (
        res["weightavg"].reshape((n_rpbins, int(zmax))) *
        res["npairs"].reshape((n_rpbins, int(zmax)))
    ).astype(np.float64)
    sigma_exp = np.sum(_dd, axis=1, dtype=np.float64)

    # do radial normalization
    sigma_exp /= rads
    # normalize by weights total
    sigma_exp /= np.sum(wh, dtype=np.float64)

    # first term of sigma_grad
    sigma_grad_1st = np.zeros((n_grads, n_rpbins), dtype=np.float64)
    for g in range(n_grads):
        res_grad = Corrfunc.theory.DDrppi(
            autocorr=False,
            nthreads=n_threads,
            pimax=zmax,
            binfile=rpbins,
            X1=xh[mask], Y1=yh[mask], Z1=zh[mask], weights1=wh_jac[g, mask],
            periodic=True, boxsize=boxsize,
            X2=xp, Y2=yp, Z2=zp, weights2=np.ones(n_parts, dtype=np.float64),
            weight_type="pair_product"
        )
        _dd_grad = (
            res_grad["weightavg"].reshape((n_rpbins, int(zmax))) *
            res["npairs"].reshape((n_rpbins, int(zmax)))
        ).astype(np.float64)
        sigma_grad_1st[g, :] = np.sum(_dd_grad, axis=1, dtype=np.float64) / rads

    # second term of sigma grad
    grad_sum = np.sum(wh_jac, axis=1, dtype=np.float64).reshape(n_grads, 1)
    sigma_row = sigma_exp.reshape(1, n_rpbins)
    sigma_grad_2nd = np.matmul(grad_sum, sigma_row, dtype=np.float64)

    sigma_grad = (
                    sigma_grad_1st - sigma_grad_2nd
                 ) / np.sum(wh, dtype=np.float64)

    # return
    return sigma_exp, sigma_grad


def sigma_mpi_kernel_cpu(
    *, xh, yh, zh, wh, wh_jac, mask, xp, yp, zp, rpbins, zmax, boxsize
):
    """
    sigma_mpi_kernel_cpu(...)
        Per-process CPU kernel for MPI-parallel sigma (surface density) computation.

    Parameters
    ---------
    xh, yh, zh, wh : array-like, shape (n_halos,)
        The arrays of positions and weights for the halos.
    wh_jac : array-like, shape (n_grads, n_halos)
        The weight gradients for the halos.
    mask : array-like, shape (n_halos,)
        A boolean array with `True` for halos to be included and `False` for halos
        to be ignored. Generally used to mask out zero weights.
    xp, yp, zp, : array-like, shape (n_particles,)
        The arrays of positions for the particles.
    rpbins : array-like, shape (n_rpbins+1,)
        Array of bin edges in the `rp` direction. Note that this array is one
        longer than the number of bins in the `rp` (radial) direction.
    zmax : float
        Maximum separation in the z direction. Particles with z distance less
        than zmax from a given halo are included in surface density calculations
        for that halo. Note this should be a whole number due to the unit
        binning of Corrfunc.
    boxsize: float
        The size of the total periodic volume.

    Returns
    -------
    sigma : array-like, shape(n_rpbins,)
        The surface density at each radial bin.
    sigma_grad_1st : array-like, shape(n_grads, n_rpbins)
        The first term of the surface density gradients at each radial bin.
    """

    assert np.allclose(rpbins[0], 0)

    # set up sizes
    n_parts = len(xp)

    n_grads = wh_jac.shape[0]

    n_rpbins = len(rpbins) - 1

    n_threads = int(os.environ.get("OMP_NUM_THREADS", psutil.cpu_count(logical=False)))

    # Note: 2nd term of gradient needs to be calculated with the finalized sigma
    # and grads, so we only handle the calculation of the first term here

    res = Corrfunc.theory.DDrppi(
        autocorr=False,
        nthreads=n_threads,
        pimax=zmax,
        binfile=rpbins,
        X1=xh[mask], Y1=yh[mask], Z1=zh[mask], weights1=wh[mask],
        periodic=False,
        X2=xp, Y2=yp, Z2=zp, weights2=np.ones(n_parts, dtype=np.float64),
        weight_type="pair_product"
    )

    _dd = (
        res["weightavg"].reshape((n_rpbins, int(zmax))) *
        res["npairs"].reshape((n_rpbins, int(zmax)))
    ).astype(np.float64)
    sigma_exp = np.sum(_dd, axis=1)

    # do partial sum for 1st grad term; don't normalize by total of weights
    sigma_grad_1st = np.zeros((n_grads, n_rpbins), dtype=np.float64)
    for g in range(n_grads):
        res_grad = Corrfunc.theory.DDrppi(
            autocorr=False,
            nthreads=n_threads,
            pimax=zmax,
            binfile=rpbins,
            X1=xh[mask], Y1=yh[mask], Z1=zh[mask], weights1=wh_jac[g, mask],
            periodic=False,
            X2=xp, Y2=yp, Z2=zp, weights2=np.ones(n_parts, dtype=np.float64),
            weight_type="pair_product"
        )
        _dd_grad = (
            res_grad["weightavg"].reshape((n_rpbins, int(zmax))) *
            res["npairs"].reshape((n_rpbins, int(zmax)))
        ).astype(np.float64)
        rads = np.pi * (rpbins[1:]**2 - rpbins[:-1]**2)
        sigma_grad_1st[g, :] = np.sum(_dd_grad, axis=1) / rads

    # do radial normalization
    sigma_exp /= np.pi * (rpbins[1:]**2 - rpbins[:-1]**2)

    # return
    return SigmaMPIData(
            sigma=sigma_exp,
            sigma_grad_1st=sigma_grad_1st,
            w_tot=np.sum(wh[mask]),
            w_jac_tot=np.sum(wh_jac[:, mask], axis=1)
    )


def delta_sigma_from_sigma(sigma, sigma_grad, rpbins):
    """
    delta_sigma_from_sigma(...)
        Compute delta sigma from sigma.

    Parameters
    ----------
    sigma : array-like, shape(n_rpbins,)
        The radially binned values of halo surface density.
    sigma_grad : array_like, shape(n_params, n_rpbins)
        The gradients of sigma.
    rpbins : array-like, shape(n_rpbins+1,)
        Array of radial bin edges in `rp`. Note that this array is one longer
        than the number of bins in the `rp` direction.
    Returns
    -------
    delta_sigma : array-like, shape(n_rpbins,)
        The radially binned excess surface mass density.
    delta_sigma_grad : array-like, shape(n_params, n_rpbins)
        The gradients of delta_sigma
    """

    n_rpbins = len(sigma)
    n_params = sigma_grad.shape[0]
    delta_sigma = np.empty(n_rpbins, dtype=np.float64)
    delta_sigma_grad = np.empty((n_params, n_rpbins), dtype=np.float64)

    for i in range(n_rpbins):
        interior_vol = np.pi * np.square(rpbins[i])

        interior_sigma = np.sum(sigma[:i]) / interior_vol
        delta_sigma[i] = interior_sigma - sigma[i]

        interior_sigma_grad = np.sum(sigma_grad[:, :i], axis=1) / interior_vol
        delta_sigma_grad[:, i] = interior_sigma_grad - sigma_grad[:, i]

    return delta_sigma, delta_sigma_grad
