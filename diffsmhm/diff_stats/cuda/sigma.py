import numpy as np
from numba import cuda

import cmath

from diffsmhm.diff_stats.cpu.sigma import _copy_periodic_points_2D


@cuda.jit(fastmath=False)
def _count_particles(xh, yh, zh, wh, xp, yp, zp, rpbins, result):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n_halos = len(xh)
    n_particles = len(xp)
    n_bins = len(rpbins) - 1

    # for each halo
    for i in range(start, n_halos, stride):
        # for each particle
        for j in range(n_particles):
            # calculate distance
            pdist = cmath.sqrt((xh[i]-xp[j])*(xh[i]-xp[j]) +
                               (yh[i]-yp[j])*(yh[i]-yp[j]))

            for r in range(n_bins):
                if pdist >= rpbins[r] and pdist < rpbins[r+1]:
                    cuda.atomic.add(result, r, wh[i])


def sigma_serial_cuda(
    *,
    xh, yh, zh, wh,
    wh_jac,
    xp, yp, zp,
    rpbins,
    boxsize,
    threads=32,
    blocks=512
):
    """
    sigma_serial_cuda(...)
        Compute sigma with derivatives for a periodic volume.

    Parameters
    ----------
    xh, yh, zh, wh : array-like, shape(n_halos,)
        The arrays of positions and weights for the halos.
    wh_jac : array-like, shape(n_grads, n_halos,)
        The array of weight gradients for the first set of points.
    xp, yp, zp : array-like, shape (n_particles,)
        The arrays of positions for the particles.
    rpbins : array-like, shape (n_rpbins+1,)
        Array of radial bin edges, note that this array is one longer than the
        number of bins in the 'rp' (xy radial) direction.
    boxsize: float
        Length of the periodic volume, not used in the cuda kernel but included
        for consistency with CPU versions.

    Returns
    -------
    sigma : array-like, shape(n_rpbins,)
        The surface density at each radial bin.
    sigma_grad : array-like, shape(n_rpbins,)
        The surface density gradients at each radial bin.
    """

    # ensure smallest bin is not zero
    assert rpbins[0] > 0

    # set up sizes
    n_grads = wh_jac.shape[0]

    n_rpbins = len(rpbins) - 1
    rads = np.pi * (np.square(rpbins[1:], dtype=np.float64) -
                    np.square(rpbins[:-1], dtype=np.float64))

    rpmax = rpbins[-1]

    # handle periodicity
    xp_p, yp_p, zp_p = _copy_periodic_points_2D(xp, yp, zp, boxsize, rpmax)

    # set up arrays
    sigma = cuda.to_device(np.zeros(n_rpbins, dtype=np.float64))
    sigma_exp = np.empty(n_rpbins, dtype=np.float64)
    sigma_grad = np.zeros((n_grads, n_rpbins), dtype=np.float64)

    # do the actual counting on GPU
    _count_particles[blocks, threads](
                                        xh, yh, zh, wh,
                                        xp_p, yp_p, zp_p,
                                        rpbins,
                                        sigma
                                     )

    sigma_exp = sigma.copy_to_host()
    # normalize by surface area
    sigma_exp /= rads
    # normalize by weights total
    sigma_exp /= np.sum(wh, dtype=np.float64)

    # do gradient

    # first term
    sigma_grad_1st = np.empty((n_grads, n_rpbins), dtype=np.float64)

    for g in range(n_grads):
        sigma_grad_1st_g = cuda.to_device(np.zeros(n_rpbins, dtype=np.float64))
        wh_jac_g = cuda.to_device(np.copy(wh_jac[g, :]))

        _count_particles[blocks, threads](
                                            xh, yh, zh, wh_jac_g,
                                            xp_p, yp_p, zp_p,
                                            rpbins,
                                            sigma_grad_1st_g
                                         )

        sigma_grad_1st[g, :] = sigma_grad_1st_g.copy_to_host() / rads

    # second term
    grad_sum = np.sum(wh_jac, axis=1, dtype=np.float64).reshape(n_grads, 1)
    sigma_row = sigma_exp.reshape(1, n_rpbins)
    sigma_grad_2nd = np.matmul(grad_sum, sigma_row, dtype=np.float64)

    # subtract and normalize
    sigma_grad = (
                    sigma_grad_1st - sigma_grad_2nd
                 ) / np.sum(wh, dtype=np.float64)

    # return
    return sigma_exp, sigma_grad


def sigma_mpi_kernel_cuda(
    *,
    xh, yh, zh, wh,
    wh_jac,
    xp, yp, zp,
    rpbins,
    boxsize,
    threads=32,
    blocks=512
):
    """
    sigma_mpi_kernel_cuda(...)
        Per-process CUDA kernel for MPI-parallel sigma computations.

    Parameters
    ----------
    xh, yh, zh, wh : array-like, shape (n_halos,)
        The arrays of positions and weights for the halos.
    wh_jac : array-like, shape (n_grads, n_halos,)
        The array of weight gradients for the first set of points.
    xp, yp, zp : array-like, shape (n_particles,)
        The arrays of positions for the particles.
    rpbins : array-like, shape (n_rpbins+1,)
        Array of radial bin edges, Note that this array is one longer than the
        number of bins in the 'rp' (xy radial) direction.
    boxsize: float
        Length of the periodic volume, not used in the cuda kernel but included
        for consistency with CPU versions.

    Returns
    -------
    sigma : array-like, shape(n_rpbins,)
        The surface density at each radial bin.
    sigma_grad_1st : array-like, shape(n_grads, n_rpbins)
        Partial sum of the first term of the gradients for sigma.
    """

    # set up sizes
    n_grads = wh_jac.shape[0]

    n_rpbins = len(rpbins) - 1

    # set up arrays
    sigma = cuda.to_device(np.zeros(n_rpbins, dtype=np.float64))

    sigma_exp = np.empty(n_rpbins, dtype=np.float64)

    # do the actual counting on GPU
    _count_particles[blocks, threads](
                                        xh, yh, zh, wh,
                                        xp, yp, zp,
                                        rpbins,
                                        sigma
                                     )

    sigma_exp = sigma.copy_to_host()

    # do partial grad sum
    sigma_grad_1st = np.empty((n_grads, n_rpbins), dtype=np.float64)

    for g in range(n_grads):
        sigma_grad_1st_g = cuda.to_device(np.zeros(n_rpbins, dtype=np.float64))
        wh_jac_g = cuda.to_device(np.copy(wh_jac[g, :]))

        _count_particles[blocks, threads](
                                            xh, yh, zh, wh_jac_g,
                                            xp, yp, zp,
                                            rpbins,
                                            sigma_grad_1st_g
                                         )

        sigma_grad_1st[g, :] = sigma_grad_1st_g.copy_to_host()

    return sigma_exp, sigma_grad_1st
