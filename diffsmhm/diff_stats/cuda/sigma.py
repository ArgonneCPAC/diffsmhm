import numpy as np
from numba import cuda

import cmath


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
                    cuda.atomic.add(result, r*n_halos+i, 1)


def sigma_mpi_kernel_cuda(
    *,
    xh, yh, zh, wh,
    wh_jac,
    xp, yp, zp,
    rpbins,
    box_length,
    threads=32,
    blocks=512
):
    """
    sigma_mpi_kernel_cuda(...)
        Calculates the 2D (x-y) surface density at provided radius bins.

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
    box_length : float
        Length of the periodic volume, not used in the cuda kernel but included
        for consistency with CPU versions.

    Returns
    -------
    sigma : array-like, shape(n_rpbins,)
        The surface density at each bin specified by radii.
    sigma_grad_1st : array-like, shape(n_grads, n_rpbins)
        Partial sum of the first term of the gradients for sigma.
    """

    # set up sizes
    n_halos = len(xh)

    n_params = wh_jac.shape[0]

    n_rpbins = len(rpbins) - 1

    # set up arrays
    sigma = cuda.to_device(np.zeros((n_rpbins * n_halos), dtype=np.double))

    sigma_exp = np.empty((n_rpbins,), dtype=np.double)
    sigma_grad = np.empty((n_params, n_rpbins), dtype=np.double)

    # do the actual counting on GPU
    _count_particles[blocks, threads](
                                        xh, yh, zh, wh,
                                        xp, yp, zp,
                                        rpbins,
                                        sigma
                                     )

    sigma_host = sigma.copy_to_host().reshape((n_rpbins, n_halos))

    # apply weights
    np.matmul(sigma_host, wh, out=sigma_exp)

    # do partial grad sum
    for g in range(n_params):
        np.matmul(sigma_host, wh_jac[g, :].T, out=sigma_grad[g, :])

    return sigma_exp, sigma_grad
