import numpy as np
from numba import cuda

import cmath


def _copy_periodic_points_3D(x, y, z, boxsize, buffer_length):
    # copy particles within buffer_length of an edge in XY
    n_points = len(x)

    # setup buffer point arrays
    x_copied_points = []
    y_copied_points = []
    z_copied_points = []

    for p in range(n_points):
        # if not near edge continue
        if (
            not (x[p] < buffer_length or x[p] > boxsize-buffer_length) and
            not (y[p] < buffer_length or y[p] > boxsize-buffer_length) and
            not (z[p] < buffer_length or z[p] > boxsize-buffer_length)
        ):
            continue

        # 6 edges

        # x low edge
        if x[p] < buffer_length:
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p])
            z_copied_points.append(z[p])
        # x high edge
        elif x[p] > boxsize-buffer_length:
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p])
            z_copied_points.append(z[p])

        # y low edge
        if y[p] < buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p])
        # y high edge
        elif y[p] > boxsize-buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p])

        # z low edge
        if z[p] < buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p])
            z_copied_points.append(z[p] + boxsize)
        # z high edge
        elif z[p] > boxsize-buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p])
            z_copied_points.append(z[p] - boxsize)

        # 8 corners

        # "bottom" corners (z low)
        # x low, y low
        if (
                x[p] < buffer_length and
                y[p] < buffer_length and
                z[p] < buffer_length
        ):
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p] + boxsize)
        # x low, y high
        elif (
                x[p] < buffer_length and
                y[p] > boxsize-buffer_length and
                z[p] < buffer_length
        ):
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p] + boxsize)
        # x high, y low
        elif (
                x[p] > boxsize-buffer_length and
                y[p] < buffer_length and
                z[p] < buffer_length
        ):
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p] + boxsize)
        # x high, y high
        elif (
                x[p] > boxsize-buffer_length and
                y[p] > boxsize-buffer_length and
                z[p] < buffer_length
        ):
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p] + boxsize)

        # "top" corners (z high)
        # x low, y low
        elif (
                x[p] < buffer_length and
                y[p] < buffer_length and
                z[p] > boxsize-buffer_length
        ):
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p] - boxsize)
        # x low, y high
        elif (
                x[p] < buffer_length and
                y[p] > boxsize-buffer_length and
                z[p] > boxsize-buffer_length
        ):
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p] - boxsize)
        # x high, y low
        elif (
                x[p] > boxsize-buffer_length and
                y[p] < buffer_length and
                z[p] > boxsize-buffer_length
        ):
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p] - boxsize)
        # x high, y high
        elif (
            x[p] > boxsize-buffer_length and
            y[p] > boxsize-buffer_length and
            z[p] > boxsize-buffer_length
        ):
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p] - boxsize)

        # 12 egdges

        # x low, y low
        if x[p] < buffer_length and y[p] < buffer_length:
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p])
        # x high, y low
        if x[p] > boxsize-buffer_length and y[p] < buffer_length:
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p])
        # x low, y high
        if x[p] < buffer_length and y[p] > boxsize-buffer_length:
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p])
        # x high, y high
        if x[p] > boxsize-buffer_length and y[p] > boxsize-buffer_length:
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p])

        # y low, z low
        if y[p] < buffer_length and z[p] < buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p] + boxsize)
        # y high, z low
        if y[p] > boxsize-buffer_length and z[p] < buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p] + boxsize)
        # y low, z high
        if y[p] < buffer_length and z[p] > boxsize-buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p] - boxsize)
        # y high, z high
        if y[p] > boxsize-buffer_length and z[p] > boxsize-buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p] - boxsize)

        # x low, z low
        if x[p] < buffer_length and z[p] < buffer_length:
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p])
            z_copied_points.append(z[p] + boxsize)
        # x high, z low
        if x[p] > boxsize-buffer_length and z[p] < buffer_length:
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p])
            z_copied_points.append(z[p] + boxsize)
        # x low, z high
        if x[p] < buffer_length and z[p] > boxsize-buffer_length:
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p])
            z_copied_points.append(z[p] - boxsize)
        # x high, z high
        if x[p] > boxsize-buffer_length and z[p] > boxsize-buffer_length:
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p])
            z_copied_points.append(z[p] - boxsize)

    # combine and return
    x_periodic = np.append(x, x_copied_points)
    y_periodic = np.append(y, y_copied_points)
    z_periodic = np.append(z, z_copied_points)

    return x_periodic, y_periodic, z_periodic


@cuda.jit(fastmath=False)
def _count_particles(
    xh, yh, zh,
    wh, wh_jac, n_grads,
    xp, yp, zp,
    rpbins,
    zmax,
    result, result_grad
):

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n_halos = len(xh)
    n_particles = len(xp)
    n_bins = len(rpbins) - 1

    # for each halo
    for i in range(start, n_halos, stride):
        # for each particle
        for j in range(n_particles):
            # ensure Z distance is within range
            if abs(zh[i] - zp[j]) > zmax:
                continue

            # calculate XY distance
            pdist = cmath.sqrt((xh[i]-xp[j])*(xh[i]-xp[j]) +
                               (yh[i]-yp[j])*(yh[i]-yp[j]))

            for r in range(n_bins):
                if pdist >= rpbins[r] and pdist < rpbins[r+1]:
                    # add weight from halo
                    cuda.atomic.add(result, r, wh[i])
                    # and for gradients
                    for g in range(n_grads):
                        cuda.atomic.add(result_grad, (g, r), wh_jac[g, i])
                    break


def sigma_serial_cuda(
    *,
    xh, yh, zh, wh,
    wh_jac,
    xp, yp, zp,
    rpbins,
    zmax,
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
    zmax : float
        Maximum separation in the z direction. Particles with z distance less
        than zmax from a given halo are included in surface density calculations
        for that halo.
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
    periodic_buffer = max(rpmax, zmax)
    print("PB:", periodic_buffer, flush=True)

    # handle periodicity
    xp_p, yp_p, zp_p = _copy_periodic_points_3D(xp, yp, zp, boxsize, periodic_buffer)

    # set up device arrays
    sigma_device = cuda.to_device(np.zeros(n_rpbins, dtype=np.float64))
    sigma_grad_1st_device = cuda.to_device(np.zeros((n_grads, n_rpbins),
                                           dtype=np.float64))

    # do the actual counting on GPU
    _count_particles[blocks, threads](
                                        xh, yh, zh, wh, wh_jac, n_grads,
                                        xp_p, yp_p, zp_p,
                                        rpbins,
                                        zmax,
                                        sigma_device,
                                        sigma_grad_1st_device
                                     )

    sigma_exp = sigma_device.copy_to_host()
    sigma_grad_1st = sigma_grad_1st_device.copy_to_host().reshape((n_grads, n_rpbins))

    # normalize by surface area
    sigma_exp /= rads
    sigma_grad_1st /= rads

    # normalize sigma by weights sum
    weights_sum = np.sum(wh, dtype=np.float64)
    sigma_exp /= weights_sum

    # second term of gradient
    grad_sum = np.sum(wh_jac, axis=1, dtype=np.float64).reshape(n_grads, 1)
    sigma_row = sigma_exp.reshape(1, n_rpbins)
    sigma_grad_2nd = np.matmul(grad_sum, sigma_row, dtype=np.float64)

    # subtract gradient terms
    sigma_grad = sigma_grad_1st - sigma_grad_2nd

    # normalize gradient by weights sum
    sigma_grad /= weights_sum

    # return
    return sigma_exp, sigma_grad


def sigma_mpi_kernel_cuda(
    *,
    xh, yh, zh, wh,
    wh_jac,
    xp, yp, zp,
    rpbins,
    zmax,
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
    zmax : float
        Maximum separation in the z direction. Particles with z distance less
        than zmax from a given halo are included in surface density calculations
        for that halo.
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

    # set up device arrays
    sigma_device = cuda.to_device(np.zeros(n_rpbins, dtype=np.float64))
    sigma_grad_1st_device = cuda.to_device(np.zeros((n_grads, n_rpbins),
                                           dtype=np.float64))

    # do the actual counting on GPU
    _count_particles[blocks, threads](
                                        xh, yh, zh, wh, wh_jac, n_grads,
                                        xp, yp, zp,
                                        rpbins,
                                        zmax,
                                        sigma_device,
                                        sigma_grad_1st_device
                                     )

    sigma_exp = sigma_device.copy_to_host()
    sigma_grad_1st = sigma_grad_1st_device.copy_to_host().reshape((n_grads, n_rpbins))

    return sigma_exp, sigma_grad_1st
