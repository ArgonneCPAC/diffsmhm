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


def _copy_periodic_points_2D(x, y, z, box_length, buffer_length):
    # copy particles within buffer_length of an edge in XY
    n_points = len(x)

    # setup position copies
    x_periodic = np.copy(x)
    y_periodic = np.copy(y)
    z_periodic = np.copy(z)

    for p in range(n_points):
        # if not near edge continue
        if (
            not (x[p] < buffer_length or x[p] > box_length-buffer_length) and
            not (y[p] < buffer_length or y[p] > box_length-buffer_length)
           ):
            continue

        # left edge
        if x[p] < buffer_length:
            x_periodic = np.append(x_periodic, x[p]+box_length)
            y_periodic = np.append(y_periodic, y[p])
            z_periodic = np.append(z_periodic, z[p])
        # right edge
        elif x[p] > box_length-buffer_length:
            x_periodic = np.append(x_periodic, x[p]-box_length)
            y_periodic = np.append(y_periodic, y[p])
            z_periodic = np.append(z_periodic, z[p])

        # upper edge
        if y[p] < buffer_length:
            x_periodic = np.append(x_periodic, x[p])
            y_periodic = np.append(y_periodic, y[p]+box_length)
            z_periodic = np.append(z_periodic, z[p])
        # lower edge
        elif y[p] > box_length-buffer_length:
            x_periodic = np.append(x_periodic, x[p])
            y_periodic = np.append(y_periodic, y[p] - box_length)
            z_periodic = np.append(z_periodic, z[p])

        # corners
        # top left
        if x[p] < buffer_length and y[p] < buffer_length:
            x_periodic = np.append(x_periodic, x[p]+box_length)
            y_periodic = np.append(y_periodic, y[p]+box_length)
            z_periodic = np.append(z_periodic, z[p])
        # bottom left
        elif x[p] < buffer_length and y[p] > box_length-buffer_length:
            x_periodic = np.append(x_periodic, x[p]+box_length)
            y_periodic = np.append(y_periodic, y[p]-box_length)
            z_periodic = np.append(z_periodic, z[p])
        # top right
        elif x[p] > box_length-buffer_length and y[p] < buffer_length:
            x_periodic = np.append(x_periodic, x[p]-box_length)
            y_periodic = np.append(y_periodic, y[p]+box_length)
            z_periodic = np.append(z_periodic, z[p])
        # bottom right
        elif x[p] > box_length-buffer_length and y[p] > box_length-buffer_length:
            x_periodic = np.append(x_periodic, x[p]-box_length)
            y_periodic = np.append(y_periodic, y[p]-box_length)
            z_periodic = np.append(z_periodic, z[p])

    return x_periodic, y_periodic, z_periodic


def sigma_cpu_serial(
    *, xh, yh, zh, wh, wh_jac, xp, yp, zp, rpbins, box_length
):
    """
    sigma_cpu_serial(...)
        Calculates the 2D (x-y) surface density at provided radius bins

    Parameters
    ----------
    xh, yh, zh, wh : array-like, shape(n_halos,)
        The arrays of positions and weights for the halos.
    wh_jac : array-like, shape(n_grads, n_pts)
        The array of weight gradients for the halos.
    xp, yp, zp : array-like, shape(n_pts,)
        The arrays of positions, weights, and weight gradients for the particles.
    rpbins : array-like, shape (n_rpbins+1,)
        Array of radial bin edges. Note that this array is one longer than
        the number of bins in the 'rp' (radial) direction.
    box_length : float
        The size of the periodic volume.

    Returns
    -------
    sigma : array-like, shape(n_rpbins,)
        The surface density at each bin specified by rpbins.
    sigma_grad : array-like, shape(n_grads, n_rpbins)
        The surface density gradients at each bin specified by rpbins.
    """

    # ensure the smallest bin is not zero
    assert rpbins[0] > 0

    # set up sizes
    n_grads = wh_jac.shape[0]

    n_rpbins = len(rpbins) - 1
    rads = np.array(np.pi * (np.square(rpbins[1:], dtype=np.float64) -
                    np.square(rpbins[:-1], dtype=np.float64)), dtype=np.float64)

    rpmax = rpbins[-1]
    pimax = np.ceil(box_length).astype(int)

    n_threads = int(os.environ.get("OMP_NUM_THREADS", psutil.cpu_count(logical=False)))

    # handle periodicity
    xp_p, yp_p, zp_p = _copy_periodic_points_2D(xp, yp, zp, box_length, rpmax)
    n_parts = len(xp_p)

    # arrays to return
    sigma_grad = np.zeros((n_grads, n_rpbins), dtype=np.float64)

    # sigma
    res = Corrfunc.theory.DDrppi(
        autocorr=False,
        nthreads=n_threads,
        pimax=pimax,
        binfile=rpbins,
        X1=xh, Y1=yh, Z1=zh, weights1=wh,
        periodic=False,
        X2=xp_p, Y2=yp_p, Z2=zp_p, weights2=np.ones(n_parts, dtype=np.float64),
        weight_type="pair_product"
    )
    _dd = (
        res["weightavg"].reshape((n_rpbins, pimax)) *
        res["npairs"].reshape((n_rpbins, pimax))
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
            pimax=pimax,
            binfile=rpbins,
            X1=xh, Y1=yh, Z1=zh, weights1=wh_jac[g, :],
            periodic=False,
            X2=xp_p, Y2=yp_p, Z2=zp_p, weights2=np.ones(n_parts, dtype=np.float64),
            weight_type="pair_product"
        )
        _dd_grad = (
            res_grad["weightavg"].reshape((n_rpbins, pimax)) *
            res["npairs"].reshape((n_rpbins, pimax))
        ).astype(np.float64)
        sigma_grad_1st[g, :] = np.sum(_dd_grad, axis=1, dtype=np.float64) / rads

    # second term of sigma grad
    grad_sum = np.sum(wh_jac, axis=1, dtype=np.float64).reshape(n_grads, 1)
    sigma_row = sigma_exp.reshape(1, n_rpbins)
    sigma_grad_2nd = np.matmul(grad_sum, sigma_row, dtype=np.float64)

    sigma_grad = (
                    sigma_grad_1st - sigma_grad_2nd
                 )/np.sum(wh, dtype=np.float64)

    # return
    return sigma_exp, sigma_grad


def sigma_mpi_kernel_cpu(
    *, xh, yh, zh, wh, wh_jac, xp, yp, zp, rpbins, box_length
):
    """
    sigma_mpi_kernel_cpu(...)
        Calculates the 2D (x-y) surface density at provided radius bins

    Parameters
    ---------
    xh, yh, zh, wh : array-like, shape (n_halos,)
        The arrays of positions and weights for the halos.
    wh_jac : array-like, shape (n_grads, n_halos)
        The weight gradients for the halos.
    xp, yp, zp, : array-like, shape (n_particles,)
        The arrays of positions, weights, and weight gradients for the particles.
    rpbins : array-like, shape (n_rpbins+1,)
        Array of radial bin edges. Note that this array is one longer than
        the number of bins in the 'rp' (radial) direction.
    box_length : float
        The size of the total periodic volume.

    Returns
    -------
    sigma : array-like, shape(n_rpbins,)
        The surface density at each bin specified by rpbins.
    sigma_grad_1st : array-like, shape(n_grads, n_rpbins)
        The first term of surface density gradients at each bin specified by
        rpbins.
    """

    # set up sizes
    n_parts = len(xp)

    n_grads = wh_jac.shape[0]

    n_rpbins = len(rpbins) - 1

    pimax = np.ceil(box_length).astype(int)

    n_threads = int(os.environ.get("OMP_NUM_THREADS", psutil.cpu_count(logical=False)))

    # Note: 2nd term of gradient needs to be calculated with the finalized sigma
    # and grads, so we only handle the calculation of the first term here

    res = Corrfunc.theory.DDrppi(
        autocorr=False,
        nthreads=n_threads,
        pimax=pimax,
        binfile=rpbins,
        X1=xh, Y1=yh, Z1=zh, weights1=wh,
        periodic=False,
        X2=xp, Y2=yp, Z2=zp, weights2=np.ones(n_parts, dtype=np.float64),
        weight_type="pair_product"
    )

    _dd = (
        res["weightavg"].reshape((n_rpbins, pimax)) *
        res["npairs"].reshape((n_rpbins, pimax))
    ).astype(np.float64)
    sigma_exp = np.sum(_dd, axis=1)

    # do partial sum for 1st grad term; don't normalize by total of weights
    sigma_grad_1st = np.zeros((n_grads, n_rpbins), dtype=np.float64)
    for g in range(n_grads):
        res_grad = Corrfunc.theory.DDrppi(
            autocorr=False,
            nthreads=n_threads,
            pimax=pimax,
            binfile=rpbins,
            X1=xh, Y1=yh, Z1=zh, weights1=wh_jac[g, :],
            periodic=False,
            X2=xp, Y2=yp, Z2=zp, weights2=np.ones(n_parts, dtype=np.float64),
            weight_type="pair_product"
        )
        _dd_grad = (
            res_grad["weightavg"].reshape((n_rpbins, pimax)) *
            res["npairs"].reshape((n_rpbins, pimax))
        ).astype(np.float64)
        rads = np.pi * (rpbins[1:]**2 - rpbins[:-1]**2)
        sigma_grad_1st[g, :] = np.sum(_dd_grad, axis=1) / rads

    # return
    return sigma_exp, sigma_grad_1st


def delta_sigma_from_sigma(rpbins, sigma):
    """
    delta_sigma_from_sigma(...)
        Calculates delta sigma based on a provided sigma array.

    Parameters
    ----------
    rpbins : array-like, shape(n_rpbins+1,)
        Array of radial bin edges, note that this array is one longer than
        the number of bins in the rp direction.
    sigma : array-like, shape(n_rpbins,)
        The radially binned values of halo surface density.

    Returns
    -------
    delta_sigma : array-like, shape(n_rpbins,)
        The radially binned surface mass density.
    """

    n_rpbins = len(sigma)
    delta_sigma = np.empty(n_rpbins, dtype=np.float64)

    for i in range(n_rpbins):
        # TODO: add cylinder length to normalization? (currently is a circle)
        interior_sigma = np.sum(sigma[:i])/(np.pi * np.square(rpbins[i]))
        delta_sigma[i] = interior_sigma - sigma[i]

    return delta_sigma
