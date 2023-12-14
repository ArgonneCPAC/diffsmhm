import numpy as np

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1


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
    n_halos = len(xh)
    n_parts = len(xp)

    n_rpbins = len(rpbins) - 1

    rpmax = rpbins[-1]

    # arrays to return
    sigma = np.zeros((n_rpbins, n_halos), dtype=np.double)

    # handle periodicity; need to copy particles within rpmax of edge
    for p in range(n_parts):
        # if not near edge, continue
        if (
                not (xp[p] < rpmax or xp[p] > box_length-rpmax) and
                not (yp[p] < rpmax or yp[p] > box_length-rpmax)
           ):
            continue

        # left edge
        if xp[p] < rpmax:
            xp = np.append(xp, xp[p]+box_length)
            yp = np.append(yp, yp[p])
        # right edge
        elif xp[p] > box_length-rpmax:
            xp = np.append(xp, xp[p]-box_length)
            yp = np.append(yp, yp[p])

        # upper edge
        if yp[p] < rpmax:
            xp = np.append(xp, xp[p])
            yp = np.append(yp, yp[p]+box_length)
        # lower edge
        elif yp[p] > box_length-rpmax:
            xp = np.append(xp, xp[p])
            yp = np.append(yp, yp[p] - box_length)

        # corners
        # top left
        if xp[p] < rpmax and yp[p] < rpmax:
            xp = np.append(xp, xp[p]+box_length)
            yp = np.append(yp, yp[p]+box_length)
        # bottom left
        elif xp[p] < rpmax and yp[p] > box_length-rpmax:
            xp = np.append(xp, xp[p]+box_length)
            yp = np.append(yp, yp[p]-box_length)
        # top right
        elif xp[p] > box_length-rpmax and yp[p] < rpmax:
            xp = np.append(xp, xp[p]-box_length)
            yp = np.append(yp, yp[p]+box_length)
        # bottom right
        elif xp[p] > box_length-rpmax and yp[p] > box_length-rpmax:
            xp = np.append(xp, xp[p]-box_length)
            yp = np.append(yp, yp[p]-box_length)

    # update number of particles
    n_parts = len(xp)

    # for each halo
    for i in range(n_halos):
        for j in range(n_parts):
            # calculate distance from halo center to each particle
            pdist = np.sqrt(np.power(xh[i] - xp[j], 2) +
                            np.power(yh[i] - yp[j], 2), dtype=np.double)

            for r in range(n_rpbins):
                if pdist >= rpbins[r] and pdist < rpbins[r+1]:
                    # note: averaged over area of annulus
                    sigma[r, i] += 1 / (np.pi * (rpbins[r+1]**2 - rpbins[r]**2))

    # apply weights to sigma and normalize
    sigma_exp = np.matmul(sigma, wh, dtype=np.double) / np.sum(wh)

    # apply weight gradients for sigma gradient
    sigma_grad_weights = np.matmul(wh_jac, sigma.T, dtype=np.double)

    grad_sum = np.sum(wh_jac, axis=1, dtype=np.double).reshape(wh_jac.shape[0], 1)
    sigma_row = sigma_exp.reshape(1, n_rpbins)

    sigma_grad = (
                    sigma_grad_weights - np.matmul(grad_sum, sigma_row, dtype=np.double)
                 )/np.sum(wh)

    # return
    return sigma_exp, sigma_grad


def sigma_mpi_kernel_cpu(
    *, xh, yh, zh, wh, wh_jac, xp, yp, zp, rpbins
):
    """
    sigma_mpi_kernel_cpu(...)
        Calculates the 2D (x-y) surface density at provided radius bins

    Parameters
    ---------
    xh, yh, zh, wh : array-like, shape (n_halos,)
        The arrays of positions and weights for the halos.
    wh_jac : array-like, shape (n_params, n_halos)
        The weight gradients for the halos.
    xp, yp, zp, : array-like, shape (n_particles,)
        The arrays of positions, weights, and weight gradients for the particles.
    rpbins : array-like, shape (n_rpbins+1,)
        Array of radial bin edges. Note that this array is one longer than
        the number of bins in the 'rp' (radial) direction.

    Returns
    -------
    sigma : array-like, shape(n_rpbins,)
        The surface density at each bin specified by rpbins.
    sigma_grad : array-like, shape(n_halos, n_rpbins)
        The surface density gradients at each bin specified by rpbins
    """

    # set up sizes
    n_halos = len(xh)
    n_parts = len(xp)

    n_params = wh_jac.shape[0]

    n_rpbins = len(rpbins) - 1

    # arrays to return
    # Note: 2nd term of gradient needs to be calculated with the finalized sigma
    # and grads, so we only handle the calculation of the first term here
    sigma = np.zeros((n_rpbins, n_halos), dtype=np.double)
    sigma_grad_1st = np.zeros((n_params, n_rpbins), dtype=np.double)

    # for each halo
    for i in range(n_halos):
        # for each particle
        for j in range(n_parts):
            # calculate distance from halo center to each particle
            pdist = np.sqrt(np.power(xh[i] - xp[j], 2)
                            + np.power(yh[i] - yp[j], 2))

            for r in range(n_rpbins):
                if pdist >= rpbins[r] and pdist < rpbins[r+1]:
                    sigma[r, i] += 1

    # do partial sum; don't normalize by total of weights
    # apply galaxy weights
    sigma_exp = np.matmul(sigma, wh)

    # first term of gradient
    for g in range(n_params):
        sigma_grad_1st[g, :] = np.matmul(sigma, wh_jac[g, :].T)

    # return
    return sigma_exp, sigma_grad_1st
