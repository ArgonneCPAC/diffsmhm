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


def sigma_mpi_comp_and_reduce(
    *,
    xh, yh, zh, wh, wh_jac,
    xp, yp, zp,
    inside_subvol,
    rpbins,
    kernel_func
):
    """
    The per-process cpu kernel for MPI-parallel surface density calculations.

    Parameters
    ---------
    xh, yh, zh, wh : array-like, shape (n_halos,)
        The arrays of positions and weights for the halos.
    wh_jac : array-like, shape (n_grads, n_pts,)
        The array of weight gradients for the halos.
    xp, yp, zp : array-like, shape (n_pts,)
        The arrays of positions, weights, and weight gradients for the particles.
    inside_subvol : array-like, shape (n_halos,)
        A boolean array with 'True' when the halo is inside the subvolume
        and 'False' otherwise.
    rpbins : array-like, shape (n_rpbins+1,)
        Array of radial bin edges. Note that this array is one longer than the
        number of bins in the 'rp' (radial) direction.
    kernel_func : function
        The per-task function to run on the rest of the input arguments. It is
        called with keywords that are the same as for this function. It should
        return a partial sum of sigma of size (n_rpbins,) and a partial sum of
        first term of the gradient of size (n_grads, n_rpbins,).

    Returns
    -------
    sigma : array-like, shape (n_halos, n_rpbins)
        The 2D surface density of each halo.
    sigma_grad : array-like, shape (n_halos, n_rpbins)
        The gradients of the 2D surface density.
    """

    # assert smallest bin is not zero
    assert rpbins[0] > 0

    # sizing parameters
    n_grads = wh_jac.shape[0]
    n_bins = len(rpbins) - 1

    # cut out halos not inside the subvolume
    xh_sv = xh[inside_subvol]
    yh_sv = yh[inside_subvol]
    zh_sv = zh[inside_subvol]
    wh_sv = wh[inside_subvol]
    wh_jac_sv = wh_jac[:, inside_subvol]

    # do local calculation
    sigma, sigma_grad_1st = kernel_func(
        xh=xh_sv, yh=yh_sv, zh=zh_sv, wh=wh_sv,
        wh_jac=wh_jac_sv,
        xp=xp, yp=yp, zp=zp,
        rpbins=rpbins
    )

    # reduction

    # sum reduce sigma
    sigma_red = np.zeros_like(sigma)
    COMM.Reduce(sigma, sigma_red, op=MPI.SUM, root=0)

    # get weight gradients total
    sum_wh_jac_rank = np.sum(wh_jac_sv, axis=1)
    sum_wh_jac_all = np.zeros(n_grads, dtype=np.double)
    COMM.Reduce(sum_wh_jac_rank, sum_wh_jac_all, op=MPI.SUM, root=0)

    # get sum of partial sums of first grad term from kernels
    sum_sigma_grad_1st = np.zeros((n_grads, n_bins), dtype=np.double)
    for p in range(n_grads):
        COMM.Reduce(sigma_grad_1st[p, :], sum_sigma_grad_1st[p, :], op=MPI.SUM, root=0)

    # do radial normalization
    sigma_red /= np.pi * (rpbins[1:]**2 - rpbins[:-1]**2)

    # get weights total
    sum_weights_rank = np.array([sum(wh[inside_subvol])], dtype=np.double)
    sum_weights_all = np.ones(1, dtype=np.double)

    COMM.Reduce(sum_weights_rank, sum_weights_all, op=MPI.SUM, root=0)

    # divide
    sigma_red /= sum_weights_all[0]

    # now finish the gradient
    sigma_grad_2nd = np.zeros((n_grads, n_bins), dtype=np.double)
    for p in range(n_grads):
        sigma_grad_2nd[p, :] = sum_wh_jac_all[p] * sigma_red
    sigma_grad_full = (sum_sigma_grad_1st - sigma_grad_2nd) / sum_weights_all

    if RANK == 0:
        return sigma_red, sigma_grad_full
    else:
        return None, None
