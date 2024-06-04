import numpy as np
import cupy as cp

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1

from diffsmhm.utils import time_step


def sigma_mpi_comp_and_reduce(
    *,
    xh, yh, zh, wh, wh_jac,
    mask,
    xp, yp, zp,
    rpbins,
    zmax,
    boxsize,
    kernel_func
):
    """
    The per-process cpu kernel for MPI-parallel sigma computations

    Parameters
    ---------
    xh, yh, zh, wh : array-like, shape (n_halos,)
        The arrays of positions and weights for the halos.
    wh_jac : array-like, shape (n_grads, n_pts,)
        The array of weight gradients for the halos.
    mask : array-like, shape (n_halos,)
        A boolean array with `True` for halos to be included and `False` for halos
        to be ignored. Generally used to mask out zero weights and halos not in
        a given subvolume. Passed as a parameter to avoid copying masked data
        with each kernel call.
    xp, yp, zp : array-like, shape (n_pts,)
        The arrays of positions for the particles.
    rpbins : array-like, shape (n_rpbins+1,)
        Array of radial bin edges. Note that this array is one longer than the
        number of bins in the 'rp' (radial) direction.
    zmax : float
        Maximum separation in the z direction. Particle with z distance less
        than zmax from a given halo are included in surface density calculations
        for that halo. Note this should be a whole number due to the unit
        binning of Corrfunc.
    boxsize: float
        Total size of the periodic volume.
    kernel_func : function
        The per-task function to run on the rest of the input arguments. It is
        called with keywords that are the same as for this function. It should
        return a partial sum of sigma of size (n_rpbins,) and a partial sum of
        first term of the gradient of size (n_grads, n_rpbins,).

    Returns
    -------
    sigma : array-like, shape (n_halos, n_rpbins)
        The 2D surface density function.
    sigma_grad : array-like, shape (n_halos, n_rpbins)
        The gradients of the 2D surface density function.
    """

    # do local calculation
    with time_step("per-process kernel"):
        data = kernel_func(
            xh=xh, yh=yh, zh=zh, wh=wh,
            wh_jac=wh_jac,
            mask=mask,
            xp=xp, yp=yp, zp=zp,
            rpbins=rpbins,
            zmax=zmax,
            boxsize=boxsize
        )

    # reduction
    with time_step("global reduction"):
        if COMM is not None and N_RANKS > 1:
            # sum reduce sigma
            sigma_red = np.zeros_like(data.sigma)
            COMM.Reduce(data.sigma, sigma_red, op=MPI.SUM, root=0)

            # get weights total
            sum_weights_all = np.zeros(1, dtype=np.float64)
            COMM.Reduce(data.w_tot, sum_weights_all, op=MPI.SUM, root=0)

            # get weight gradients total
            n_grads = data.sigma_grad_1st.shape[0]
            sum_wh_jac_all = np.zeros(n_grads, dtype=np.float64)
            COMM.Reduce(data.w_jac_tot, sum_wh_jac_all, op=MPI.SUM, root=0)

            # get sum of partial sums of first grad term from kernels
            n_bins = len(data.sigma)
            sum_sigma_grad_1st = np.zeros((n_grads, n_bins), dtype=np.float64)
            for p in range(n_grads):
                COMM.Reduce(data.sigma_grad_1st[p, :],
                            sum_sigma_grad_1st[p, :], op=MPI.SUM, root=0)

        else:
            sigma_red = data.sigma
            sum_weights_all = data.w_tot
            sum_wh_jac_all = data.w_jac_tot
            sum_sigma_grad_1st = data.sigma_grad_1st

        with time_step("final post-processing"):
            if RANK == 0:
                # divide
                sigma_red /= sum_weights_all[0]

                # now finish the gradient
                sigma_grad_2nd = np.zeros((n_grads, n_bins), dtype=np.float64)
                for p in range(n_grads):
                    sigma_grad_2nd[p, :] = sum_wh_jac_all[p] * sigma_red
                sigma_grad_full = (
                                    sum_sigma_grad_1st - sigma_grad_2nd
                                  ) / sum_weights_all

                return sigma_red, sigma_grad_full

            else:
                return None, None
