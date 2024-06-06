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

from diffsmhm.diff_stats.cpu.wprp_utils import compute_rr_rrgrad_eff
from diffsmhm.utils import time_step


def wprp_mpi_comp_and_reduce(
    *,
    x1, y1, z1, w1,
    w1_jac,
    inside_subvol,
    rpbins_squared,
    zmax,
    boxsize,
    kernel_func
):
    """The per-process CPU kernel for MPI-parallel wprp computations.

    Parameters
    ----------
    x1, y1, z1, w1 : array-like, shape (n_pts,)
        The arrays of positions and weights for the first set of points.
    w1_jac : array-lke, shape (n_grads, n_pts,)
        The array of weight gradients for the first set of points.
    inside_subvol : array-like, shape (n_pts,)
        A boolean array with `True` when the point is inside the subvolume
        and `False` otherwise.
    rpbins_squared : array-like, shape (n_rpbins+1,)
        Array of the squared bin edges in the `rp` direction. Note that
        this array is one longer than the number of bins in `rp` direction.
    zmax : float
        The maximum separation in the `pi` (or typically `z`) direction. Output
        bins in `z` direction are unit width, so make sure this is a whole number.
    boxsize : float
        The size of the total periodic volume of the simulation.
    kernel_func : function
        The per-task function to run on the rest of the inpout arguments. It is
        called with keywords that are the same as those for this function. It should
        return a `WprpMPIData` named tuple.

    Returns
    -------
    wprp : array-like, shape (n_rpbins,)
        The projected correlation function.
    wprp_grad : array-like, shape (n_grads, n_rpbins)
        The gradients of the projected correlation function.
    """
    with time_step("per-process kernel"):
        data = kernel_func(
            x1=x1, y1=y1, z1=z1,
            w1=w1, w1_jac=w1_jac,
            inside_subvol=inside_subvol,
            rpbins_squared=rpbins_squared,
            zmax=zmax,
            boxsize=boxsize,
        )

    with time_step("global reduction"):
        if COMM is not None and N_RANKS > 1:
            w_tot = np.zeros_like(data.w_tot)
            COMM.Reduce(data.w_tot, w_tot, MPI.SUM, root=0)
            w_tot = w_tot[0]

            w2_tot = np.zeros_like(data.w2_tot)
            COMM.Reduce(data.w2_tot, w2_tot, MPI.SUM, root=0)
            w2_tot = w2_tot[0]

            dw_tot = np.zeros_like(data.w_jac_tot)
            COMM.Reduce(data.w_jac_tot, dw_tot, MPI.SUM, root=0)

            wdw_tot = np.zeros_like(data.ww_jac_tot)
            COMM.Reduce(data.ww_jac_tot, wdw_tot, MPI.SUM, root=0)

            dd = np.zeros_like(data.dd)
            COMM.Reduce(data.dd, dd, MPI.SUM, root=0)

            dd_jac = np.zeros_like(data.dd_jac)
            COMM.Reduce(data.dd_jac, dd_jac, MPI.SUM, root=0)
        else:
            w_tot = data.w_tot[0]
            w2_tot = data.w2_tot[0]
            wdw_tot = data.ww_jac_tot
            dw_tot = data.w_jac_tot
            dd = data.dd
            dd_jac = data.dd_jac

    with time_step("final post-processing"):
        # now do norm by RR and compute proper grad
        if RANK == 0:
            # have the kernel handle any device/host transfers
            rpbins_sq = data.rpbins_squared

            n_eff = w_tot**2 / w2_tot

            # this is the volume of the shell
            n_pi = int(zmax)
            dpi = 1.0  # here to make the code clear, always true
            volfac = np.pi * (rpbins_sq[1:] - rpbins_sq[:-1])
            volratio = volfac[:, None] * np.ones(n_pi) * dpi / boxsize ** 3

            # finally get rr and drr
            rr, rr_grad = compute_rr_rrgrad_eff(w_tot, dw_tot, wdw_tot, n_eff, volratio)

            # now produce value and derivs
            xirppi = dd / rr - 1
            xirppi_grad = (
                dd_jac / rr[None, :, :] - dd[None, :, :] / rr[None, :, :] ** 2 * rr_grad
            )

            # integrate over pi
            wprp = 2.0 * dpi * np.sum(xirppi, axis=-1)
            wprp_grad = 2.0 * dpi * np.sum(xirppi_grad, axis=-1)

            return wprp, wprp_grad
        else:
            return None, None
