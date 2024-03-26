from diffsmhm.diff_stats.mpi.sigma import sigma_mpi_comp_and_reduce
from diffsmhm.diff_stats.cuda.sigma import sigma_mpi_kernel_cuda
from diffsmhm.diff_stats.cpu.sigma import delta_sigma_from_sigma

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1


def compute_delta_sigma(
    *,
    xh, yh, zh,
    wh, wh_jac,
    xp, yp, zp,
    inside_subvol,
    rpbins,
    zmax,
    boxsize
):
    """Delta Sigma for a given set of halos. Essentially a wrapper for
    diff_stats.mpi.delta_sigma.

    Parameters
    ----------
    xh, yh, zh, wh : array-like, shape (n_halos,)
        Halo positions and weights.
    wh_jac : array-like, shape (n_halos,n_params)
        Halo weight jacobian.
    xp, yp, zp : array-like, shape (n_particles,)
        Particle positions.
    inside_subvol : array-like, shape (n_halos,)
        Boolean array with `True` when the point is inside the subvolume
        and `False` otherwise.
    rpbins : array-like, shape (n_rpbins+1,)
        Array of the bin edges in the `rp` direction. Note that this array is
        one longer than the number of bins in the `rp` direction.
    zmax : float
        The maximum separation in the `pi` (or typically `z`) direction. Particles
        with z distance less than zmax for a given halo are included in delta
        sigma calculations. Note this should be a whole number due to unit binning
        of Corrfunc.
    boxsize : float
        The size of the total periodic volume of the simulation.

    Returns
    -------
    delta_sigma : array-like, shape ( n_rpbins,)
        The 2D surface overdensity function.
    delta_sigma_grad : array-like, shape (n_rpbins, n_params)
        The gradients of the 2D surface overdensity function.
    """

    sigma, sigma_grad = sigma_mpi_comp_and_reduce(
                                xh=xh, yh=yh, zh=zh,
                                wh=wh,
                                wh_jac=wh_jac,
                                xp=xp, yp=yp, zp=zp,
                                inside_subvol=inside_subvol,
                                rpbins=rpbins,
                                zmax=zmax,
                                boxsize=boxsize,
                                kernel_func=sigma_mpi_kernel_cuda
    )

    delta_sigma, delta_sigma_grad = None, None
    if RANK == 0:
        delta_sigma, delta_sigma_grad = delta_sigma_from_sigma(
                                            sigma,
                                            sigma_grad,
                                            rpbins
        )

    return delta_sigma, delta_sigma_grad
