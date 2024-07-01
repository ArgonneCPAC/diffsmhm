from diffsmhm.diff_stats.mpi.wprp import wprp_mpi_comp_and_reduce
from diffsmhm.diff_stats.cuda.wprp import wprp_mpi_kernel_cuda

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1


# method for rpwp given weights
def compute_rpwp(
    *,
    x1, y1, z1,
    w1, w1_jac,
    inside_subvol,
    rpbins,
    zmax,
    boxsize
):
    """
    Compute rp wp(rp).

    Parameters
    ----------
    x1, y1, z1, w1 : array_like, shape (n_gals,)
        The arrays of positions and weights for the halos.
    w1_jac : array_like, shape (n_params, n_gals)
        The weight gradients for the halos.
    inside_subvol : array_like, shape (n_gals,)
        A boolean array with `True` when the point is inside the subvolume
        and `False` otherwise.
    rpbins : array_like, shape (n_rpbins+1,)
        Array of the bin edges in the `rp` direction. Note that this array is one
        longer than the number of bins in the `rp` direction.
    zmax : float
        The maximum separation in the `pi` (or typically `z`) direction. Output
        bins in the `z` direction are unit width, so make sure this is a whole number.
    boxsize : float
        The size of the total periodic volume of the simulation.

    Returns
    -------
    rpwp : array_like, shape (n_rpbins,)
        The rp wp(rp) computation.
    rpwp_grad : array_like, shape (n_params, n_rpbins)
        The gradients of rp wp(rp).
    """

    # calcualte wp(rp)
    wp, wp_grad = wprp_mpi_comp_and_reduce(
                                x1=x1, y1=y1, z1=z1,
                                w1=w1,
                                w1_jac=w1_jac,
                                inside_subvol=inside_subvol,
                                rpbins_squared=rpbins**2,
                                zmax=zmax,
                                boxsize=boxsize,
                                kernel_func=wprp_mpi_kernel_cuda
    )

    # compute rp wp(rp)
    # TODO: is this rp mult okay or do we want average radius?
    rpwp, rpwp_grad = None, None
    if RANK == 0:
        rpwp = wp * rpbins[:-1]
        rpwp_grad = wp_grad * rpbins[:-1]

    return rpwp, rpwp_grad
