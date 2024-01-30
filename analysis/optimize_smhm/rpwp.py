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
                                

