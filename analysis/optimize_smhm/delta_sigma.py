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
    """
    
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
