import numpy as np

from diffsmhm.diff_stats.mpi.wprp import wprp_mpi_comp_and_reduce
from diffsmhm.diff_stats.cuda.wprp import wprp_mpi_kernel_cuda

from diff_sm import (
    compute_sm_and_jac,
    compute_sigma_and_jac
)

from diffsmhm.diff_stats.cuda.tw_kernels import (
    tw_kern_mstar_bin_weights_and_derivs_cuda
)

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1


# squared error
def se_rpwp(
    rpwp_target,
    mass_bin_low, mass_bin_high,
    idx_to_deposit,
    x1, y1, z1,
    inside_subvol,
    rpbins,
    zmax,
    boxsize,
    log_hm,
    log_host_hm,
    log_vmax_by_vmpeak,
    theta,
    threads=32,
    blocks=512
):
    """
    se_wprp(...)
        Squared error for binned rp * wp(rp) computations.

    Parameters
    ----------
    rpwp_target : array-like, shape(n_rpbins,)
        Array to take error with respect to.
    mass_bin_low, mass_bin_high : np.float64
        Bounds of the stellar mass bin. Note that Jax is picky about types
        when it comes to differentiation, so it's best to declare a numpy
        array that houses these values and pass arguments as arr[low], arr[high].
    idx_to_deposit : array-like, shape(n_halos,)
        Where each galaxy deposits merged mass.
    x1, y1, z1 : array-like, shape(n_halos,)
        Positions of each halo.
    inside_subvol : array-like, shape(n_halos,)
        A boolean array w2ith `True` when the point is inside the subvolume
        and `False` otherwise.
    rpbins : array-like, shape(n_rpbins+1,)
        Array of radial bin edges. Note that this array is one longer than the
        number of bins in the `rp` (radial) direction.
    zmax : float
        Maximum separation in the `pi` (or typically `z`) direction. Output
        bins in `z` direction are unit width, so make sure this is a whole number.
    boxsize : float
        The size of the total periodic volume of the simulation.
    log_hm : array-like, shape(n_halos,)
        Log halo mass.
    log_host_hm : array-like, shape(n_halos,)
        Log host halo mass.
    log_vmax_by_vmpeak : array-like, shape(n_halos,)
        Log of the maximum velocity divided by the velocity at Mpeak.
    theta : array-like, shape(n_params,)
        Model parameters for smhm, smhm sigma, and halo disruption.
    threads : int, optional
        The # of threads per block. Default is set to 32.
    blocks : int, optional
        The # of blocks on the GPU. Default is set to 512.

    Returns
    -------
    squared_error : float
        Squared error between rpwp_target and computed rpwp for given
        parameters theta.
    squared_error_grad : array_like, shape(n_params,)
        Gradient with respect to theta of the squared error.
    """

    # for array sizing
    n_halos = len(log_hm)
    n_params = len(theta)

    # obtain bin weights for given parameters
    sm, sm_jac = compute_sm_and_jac(
                                log_hm,
                                log_host_hm,
                                log_vmax_by_vmpeak,
                                idx_to_deposit,
                                theta
    )
    sigma, sigma_jac = compute_sigma_and_jac(log_hm, theta)

    wgt = np.zeros(n_halos, dtype=np.float64)
    dwgt = np.zeros((n_params, n_halos), dtype=np.float64)

    tw_kern_mstar_bin_weights_and_derivs_cuda[blocks, threads](
                                sm,
                                sm_jac,
                                sigma,
                                sigma_jac,
                                mass_bin_low, mass_bin_high,
                                wgt,
                                dwgt
    )

    # mask out gals with weight equal to zero
    wgt_mask = wgt > 0

    # then calculate wprp for those parameters
    wp, wp_grad = wprp_mpi_comp_and_reduce(
                                x1=x1[wgt_mask],
                                y1=y1[wgt_mask],
                                z1=z1[wgt_mask],
                                w1=wgt[wgt_mask],
                                w1_jac=dwgt[:, wgt_mask],
                                inside_subvol=inside_subvol[wgt_mask],
                                rpbins_squared=rpbins**2,
                                zmax=zmax,
                                boxsize=boxsize,
                                kernel_func=wprp_mpi_kernel_cuda
    )
    
    # convert wp to rpwp
    # TODO: Is this rp mult okay or do we want average radius?
    rpwp, rpwp_grad = None, None
    if RANK == 0:
       rpwp = wp * rpbins[:-1]
       rpwp_grad = wp_grad * rpbins[:-1]

    # compare to target
    if RANK == 0:
        err = np.sum((rpwp-rpwp_target)*(rpwp-rpwp_target))
        err_grad = np.sum(2 * rpwp_grad * (rpwp - rpwp_target), axis=1)
        return err, err_grad
    else:
        return None, None
