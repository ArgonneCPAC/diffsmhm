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

from diffsmhm.analysis.tools.diff_sm import (
    compute_weight_and_jac,
    compute_weight_and_jac_quench
)

from diffsmhm.analysis.tools.rpwp import compute_rpwp


def mse_rpwp_quench(
    *,
    rpwp_q_goal,
    rpwp_nq_goal,
    logmpeak,
    loghost_mpeak,
    log_vmax_by_vmpeak,
    halo_x, halo_y, halo_z,
    upid,
    time_since_infall,
    idx_to_deposit,
    inside_subvol,
    rpbins,
    mass_bin_low,
    mass_bin_high,
    zmax,
    boxsize,
    theta,
    return_rpwp=False
):
    """Mean squared error for quenched rp wp(rp).

    Parameters
    ----------
    rpwp_q_goal : array_like, shape (n_rpbins,)
        Goal quenched rp wp(rp) measurement to compute error against.
    rpwp_nq_goal : array_like, shape (n_rpbins,)
        Goal unquenched rp wp(rp) measurement to compute error against.
    logmpeak : array_like, shape (n_gals,)
        The array of log10 halo masses.
    loghost_mpeak : array_like, shape (n_gals,)
        The array of log10 host halo masses.
    log_vmax_by_vmpeak : array_like, shape (n_gals,)
        The array of log10 maximum halo velocity divided by halo velocity at mpeak.
    halo_x, halo_y, halo_z : array_like, shape (n_gals,)
        The arrays of halo positions.
    upid : array_like, shape (n_gals,)
        The array of IDs for a (sub)halo. Should be -1 to indicate a (sub)halo
        has no parents.
    time_since_infall : array_like, shape (n_gals,)
        Time since infall for satellite halos.
    idx_to_deposit : array_like, shape (n_gals,)
        Index of each halo's UPID in the above arrays.
    inside_subvol : array_like, shape (n_gals,)
        A boolean array with `True` when the point is inside the subvolume
        and `False` otherwise.
    rpbins : array_like, shape (n_rpbins+1,)
        Array of the bin edges in the `rp` direction. Note that this array is
        one longer than the number of bins in `rp` direction.
    mass_bin_low : float
        Lower limit for the stellar mass bin.
    mass_bin_high : float
        Upper limit for the stellar mass bin.
    zmax : float
        The maximum separation in the `pi` (or typically `z`) direction. Output
        bins in `z` direction are unit width, so make sure this is a whole number.
    boxsize : float
        The size of the periodic volume.
    theta : array_like, shape (n_params,)
        Stellar to halo mass relation parameters.

    Returns
    -------
    err_sum : float
        Sum of mean squared error of the quenched and unquenched rp wp(rp) measurement.
    err_sum_grad : array_like, shape (n_params,)
        Gradient of the error sum with respect to model parameters.
    rpwp_q : array_like, shape (n_rpbins,)
        The rpwp computation for quenched galaxies given input data.
    rpwp_nq : array_like, shape (n_rpbins,)
        The rpwp computation for unquenched galaxies given input data.
    """

    # calculate weights
    w_q, dw_q, w_nq, dw_nq = compute_weight_and_jac_quench(
                                logmpeak,
                                loghost_mpeak,
                                log_vmax_by_vmpeak,
                                upid,
                                time_since_infall,
                                idx_to_deposit,
                                mass_bin_low, mass_bin_high,
                                theta
    )

    # do measurement
    rpwp_q, rpwp_q_jac = compute_rpwp(
                            x1=halo_x,
                            y1=halo_y,
                            z1=halo_z,
                            w1=w_q,
                            w1_jac=dw_q,
                            inside_subvol=inside_subvol,
                            rpbins=rpbins,
                            zmax=zmax,
                            boxsize=boxsize
    )

    rpwp_nq, rpwp_nq_jac = compute_rpwp(
                            x1=halo_x,
                            y1=halo_y,
                            z1=halo_z,
                            w1=w_q,
                            w1_jac=dw_q,
                            inside_subvol=inside_subvol,
                            rpbins=rpbins,
                            zmax=zmax,
                            boxsize=boxsize
    )

    # error
    err_sum, err_sum_grad = None, None
    if RANK == 0:
        err_q = np.sum((rpwp_q-rpwp_q_goal)*(rpwp_q-rpwp_q_goal)) / len(rpwp_q)
        err_q_jac = np.sum(2 * rpwp_q_jac * (rpwp_q-rpwp_q_goal), axis=1)
        err_q_jac /= len(rpwp_q)

        err_nq = np.sum((rpwp_nq-rpwp_nq_goal)*(rpwp_nq-rpwp_nq_goal)) / len(rpwp_nq)
        err_nq_jac = np.sum(2 * rpwp_nq_jac * (rpwp_nq-rpwp_nq_goal), axis=1)
        err_nq_jac /= len(rpwp_nq)

        err_sum = err_q + err_nq
        err_sum_grad = err_q_jac + err_nq_jac

    return err_sum, err_sum_grad, rpwp_q, rpwp_nq


# wrapper for use within adam function
def mse_rpwp_quench_adam_wrapper(static_params, opt_params):
    """Wrapper of above for use as adam error function.

    Parameters
    ----------
    static_params : array-like
        [rpwp_q_goal, rpwp_nq_goal,
         logmasspeak, loghost_mpeak, log_vmax_by_vmpeak,
         halo_x, halo_y, halo_z,
         upid, time_since_infall, idx_to_deposit, inside_subvol,
         rpbins,
         mass_bin_low, mass_bin_high,
         zmax,
         boxsize]
   opt_params : array-like
        Model parameters.

    Returns
    -------
    error : float
        Mean squared error of quenched and unquenched rp wp(rp).
    error_jac : array-like, shape(n_params,)
        Jacobian of error with respect to model parameters.

    """

    error, error_jac = mse_rpwp_quench(
            static_params[0], static_params[1],
            static_params[2], static_params[3], static_params[4],
            static_params[5], static_params[6], static_params[7],
            static_params[8], static_params[9], static_params[10], static_params[11],
            static_params[12],
            static_params[13], static_params[14],
            static_params[15],
            static_params[16],
            opt_params
    )

    return error, error_jac


def mse_rpwp(
    *,
    rpwp_goal,
    logmpeak,
    loghost_mpeak,
    log_vmax_by_vmpeak,
    halo_x, halo_y, halo_z,
    upid,
    idx_to_deposit,
    inside_subvol,
    rpbins,
    mass_bin_low, mass_bin_high,
    zmax,
    boxsize,
    theta
):
    """
    Mean squared error for rp wp(rp).

    Parameters
    ----------
    rpwp_goal : array_like, shape (n_rpbins,)
        Goal rp wp(rp) measurement to compute error against.
    logmpeak : array_like, shape (n_gals,)
        The array of log10 halo masses.
    loghost_mpeak : array_like, shape (n_gals,)
        The array of log10 host halo masses.
    log_vmax_by_vmpeak : array_like, shape (n_gals,)
        The array of log10 maximum halo velocity divided by halo velocity at mpeak.
    halo_x, halo_y, halo_z : array_like, shape (n_gals,)
        The arrays of halo positions.
    upid : array_like, shape (n_gals,)
        The array of IDs for a (sub)halo. Should be -1 to indicate a (sub)halo
        has no parents.
    idx_to_deposit : array_like, shape (n_gals,)
        Index of each halo's UPID in the above arrays.
    inside_subvol : array_like, shape (n_gals,)
        A boolean array with `True` when the point is inside the subvolume
        and `False` otherwise.
    rpbins : array_like, shape (n_rpbins+1,)
        Array of the bin edges in the `rp` direction. Note that this array is
        one longer than the number of bins in `rp` direction.
    mass_bin_low : float
        Lower limit for the stellar mass bin.
    mass_bin_high : float
        Upper limit for the stellar mass bin.
    zmax : float
        The maximum separation in the `pi` (or typically `z`) direction. Output
        bins in `z` direction are unit width, so make sure this is a whole number.
    boxsize : float
        The size of the periodic volume.
    theta : array_like, shape (n_params,)
        Stellar to halo mass relation parameters.

    Returns
    -------
    err: float
        Mean squared error of the rp wp(rp) measurement.
    err_sum_grad : array_like, shape (n_params,)
        Gradient of the error with respect to model parameters.
    rpwp : array_like, shape (n_rpbins,)
        The computed rpwp values for the given inputs.
    """

    w, dw = compute_weight_and_jac(
                logmpeak, loghost_mpeak,
                log_vmax_by_vmpeak,
                upid,
                idx_to_deposit,
                mass_bin_low, mass_bin_high,
                theta
    )

    # do measurement
    rpwp, rpwp_jac = compute_rpwp(
                        x1=halo_x,
                        y1=halo_y,
                        z1=halo_z,
                        w1=w,
                        w1_jac=dw,
                        inside_subvol=inside_subvol,
                        rpbins=rpbins,
                        zmax=zmax,
                        boxsize=boxsize
    )

    # error
    err, err_grad = None, None
    if RANK == 0:
        err = np.sum((rpwp-rpwp_goal)*(rpwp-rpwp_goal)) / len(rpwp)
        err_grad = np.sum(2 * rpwp_jac * (rpwp-rpwp_goal), axis=1) / len(rpwp)

    return err, err_grad, rpwp


def mse_rpwp_adam_wrapper(static_params, opt_params):
    """
    Wrapper for the above.

    Parameters
    ----------
    static_params : [
                        rpwp_goal,
                        logmpeak,
                        loghost_mpeak,
                        log_vmax_by_vmpeak,
                        halo_x, halo_y, halo_z,
                        upid,
                        idx_to_deposit,
                        inside_subvol,
                        rpbins,
                        mass_bin_low, mass_bin_high,
                        zmax,
                        boxsize
                    ]
    opt_params : array_like, shape (n_params,)
        Model parameters to optimize.

    Returns
    -------
    error : float
        Mean squared error from rpwp_goal.
    error_grad : array_like, shape (n_params,)
        Gradient of the mean squared error with respect to opt_params.
    """

    error, error_grad = mse_rpwp(
                        static_params[0],
                        static_params[1], static_params[2], static_params[3],
                        static_params[4], static_params[5], static_params[6],
                        static_params[7], static_params[8],
                        static_params[9],
                        static_params[10],
                        static_params[11], static_params[12],
                        static_params[13],
                        static_params[14],
                        opt_params
    )

    return error, error_grad
