import numpy as np

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
except ImportError:
    RANK = -1

from diff_sm import compute_weight_and_jac, compute_weight_and_jac_quench

from smf import smf_mpi_comp_and_reduce
from rpwp import compute_rpwp
from delta_sigma import compute_delta_sigma 

def mse_smhm_no_quench(
    smf_goal,
    rpwp_goal,
    delta_sigma_goal,
    log_halomass,
    log_hostmass,
    log_vmax_by_vmpeak,
    halo_x, halo_y, halo_z,
    upid,
    inside_subvol,
    time_since_infall,
    idx_to_deposit,
    part_x, part_y, part_z,
    rpbins,
    smf_bin_edges,
    mass_bin_low, mass_bin_high,
    zmax,
    boxsize,
    theta
):

    # compute weights
    w, dw = compute_weight_and_jac(
                        log_halomass,
                        log_hostmass,
                        log_vmax_by_vmpeak,
                        upid,
                        idx_to_deposit,
                        mass_bin_low, mass_bin_high,
                        theta
    )

    wgt_mask = w > 0

    # do measurements
    smf, smf_jac = smf_mpi_comp_and_reduce(
                        log_halomass,
                        log_hostmass,
                        log_vmax_by_vmpeak,
                        upid,
                        idx_to_deposit,
                        inside_subvol,
                        smf_bin_edges,
                        theta
    )

    rpwp, rpwp_jac = compute_rpwp(
                        x1=halo_x[wgt_mask],
                        y1=halo_y[wgt_mask],
                        z1=halo_z[wgt_mask],
                        w1=w[wgt_mask],
                        w1_jac=dw[:, wgt_mask],
                        inside_subvol=inside_subvol[wgt_mask],
                        rpbins=rpbins,
                        zmax=zmax,
                        boxsize=boxsize
    )

    delta_sigma, delta_sigma_jac = compute_delta_sigma(
                        xh=halo_x[wgt_mask],
                        yh=halo_y[wgt_mask],
                        zh=halo_z[wgt_mask],
                        wh=w[wgt_mask],
                        wh_jac=dw[:, wgt_mask],
                        xp=part_x,
                        yp=part_y,
                        zp=part_z,
                        inside_subvol=inside_subvol[wgt_mask],
                        rpbins=rpbins,
                        zmax=zmax,
                        boxsize=boxsize
    )

    # compute error
    err_sum, err_jac_sum = None, None
    if RANK == 0:
        err_smf = np.sum((smf-smf_goal)*(smf-smf_goal)) / len(smf)
        err_smf_jac = np.sum(2 * smf_jac * (smf-smf_goal), axis=1)

        err_rpwp = np.sum((rpwp-rpwp_goal)*(rpwp-rpwp_goal)) / len(rpwp)
        err_rpwp_jac = np.sum(2 * rpwp_jac * (rpwp-rpwp_goal), axis=1) 

        err_ds = np.sum((delta_sigma-delta_sigma_goal)*(delta_sigma-delta_sigma_goal)) / len(delta_sigma)
        err_ds_jac = np.sum(2 * delta_sigma_jac * (delta_sigma-delta_sigma_goal), axis=1)

        err_sum = err_smf + err_rpwp + err_ds
        err_jac_sum = err_smf_jac + err_rpwp_jac + err_ds_jac

    return err_sum, err_jac_sum


# wrapper for the above
def mse_smhm_no_quench_adam_wrapper(static_params, opt_params):
    """Wrapper of mse_smhm for adam.

    Parameters
    ---------
    static_params : array-like
        [smf_goal, rpwp_goal, delta_sigma_goal,
         log_halomass, log_hostmass, log_vmax_by_vmpeak,
         halo_x, halo_y, halo_z,
         upid, inside_subvol, time_since_infall, idx_to_deposit,
         part_x, part_y, part_z,
         rpbins, smf_bin_edges,
         mass_bin_low, mass_bin_high,
         zmax, boxsize]
    opt_params : array-like, shape(n_params,)
        Model parameters

    Returns
    -------
    error : float
        Mean squared error.
    error_jac : array-like, shape(n_params,)
        Jacobian of mean squared error with respect to model parameters.    

    """

    error, error_jac = mse_smhm(
            static_params[0], static_params[1], static_params[2],
            static_params[3], static_params[4], static_params[5],
            static_params[6], static_params[7], static_params[8],
            static_params[9], static_params[10], static_params[11], static_params[12],
            static_params[13], static_params[14], static_params[15],
            static_params[16], static_params[17],
            static_params[18], static_params[19],
            static_params[20], static_params[21],
            opt_params
    )

    return error, error_jac

def mse_rpwp_quench(
    rpwp_q_goal,
    rpwp_nq_goal,
    log_halomass,
    log_hostmass,
    log_vmax_by_vmpeak,
    halo_x, halo_y, halo_z,
    upid,
    inside_subvol,
    time_since_infall,
    idx_to_deposit,
    rpbins,
    mass_bin_low, mass_bin_high,
    zmax,
    boxsize,
    theta
):
    """Mean squared error for quenched rp wp(rp).

    Parameters
    ---------

    """

    # calculate weights
    w_q, dw_q, w_nq, dw_nq = compute_weight_and_jac_quench(
                                log_halomass,
                                log_hostmass,
                                log_vmax_by_vmpeak,
                                upid,
                                time_since_infall,
                                idx_to_deposit,
                                mass_bin_low, mass_bin_high,
                                theta
    )

    wgt_mask_quench = w_q > 0
    wgt_mask_no_quench = w_nq > 0

    # do measurement
    rpwp_q, rpwp_q_jac = compute_rpwp(
                            x1=halo_x[wgt_mask_quench],
                            y1=halo_y[wgt_mask_quench],
                            z1=halo_z[wgt_mask_quench],
                            w1=w_q[wgt_mask_quench],
                            w1_jac=dw_q[:, wgt_mask_quench],
                            inside_subvol=inside_subvol[wgt_mask_quench],
                            rpbins=rpbins,
                            zmax=zmax,
                            boxsize=boxsize
    )
                        
    rpwp_nq, rpwp_nq_jac = compute_rpwp(
                            x1=halo_x[wgt_mask_no_quench],
                            y1=halo_y[wgt_mask_no_quench],
                            z1=halo_z[wgt_mask_no_quench],
                            w1=w_q[wgt_mask_no_quench],
                            w1_jac=dw_q[:, wgt_mask_no_quench],
                            inside_subvol=inside_subvol[wgt_mask_no_quench],
                            rpbins=rpbins,
                            zmax=zmax,
                            boxsize=boxsize
    )

    # error 
    err_sum, err_sum_jac = None, None
    if RANK == 0:
        err_q = np.sum((rpwp_q-rpwp_q_goal)*(rpwp_q-rpwp_q_goal)) / len(rpwp_q)
        err_q_jac = np.sum(2 * rpwp_q_jac * (rpwp_q-rpwp_q_goal), axis=1) / len(rpwp_q)

        err_nq = np.sum((rpwp_nq-rpwp_nq_goal)*(rpwp_nq-rpwp_nq_goal)) / len(rpwp_nq)
        err_nq_jac = np.sum(2 * rpwp_nq_jac * (rpwp_nq-rpwp_nq_goal), axis=1) / len(rpwp_nq)

        err_sum = err_q + err_nq
        err_sum_jac = err_q_jac + err_nq_jac

    return err_sum, err_sum_jac


# wrapper for use within adam function
def mse_rpwp_quench_adam_wrapper(static_params, opt_params):
    """Wrapper of above for use as adam error function.

    Parameters
    ----------
    static_params : array-like
        [rpwp_q_goal, rpwp_nq_goal,
         log_halomass, log_halo_hostmass, log_vmax_by_vmpeak,
         halo_x, halo_y, halo_z,
         upid, inside_subvol, time_since_infall, idx_to_deposit,
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
    
