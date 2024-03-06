import numpy as np
import matplotlib.pyplot as plt

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1

from diffsmhm.galhalo_models.sigmoid_smhm import (
    DEFAULT_PARAM_VALUES as smhm_params
)
from diffsmhm.galhalo_models.sigmoid_smhm_sigma import (
    DEFAULT_PARAM_VALUES as smhm_sigma_params
)
from diffsmhm.galhalo_models.sigmoid_disruption import (
    DEFAULT_PARAM_VALUES as disruption_params
)
from diffsmhm.galhalo_models.sigmoid_quenching import (
    DEFAULT_PARAM_VALUES as quenching_params
)

from diffsmhm.loader import load_and_chop_data_bolshoi_planck

from diffsmhm.galhalo_models.merging import _calculate_indx_to_deposit

from diff_sm import compute_weight_and_jac_quench
from rpwp import compute_rpwp

from adam import adam
from error import mse_rpwp_quench_adam_wrapper


# 1) load watson data for "ideal" metric
quenchfile = "/home/jwick/data/watson/smthresh10.6.passive.wp"
unquenchfile = "/home/jwick/data/watson/smthresh10.6.active.wp"

def read_watson_wp(fname,h=0.7):
    dw = np.genfromtxt(fname,usecols=[1,2,3],names=['r','wp','wperr'])
    dw['r'] /= h
    dw['wp'] /= h
    dw['wperr'] /= h
    return dw

watson_quench = read_watson_wp(quenchfile, h=1.0)
watson_unquench = read_watson_wp(unquenchfile, h=1.0)

# 2) load simulation data
halo_file="/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"                                          
particle_file="/home/jwick/data/hlist_1.00231.particles.halotools_v0p4.hdf5"    
box_length = 250.0 # Mpc                                                        
buff_wprp = 21.0 # Mpc  
zmax = 20.0 # Mpc

mass_bin_edges = np.array([10.6, 100.0], dtype=np.float64)

# TODO: this is possibly not right
rpbins = np.append(watson_quench["r"], 20.0)
if RANK == 0: print(rpbins, flush=True)

theta = np.array([
            smhm_params["smhm_logm_crit"],
            smhm_params["smhm_ratio_logm_crit"],
            smhm_params["smhm_k_logm"],
            smhm_params["smhm_lowm_index"],
            smhm_params["smhm_highm_index"],
            smhm_sigma_params["smhm_sigma_low"],
            smhm_sigma_params["smhm_sigma_high"],
            smhm_sigma_params["smhm_sigma_logm_pivot"],
            smhm_sigma_params["smhm_sigma_logm_width"],
            disruption_params["satmerg_logmhost_crit"],
            disruption_params["satmerg_logmhost_k"],
            disruption_params["satmerg_logvr_crit_dwarfs"],
            disruption_params["satmerg_logvr_crit_clusters"],
            disruption_params["satmerg_logvr_k"],
            quenching_params["fq_cens_logm_crit"],
            quenching_params["fq_cens_k"],
            quenching_params["fq_cens_ylo"],
            quenching_params["fq_cens_yhi"],
            quenching_params["fq_satboost_logmhost_crit"],
            quenching_params["fq_satboost_logmhost_k"],
            quenching_params["fq_satboost_clusters"],
            quenching_params["fq_sat_delay_time"],
            quenching_params["fq_sat_tinfall_k"]
], dtype=np.float64)
theta_init = np.copy(theta)

n_params = len(theta)
n_rpbins = len(rpbins) - 1

halos, _ = load_and_chop_data_bolshoi_planck(
                particle_file,
                halo_file,
                box_length,
                buff_wprp,
                host_mpeak_cut=0.0)

idx_to_deposit = _calculate_indx_to_deposit(halos["upid"], halos["halo_id"])

# 3) optimize

# copy necessary params into static_params array
static_params = [
                    watson_quench["wp"]*watson_quench["r"],
                    watson_unquench["wp"]*watson_quench["r"],
                    halos["logmpeak"], halos["loghost_mpeak"], halos["logvmax_frac"],
                    halos["halo_x"], halos["halo_y"], halos["halo_z"],
                    halos["upid"], halos["_inside_subvol"], halos["time_since_infall"],
                    idx_to_deposit,
                    rpbins,
                    mass_bin_edges[0],
                    mass_bin_edges[1],
                    zmax,
                    box_length
]

theta, error_history = adam(
                        static_params, 
                        theta,
                        maxiter=9999999,
                        minerr=1.0,
                        tmax=60*100,
                        err_func=mse_rpwp_quench_adam_wrapper
)


# 4) export data for ploting
# TODO: do something actually nice for this

# given original and new theta, let's plot the change of rpwp
# first, we get initial guess rpwp
w_q, dw_q, w_nq, dw_nq = compute_weight_and_jac_quench(
                            halos["logmpeak"],
                            halos["loghost_mpeak"],
                            halos["logvmax_frac"],
                            halos["upid"],
                            halos["time_since_infall"],
                            idx_to_deposit,
                            mass_bin_edges[0], mass_bin_edges[1],
                            theta_init
)

wgt_mask_quench = w_q > 0
wgt_mask_no_quench = w_nq > 0

rpwp_q_init, _ = compute_rpwp(
                    x1=halos["halo_x"][wgt_mask_quench],
                    y1=halos["halo_y"][wgt_mask_quench],
                    z1=halos["halo_z"][wgt_mask_quench],
                    w1=w_q[wgt_mask_quench],
                    w1_jac=dw_q[:, wgt_mask_quench],
                    inside_subvol=halos["_inside_subvol"][wgt_mask_quench],
                    rpbins=rpbins,
                    zmax=zmax,
                    boxsize=box_length
)

rpwp_nq_init, _ = compute_rpwp(
                    x1=halos["halo_x"][wgt_mask_no_quench],
                    y1=halos["halo_y"][wgt_mask_no_quench],
                    z1=halos["halo_z"][wgt_mask_no_quench],
                    w1=w_nq[wgt_mask_no_quench],
                    w1_jac=dw_nq[:, wgt_mask_no_quench],
                    inside_subvol=halos["_inside_subvol"][wgt_mask_no_quench],
                    rpbins=rpbins,
                    zmax=zmax,
                    boxsize=box_length
)

# now final quess rpwp
w_q, dw_q, w_nq, dw_nq = compute_weight_and_jac_quench(
                            halos["logmpeak"],
                            halos["loghost_mpeak"],
                            halos["logvmax_frac"],
                            halos["upid"],
                            halos["time_since_infall"],
                            idx_to_deposit,
                            mass_bin_edges[0], mass_bin_edges[1],
                            theta
)

wgt_mask_quench = w_q > 0
wgt_mask_no_quench = w_nq > 0

rpwp_q_final, _ = compute_rpwp(
                    x1=halos["halo_x"][wgt_mask_quench],
                    y1=halos["halo_y"][wgt_mask_quench],
                    z1=halos["halo_z"][wgt_mask_quench],
                    w1=w_q[wgt_mask_quench],
                    w1_jac=dw_q[:, wgt_mask_quench],
                    inside_subvol=halos["_inside_subvol"][wgt_mask_quench],
                    rpbins=rpbins,
                    zmax=zmax,
                    boxsize=box_length
)

rpwp_nq_final, _ = compute_rpwp(
                    x1=halos["halo_x"][wgt_mask_no_quench],
                    y1=halos["halo_y"][wgt_mask_no_quench],
                    z1=halos["halo_z"][wgt_mask_no_quench],
                    w1=w_nq[wgt_mask_no_quench],
                    w1_jac=dw_nq[:, wgt_mask_no_quench],
                    inside_subvol=halos["_inside_subvol"][wgt_mask_no_quench],
                    rpbins=rpbins,
                    zmax=zmax,
                    boxsize=box_length
)

if RANK == 0:
    print(len(error_history), error_history[0], error_history[-1])
    print(theta)

    fig, axs = plt.subplots(1,2, figsize=(16,8), facecolor="w")

    # quenched
    axs[0].plot(rpbins[:-1], rpwp_q_init, linewidth=4)
    axs[0].plot(rpbins[:-1], rpwp_q_final, linewidth=4)
    axs[0].plot(rpbins[:-1], watson_quench["wp"]*watson_quench["r"], linewidth=2, c="k")

    axs[0].set_xscale("log")

    axs[0].legend(["Default Params", "Optimized Params", "Data"])

    axs[0].set_xlabel("rp", fontsize=16)
    axs[0].set_ylabel("rp wp(rp)", fontsize=16)
    axs[0].set_title("Quenched Correlation Function", fontsize=20)

    # un quenched
    axs[1].plot(rpbins[:-1], rpwp_nq_init, linewidth=4)
    axs[1].plot(rpbins[:-1], rpwp_nq_final, linewidth=4)
    axs[1].plot(rpbins[:-1], watson_unquench["wp"]*watson_unquench["r"], linewidth=2, c="k")

    axs[1].set_xscale("log")

    axs[1].set_xlabel("rp", fontsize=16)
    axs[1].set_ylabel("rp wp(rp)", fontsize=16)
    axs[1].set_title("Unquenched Correlation Function", fontsize=20)

    plt.savefig("rpwp_watson.png")
