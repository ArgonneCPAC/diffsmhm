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
    DEFAULT_PARAM_VALUES as smhm_params_default
)
from diffsmhm.galhalo_models.sigmoid_smhm_sigma import (
    DEFAULT_PARAM_VALUES as smhm_sigma_params_default
)
from diffsmhm.galhalo_models.sigmoid_disruption import (
    DEFAULT_PARAM_VALUES as disruption_params_default
)

from diffsmhm.loader import load_and_chop_data_bolshoi_planck

from diffsmhm.galhalo_models.merging import _calculate_indx_to_deposit

from diff_sm import compute_weight_and_jac

from smf import smf_mpi_comp_and_reduce
from rpwp import compute_rpwp
from delta_sigma import compute_delta_sigma

from adam import adam
from error import mse_smhm_adam_wrapper

# data files and params 
halo_file="/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"
particle_file="/home/jwick/data/hlist_1.00231.particles.halotools_v0p4.hdf5"
box_length = 250.0 # Mpc
buff_wprp = 25.0 # Mpc

mass_bin_edges = np.array([10.6, 10.8], dtype=np.float64)
smf_bin_edges = np.arange(10.6, 10.8, 0.025, dtype=np.float64)

rpbins = np.logspace(-1, 1.5, 13, dtype=np.float64)
zmax = 20.0

theta = np.array([
            smhm_params_default["smhm_logm_crit"],
            smhm_params_default["smhm_ratio_logm_crit"],
            smhm_params_default["smhm_k_logm"],
            smhm_params_default["smhm_lowm_index"],
            smhm_params_default["smhm_highm_index"],

            smhm_sigma_params_default["smhm_sigma_low"],
            smhm_sigma_params_default["smhm_sigma_high"],
            smhm_sigma_params_default["smhm_sigma_logm_pivot"],
            smhm_sigma_params_default["smhm_sigma_logm_width"],

            disruption_params_default["satmerg_logmhost_crit"],
            disruption_params_default["satmerg_logmhost_k"],
            disruption_params_default["satmerg_logvr_crit_dwarfs"],
            disruption_params_default["satmerg_logvr_crit_clusters"],
            disruption_params_default["satmerg_logvr_k"]
], dtype=np.float64)

n_params = len(theta)
n_rpbins = len(rpbins) - 1

# 1) load data
halos, particles = load_and_chop_data_bolshoi_planck(
                        particle_file,
                        halo_file,
                        box_length,
                        buff_wprp,
                        host_mpeak_cut=14.7
)

# TMP: for laptop speeed boost
particles["x"] = particles["x"][0:10:-1]
particles["y"] = particles["y"][0:10:-1]
particles["z"] = particles["z"][0:10:-1]

idx_to_deposit = _calculate_indx_to_deposit(halos["upid"], halos["halo_id"])

print(RANK, len(halos["halo_id"]), flush=True)

# 2) obtain "goal" measurements
parameter_perturbations = np.random.uniform(low=0.95, high=1.05, size=n_params)

theta_goal = theta * parameter_perturbations

# smf only needs model params as we reweight for each small bin
smf_goal, _ = smf_mpi_comp_and_reduce(
                        halos["logmpeak"],
                        halos["loghost_mpeak"],
                        halos["logvmax_frac"],
                        halos["upid"],
                        idx_to_deposit,
                        halos["_inside_subvol"],
                        smf_bin_edges,
                        theta_goal
)

# delta sigma and rpwp need weights, so we calculate them once here
w, dw = compute_weight_and_jac(
                        halos["logmpeak"],
                        halos["loghost_mpeak"],
                        halos["logvmax_frac"],
                        halos["upid"],
                        idx_to_deposit,
                        mass_bin_edges[0], mass_bin_edges[1],
                        theta_goal
)

wgt_mask = w > 0

# rpwp
rpwp_goal, _ = compute_rpwp(
                        x1=halos["halo_x"][wgt_mask],
                        y1=halos["halo_y"][wgt_mask],
                        z1=halos["halo_z"][wgt_mask],
                        w1=w[wgt_mask],
                        w1_jac=dw[:, wgt_mask],
                        inside_subvol=halos["_inside_subvol"][wgt_mask],
                        rpbins=rpbins,
                        zmax=zmax,
                        boxsize=box_length
)

# delta sigma
delta_sigma_goal, _ = compute_delta_sigma(
                        xh=halos["halo_x"][wgt_mask],
                        yh=halos["halo_y"][wgt_mask],
                        zh=halos["halo_z"][wgt_mask],
                        wh=w[wgt_mask],
                        wh_jac=dw[:, wgt_mask],
                        xp=particles["x"],
                        yp=particles["y"],
                        zp=particles["z"],
                        inside_subvol=halos["_inside_subvol"][wgt_mask],
                        rpbins=rpbins,
                        zmax=zmax,
                        boxsize=box_length
)

# 3) do optimization
theta_init = np.copy(theta)

static_params = [
                 smf_goal, rpwp_goal, delta_sigma_goal,
                 halos["logmpeak"], halos["loghost_mpeak"], halos["vmax_frac"],
                 halos["halo_x"], halos["halo_y"], halos["halo_z"],
                 halos["upid"], halos["_inside_subvol"], halos["time_since_infall"],
                 idx_to_deposit,
                 particles["x"], particles["y"], particles["z"],
                 rpbins, smf_bin_edges,
                 mass_bin_edges[0], mass_bin_edges[1],
                 zmax,
                 box_length
]

theta, error_history = adam(
                        static_params,
                        theta,
                        maxiter=5,
                        minerr=4.0,
                        err_func=mse_smhm_adam_wrapper
)

# 4) make figures

# calculate quantites with initial theta parameters
smf_init, _ = smf_mpi_comp_and_reduce(
                        halos["logmpeak"],
                        halos["loghost_mpeak"],
                        halos["logvmax_frac"],
                        halos["upid"],
                        idx_to_deposit,
                        halos["_inside_subvol"],
                        smf_bin_edges,
                        theta_init
)

w, dw = compute_weight_and_jac(
                        halos["logmpeak"],
                        halos["loghost_mpeak"],
                        halos["logvmax_frac"],
                        halos["upid"],
                        idx_to_deposit,
                        mass_bin_edges[0], mass_bin_edges[1],
                        theta_init
)

wgt_mask = w > 0

rpwp_init, _ = compute_rpwp(
                        x1=halos["halo_x"][wgt_mask],
                        y1=halos["halo_y"][wgt_mask],
                        z1=halos["halo_z"][wgt_mask],
                        w1=w[wgt_mask],
                        w1_jac=dw[:, wgt_mask],
                        inside_subvol=halos["_inside_subvol"][wgt_mask],
                        rpbins=rpbins,
                        zmax=zmax,
                        boxsize=box_length
)

delta_sigma_init, _ = compute_delta_sigma(
                        xh=halos["halo_x"][wgt_mask],
                        yh=halos["halo_y"][wgt_mask],
                        zh=halos["halo_z"][wgt_mask],
                        wh=w[wgt_mask],
                        wh_jac=dw[:, wgt_mask],
                        xp=particles["x"],
                        yp=particles["y"],
                        zp=particles["z"],
                        inside_subvol=halos["_inside_subvol"][wgt_mask],
                        rpbins=rpbins,
                        zmax=zmax,
                        boxsize=box_length
)


# err history figure
if RANK == 0:
    fig = plt.figure(figsize=(12,8), facecolor="w")

    plt.plot(error_history)

    plt.xlabel("Iteration Number", fontsize=16)
    plt.ylabel("Sum Squared Errors", fontsize=16)
    plt.title("Error per Iteration", fontsize=20)

    plt.tight_layout()

    plt.savefig("error_history.png")

# smf figure
smf, _ = smf_mpi_comp_and_reduce(
                        halos["logmpeak"],
                        halos["loghost_mpeak"],
                        halos["logvmax_frac"],
                        halos["upid"],
                        idx_to_deposit,
                        halos["_inside_subvol"],
                        smf_bin_edges,
                        theta
)

if RANK == 0:
    fig = plt.figure(figsize=(12,8), facecolor="w")
    
    plt.plot(smf_bin_edges[:-1], smf_init, linewidth=4)
    plt.plot(smf_bin_edges[:-1], smf, linewidth=4)
    plt.plot(smf_bin_edges[:-1], smf_goal, linestyle="dashed")

    plt.legend(["Initial Guess", "Final Guess", "Goal"])

    plt.xlabel("log10 Stellar Mass", fontsize=16)
    plt.ylabel("Quantity Galaxies", fontsize=16)
    plt.title("Stellar Mass Function", fontsize=20)

    plt.savefig("smf_opt.png")

# rpwp figure
w, dw = compute_weight_and_jac(
                        halos["logmpeak"],
                        halos["loghost_mpeak"],
                        halos["logvmax_frac"],
                        halos["upid"],
                        idx_to_deposit,
                        mass_bin_edges[0], mass_bin_edges[1],
                        theta
)

wgt_mask = w > 0

rpwp, _ = compute_rpwp(
                        x1=halos["halo_x"][wgt_mask],
                        y1=halos["halo_y"][wgt_mask],
                        z1=halos["halo_z"][wgt_mask],
                        w1=w[wgt_mask],
                        w1_jac=dw[:, wgt_mask],
                        inside_subvol=halos["_inside_subvol"][wgt_mask],
                        rpbins=rpbins,
                        zmax=zmax,
                        boxsize=box_length
)

if RANK == 0:
    fig = plt.figure(figsize=(12,8), facecolor="w")

    plt.plot(rpbins[:-1], rpwp_init, linewidth=4)
    plt.plot(rpbins[:-1], rpwp, linewidth=4)
    plt.plot(rpbins[:-1], rpwp_goal, linestyle="dashed")

    plt.legend(["Initial Guess", "Final Guess", "Goal"])

    plt.xlabel("rp", fontsize=16)
    plt.ylabel("rp wp(rp)", fontsize=16)
    plt.title("Correlation Function", fontsize=20)

    plt.savefig("rpwp_opt.png")

# delta sigma figure

delta_sigma, _ = compute_delta_sigma(
                        xh=halos["halo_x"][wgt_mask],
                        yh=halos["halo_y"][wgt_mask],
                        zh=halos["halo_z"][wgt_mask],
                        wh=w[wgt_mask],
                        wh_jac=dw[:, wgt_mask],
                        xp=particles["x"],
                        yp=particles["y"],
                        zp=particles["z"],
                        inside_subvol=halos["_inside_subvol"][wgt_mask],
                        rpbins=rpbins,
                        zmax=zmax,
                        boxsize=box_length
)
if RANK == 0:
    fig = plt.figure(figsize=(12,8), facecolor="w")

    plt.plot(rpbins[:-1], delta_sigma_init, linewidth=4)
    plt.plot(rpbins[:-1], delta_sigma, linewidth=4)
    plt.plot(rpbins[:-1], delta_sigma_goal, linestyle="dashed")

    plt.legend(["Initial Guess", "Final Guess", "Goal"])

    plt.xlabel("rp", fontsize=16)
    plt.ylabel("Delta Sigma", fontsize=16)
    plt.title("Lensing", fontsize=20)

    plt.savefig("delta_sigma_opt.png")
