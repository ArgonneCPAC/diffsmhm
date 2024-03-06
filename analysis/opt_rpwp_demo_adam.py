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

from diffsmhm.loader import load_and_chop_data_bolshoi_planck

from diffsmhm.galhalo_models.merging import _calculate_indx_to_deposit

from diff_sm import compute_weight_and_jac

from rpwp import compute_rpwp

from adam import adam

from error import mse_rpwp_adam_wrapper

# data files and params
halo_file="/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"
particle_file="/home/jwick/data/hlist_1.00231.particles.halotools_v0p4.hdf5"
box_length = 250.0 # Mpc
buff_wprp = 20.0 # Mpc

# 10.6, 11.2
mass_bin_edges = np.array([10.6, 11.2], dtype=np.float64)

rpbins = np.logspace(-1, 1.2, 13, dtype=np.float64)
zmax = 20.0 # Mpc

theta = np.array(list(smhm_params.values()) + 
                 list(smhm_sigma_params.values()) + 
                 list(disruption_params.values()), dtype=np.float64)

n_params = len(theta)
n_rpbins = len(rpbins) - 1

# 1) load data
halos, _ = load_and_chop_data_bolshoi_planck(
            particle_file,
            halo_file,
            box_length,
            buff_wprp,
            host_mpeak_cut=0.0)

idx_to_deposit = _calculate_indx_to_deposit(halos["upid"], halos["halo_id"])

print(RANK, len(halos["halo_id"]), flush=True)

# 2) obtain "goal" measurement
np.random.seed(999)
parameter_perturbations = np.random.uniform(low=0.98, high=1.02, size=n_params)

theta_goal = theta * parameter_perturbations

# weights
w, dw = compute_weight_and_jac(
            halos["logmpeak"],
            halos["loghost_mpeak"],
            halos["vmax_frac"],
            halos["upid"],
            idx_to_deposit,
            mass_bin_edges[0], mass_bin_edges[1],
            theta_goal
)

wgt_mask = w > 0.0

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

if RANK == 0: print("goal measurement done", flush=True)

# 3) do optimization
theta_init = np.copy(theta)

# copy necessary params into static_params array
static_params = [
                    rpwp_goal,
                    halos["logmpeak"], halos["loghost_mpeak"], halos["logvmax_frac"],
                    halos["halo_x"], halos["halo_y"], halos["halo_z"],
                    halos["upid"], halos["_inside_subvol"],
                    idx_to_deposit,
                    rpbins,
                    mass_bin_edges[0], mass_bin_edges[1],
                    zmax,
                    box_length
]

theta, error_history = adam(
                        static_params,
                        theta,
                        err_func=mse_rpwp_adam_wrapper,
                        maxiter=999999999,
                        minerr=0.0,
                        tmax=105*60
)

if RANK == 0:
    print(len(error_history), error_history[0], error_history[-1], flush=True)

# 4) make figure

# calculate rpwp with initial theta parameters
w, dw = compute_weight_and_jac(
            halos["logmpeak"],
            halos["loghost_mpeak"],
            halos["logvmax_frac"],
            halos["upid"],
            idx_to_deposit,
            mass_bin_edges[0], mass_bin_edges[1],
            theta_init
)

wgt_mask = w > 0.0

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

if RANK == 0: print("init calc done", flush=True)

# calculate rpwp with the final theta parameters
w, dw = compute_weight_and_jac(
            halos["logmpeak"],
            halos["loghost_mpeak"],
            halos["logvmax_frac"],
            halos["upid"],
            idx_to_deposit,
            mass_bin_edges[0], mass_bin_edges[1],
            theta
)

wgt_mask = w > 0.0

rpwp_final, _ = compute_rpwp(
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

if RANK == 0: print("final calc done", flush=True)

# rpwp figure
if RANK == 0:
    fig = plt.figure(figsize=(10,8), facecolor="w")

    plt.plot(rpbins[:-1], rpwp_init, linewidth=4, c="tab:blue")
    plt.plot(rpbins[:-1], rpwp_final, linewidth=4, c="tab:orange")
    plt.plot(rpbins[:-1], rpwp_goal, linewidth=1, c="k")

    plt.xlabel("rp", fontsize=16)
    plt.ylabel("rp wp(rp)", fontsize=16)
    plt.title("Full Correlation Function", fontsize=20)

    plt.legend(["initial guess", "final guess", "goal"])

    plt.xscale("log")

    plt.savefig("rpwp_demo.png")
