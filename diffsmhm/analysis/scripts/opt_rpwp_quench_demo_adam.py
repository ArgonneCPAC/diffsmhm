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

# data files and params
halo_file = "/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"
particle_file = "/home/jwick/data/hlist_1.00231.particles.halotools_v0p4.hdf5"
box_length = 250.0  # Mpc
buff_wprp = 20.0  # Mpc

mass_bin_edges = np.array([10.6, 11.2], dtype=np.float64)

rpbins = np.logspace(-1, 1.2, 13, dtype=np.float64)
zmax = 20.0  # Mpc

theta = np.array(list(smhm_params.values()) +
                 list(smhm_sigma_params.values()) +
                 list(disruption_params.values()) +
                 list(quenching_params.values()), dtype=np.float64)

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

# rpwp, quenched and unquenched
w_q, dw_q, w_nq, dw_nq = compute_weight_and_jac_quench(
                        halos["logmpeak"],
                        halos["loghost_mpeak"],
                        halos["logvmax_frac"],
                        halos["upid"],
                        halos["time_since_infall"],
                        idx_to_deposit,
                        mass_bin_edges[0], mass_bin_edges[1],
                        theta_goal
)

if RANK == 0:
    print("goal weights done", flush=True)

# goal rpwp computation
rpwp_q_goal, _ = compute_rpwp(
                        x1=halos["halo_x"],
                        y1=halos["halo_y"],
                        z1=halos["halo_z"],
                        w1=w_q,
                        w1_jac=dw_q,
                        inside_subvol=halos["_inside_subvol"],
                        rpbins=rpbins,
                        zmax=zmax,
                        boxsize=box_length
)

rpwp_nq_goal, _ = compute_rpwp(
                        x1=halos["halo_x"],
                        y1=halos["halo_y"],
                        z1=halos["halo_z"],
                        w1=w_nq,
                        w1_jac=dw_nq,
                        inside_subvol=halos["_inside_subvol"],
                        rpbins=rpbins,
                        zmax=zmax,
                        boxsize=box_length
)

if RANK == 0:
    print("goal wprp done", flush=True)

# 3) do optimization
theta_init = np.copy(theta)

# copy necessary params into static_params arrray
static_params = [
                 rpwp_q_goal, rpwp_nq_goal,
                 halos["logmpeak"], halos["loghost_mpeak"], halos["logvmax_frac"],
                 halos["halo_x"], halos["halo_y"], halos["halo_z"],
                 halos["time_since_infall"],
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
                        maxiter=999999,
                        minerr=0.0,
                        tmax=110*60,
                        err_func=mse_rpwp_quench_adam_wrapper,
                        a=0.005
)

# 4) Make figures

# calculate wprp with initial theta parameters
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

rpwp_q_init, _ = compute_rpwp(
                    x1=halos["halo_x"],
                    y1=halos["halo_y"],
                    z1=halos["halo_z"],
                    w1=w_q,
                    w1_jac=dw_q,
                    inside_subvol=halos["_inside_subvol"],
                    rpbins=rpbins,
                    zmax=zmax,
                    boxsize=box_length
)

rpwp_nq_init, _ = compute_rpwp(
                    x1=halos["halo_x"],
                    y1=halos["halo_y"],
                    z1=halos["halo_z"],
                    w1=w_nq,
                    w1_jac=dw_nq,
                    inside_subvol=halos["_inside_subvol"],
                    rpbins=rpbins,
                    zmax=zmax,
                    boxsize=box_length
)

# calculate wprp with final theta parameters
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

rpwp_q, _ = compute_rpwp(
                    x1=halos["halo_x"],
                    y1=halos["halo_y"],
                    z1=halos["halo_z"],
                    w1=w_q,
                    w1_jac=dw_q,
                    inside_subvol=halos["_inside_subvol"],
                    rpbins=rpbins,
                    zmax=zmax,
                    boxsize=box_length
)

rpwp_nq, _ = compute_rpwp(
                    x1=halos["halo_x"],
                    y1=halos["halo_y"],
                    z1=halos["halo_z"],
                    w1=w_nq,
                    w1_jac=dw_nq,
                    inside_subvol=halos["_inside_subvol"],
                    rpbins=rpbins,
                    zmax=zmax,
                    boxsize=box_length
)

if RANK == 0:
    print("eh:", len(error_history), error_history[0], error_history[-1])
    print("goal params:", theta_goal)
    print("start params:", theta_init)
    print("end params:", theta)

# error history figure
if RANK == 0:
    err_diff = error_history[:-1] - error_history[1:]

    fig = plt.figure(figsize=(12, 8), facecolor="w")

    plt.plot(err_diff)

    plt.xlabel("Iteration Number", fontsize=16)
    plt.ylabel("Change in Error", fontsize=16)
    plt.title("Change in Error per Iteration", fontsize=20)

    plt.yscale("log")

    plt.tight_layout()
    plt.savefig("figures/rpwp_quench_demo_adam_edpi.png")

# rpwp figure
if RANK == 0:
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), facecolor="w")

    # quenched
    axs[0].plot(rpbins[:-1], rpwp_q_init, linewidth=4)
    axs[0].plot(rpbins[:-1], rpwp_q, linewidth=4)
    axs[0].plot(rpbins[:-1], rpwp_q_goal, linewidth=2, c="k")

    axs[0].legend(["Initial Guess", "Final Guess", "Goal"])

    axs[0].set_xlabel("rp", fontsize=16)
    axs[0].set_ylabel("rp wp(rp)", fontsize=16)
    axs[0].set_title("Quenched Correlation Function; 10.6->11.2", fontsize=20)

    axs[0].set_xscale("log")

    # un quenched
    axs[1].plot(rpbins[:-1], rpwp_nq_init, linewidth=4)
    axs[1].plot(rpbins[:-1], rpwp_nq, linewidth=4)
    axs[1].plot(rpbins[:-1], rpwp_nq_goal, linewidth=2, c="k")

    axs[1].set_xlabel("rp", fontsize=16)
    axs[1].set_ylabel("rp wp(rp)", fontsize=16)
    axs[1].set_title("Unquenched Correlation Function", fontsize=20)

    axs[1].set_xscale("log")

    plt.savefig("figures/rpwp_quench_demo_adam.png")
