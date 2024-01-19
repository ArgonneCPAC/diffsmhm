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

if COMM is not None:
    import mpipartition

from diffsmhm.loader import load_and_chop_data_bolshoi_planck

from diffsmhm.galhalo_models.merging import _calculate_indx_to_deposit
from diffsmhm.diff_stats.mpi.wprp import wprp_mpi_comp_and_reduce
from diffsmhm.diff_stats.cuda.wprp import wprp_mpi_kernel_cuda

from diffsmhm.galhalo_models.sigmoid_smhm import (
    DEFAULT_PARAM_VALUES as smhm_params_default
)
from diffsmhm.galhalo_models.sigmoid_smhm_sigma import (
    DEFAULT_PARAM_VALUES as smhm_sigma_params_default
)
from diffsmhm.galhalo_models.sigmoid_disruption import (
    DEFAULT_PARAM_VALUES as disruption_params_default
)

from diff_sm import (
    compute_sm_and_jac,
    compute_sigma_and_jac,
    compute_weight_and_jac
)

from squared_error import se_rpwp

import matplotlib.pyplot as plt

# jax doesn't like integers
disruption_params_default["satmerg_logvr_crit_dwarfs"] *= 1.0

# data files and params
halo_file="/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"
particle_file="/home/jwick/data/hlist_1.00231.particles.halotools_v0p4.hdf5"
box_length = 250.0 # Mpc
buff_wprp = 25.0 # Mpc

important_keys_wprp = ["halo_x", "halo_y", "halo_z", "w", "dw", "_inside_subvol"]

rpbins = np.logspace(-1, 1.5, 13, dtype=np.float64)

# stellar mass bin
mass_bin_edges = np.array([10.6, 10.7], dtype=np.float64)

# perturb default parameters to obtain "goal" params
theta_goal = [
                smhm_params_default["smhm_logm_crit"]*1.02,
                smhm_params_default["smhm_ratio_logm_crit"]*1.02,
                smhm_params_default["smhm_k_logm"]*1.02,
                smhm_params_default["smhm_lowm_index"]*1.02,
                smhm_params_default["smhm_highm_index"]*1.02,
                smhm_sigma_params_default["smhm_sigma_low"]*0.92,
                smhm_sigma_params_default["smhm_sigma_high"]*0.92,
                smhm_sigma_params_default["smhm_sigma_logm_pivot"]*0.92,
                smhm_sigma_params_default["smhm_sigma_logm_width"]*0.92,
                disruption_params_default["satmerg_logmhost_crit"]*1.02,
                disruption_params_default["satmerg_logmhost_k"]*1.02,
                disruption_params_default["satmerg_logvr_crit_dwarfs"]*1.02,
                disruption_params_default["satmerg_logvr_crit_clusters"]*1.02,
                disruption_params_default["satmerge_logvr_k"]*1.02
]

n_params = len(theta_goal)

# load data
halos, particles = load_and_chop_data_bolshoi_planck(
                                                    particle_file,
                                                    halo_file,
                                                    box_length,
                                                    buff_wprp,
                                                    host_mpeak_cut=9.0
)

print(RANK, "data loaded", len(halos["halo_id"]), flush=True)

idx_to_deposit = _calculate_indx_to_deposit(halos["upid"], halos["halo_id"])

# calculate "ideal" wprp

halos["w"], halos["dw"] = compute_weight_and_jac(
                        halos["logmpeak"],
                        halos["loghost_mpeak"],
                        halos["logvmax_frac"],
                        idx_to_deposit,
                        mass_bin_edges[0], mass_bin_edges[1],
                        theta_goal
)

print(RANK, "Weights calculated", flush=True)

# mask out gals not in this bin for faster computation
bin_mask = halos["w"] > 0

# do the "ideal" wprp computation
wp_ideal, wp_ideal_grads = wprp_mpi_comp_and_reduce(
                            x1=halos["halo_x"][bin_mask],
                            y1=halos["halo_y"][bin_mask],
                            z1=halos["halo_z"][bin_mask],
                            w1=halos["w"][bin_mask],
                            w1_jac=halos["dw"][:, bin_mask],
                            inside_subvol=halos["_inside_subvol"][bin_mask],
                            rpbins_squared=rpbins**2,
                            zmax=buff_wprp,
                            boxsize=box_length,
                            kernel_func=wprp_mpi_kernel_cuda
)

rpwp_ideal = np.empty(len(rpbins)-1, dtype=np.float64)
rpwp_ideal_grads = np.empty((n_params, len(rpbins)-1), dtype=np.float64)
if RANK == 0:
    rpwp_ideal = rpbins[:-1] * wp_ideal
    rpwp_ideal_grads = rpbins[:-1] * wp_ideal_grads

print(RANK, "ideal computed", flush=True)

# parameters to optimize; as an array for easier updating
theta_init = np.array([
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
            disruption_params_default["satmerge_logvr_k"]
], dtype=np.float64)
theta = np.copy(theta_init)

# adam hyper params, note these are the values recomended in the paper
a = 0.001
b1 = 0.9
b2 = 0.999
eps = 10**-8

# initialize vectors
n_params = 14
m = np.zeros(n_params, dtype=np.float64)
v = np.zeros(n_params, dtype=np.float64)
t = 0

se_history = []

# Adam loop
se = float("inf")
maxiter = 50
while se > 5.0 and t < maxiter:
    t += 1

    # obtain error
    se, se_grad = se_rpwp(
                rpwp_ideal,
                mass_bin_edges[0], mass_bin_edges[1],
                idx_to_deposit,
                halos["halo_x"], halos["halo_y"], halos["halo_z"],
                halos["_inside_subvol"],
                rpbins,
                buff_wprp,
                box_length,
                halos["logmpeak"], halos["loghost_mpeak"], halos["logvmax_frac"],
                theta
    )
    # broadcast error
    se = COMM.bcast(se, root=0)
    se_grad = COMM.bcast(se_grad)

    # update biased first moment
    m = b1*m + (1-b1)*se_grad
    # update biased second moment
    v = b2*v + (1-b2)*se_grad**2
    # bias correct first moment
    mhat = m/(1 - b1**t)
    # bias correct second moment
    vhat = v/(1 - b2**t)
    # update parameters
    theta -= a*mhat/(np.sqrt(vhat)+eps)

    if RANK == 0: 
        print(se)
    se_history.append(se)

if RANK == 0: print("opt done", flush=True)

# let's make a figure
# plot initial guess, final guess, goal rpwp

fig = None
if RANK == 0:
    fig = plt.figure(figsize=(12,8), facecolor='w')

# plot initial guess
# weights 
halos["w"], halos["dw"] = compute_weight_and_jac(
                        halos["logmpeak"],
                        halos["loghost_mpeak"],
                        halos["logvmax_frac"],
                        idx_to_deposit,
                        mass_bin_edges[0], mass_bin_edges[1],
                        theta_init
)

# mask out gals not in this bin
bin_mask = halos["w"] > 0

wp_init, _ = wprp_mpi_comp_and_reduce(
                            x1=halos["halo_x"][bin_mask],
                            y1=halos["halo_y"][bin_mask],
                            z1=halos["halo_z"][bin_mask],
                            w1=halos["w"][bin_mask],
                            w1_jac=halos["dw"][:, bin_mask],
                            inside_subvol=halos["_inside_subvol"][bin_mask],
                            rpbins_squared=rpbins**2,
                            zmax=buff_wprp,
                            boxsize=box_length,
                            kernel_func=wprp_mpi_kernel_cuda
)

if RANK == 0:
    plt.plot(rpbins[:-1], rpbins[:-1]*wp_init, linewidth=4)
    print("ig done", flush=True)

# plot final guess 
halos["w"], halos["dw"] = compute_weight_and_jac(
                        halos["logmpeak"],
                        halos["loghost_mpeak"],
                        halos["logvmax_frac"],
                        idx_to_deposit,
                        mass_bin_edges[0], mass_bin_edges[1],
                        theta
)

# mask out halos not in this bin 
bin_mask = halos["w"] > 0
wp_final, _ = wprp_mpi_comp_and_reduce(
                            x1=halos["halo_x"][bin_mask],
                            y1=halos["halo_y"][bin_mask],
                            z1=halos["halo_z"][bin_mask],
                            w1=halos["w"][bin_mask],
                            w1_jac=halos["dw"][:, bin_mask],
                            inside_subvol=halos["_inside_subvol"][bin_mask],
                            rpbins_squared=rpbins**2,
                            zmax=buff_wprp,
                            boxsize=box_length,
                            kernel_func=wprp_mpi_kernel_cuda
)

if RANK == 0:
    plt.plot(rpbins[:-1], rpbins[:-1]*wp_final, linewidth=4)
    print("fg done", flush=True)

# plot goal
if RANK == 0:
    plt.plot(rpbins[:-1], rpwp_ideal, linestyle='dashed')

    plt.legend(["Initial Guess", "Final Guess", "Target"])

    plt.xlabel("rp", fontsize=16)
    plt.ylabel("rp wp(rp)", fontsize=16)
    plt.title("Adam Optimizing rpwp(rp)", fontsize=20)

    plt.savefig("rpwp_opt.png")

# also plot error history
if RANK == 0:
    f2 = plt.figure(figsize=(12,8), facecolor="w")
    plt.plot(se_history)

    plt.xlabel("Iteration Number", fontsize=16)
    plt.ylabel("Error", fontsize=16)
    plt.title("Error per Iteration", fontsize=20)

    plt.savefig("se_history.png")
