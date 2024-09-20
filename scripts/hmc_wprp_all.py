import argparse
import h5py

import cupy as cp
import numpy as np

import jax
import jax.numpy as jnp
from jax import custom_vjp

import corner
import pandas as pd
import matplotlib.pyplot as plt

from collections import OrderedDict

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1

from diffsmhm.loader import load_and_chop_data_bolshoi_planck
from diffsmhm.galhalo_models.merging import _calculate_indx_to_deposit

from diffsmhm.diff_stats.cuda.wprp import wprp_mpi_kernel_cuda
from diffsmhm.diff_stats.mpi.wprp import wprp_mpi_comp_and_reduce

from diffsmhm.analysis.diff_sm import compute_weight_and_jac
from diffsmhm.analysis.util import (
    get_default_params,
    get_param_bounds,
    get_param_names
)


# this is what we pure_callback to, returns value and gradient
# again, only rank 0 deals with this
def potential(theta):
    COMM.Bcast([theta, MPI.DOUBLE], root=0)

    w, dw = compute_weight_and_jac(
                logmpeak=halos_jax["logmpeak"],
                loghost_mpeak=halos_jax["loghost_mpeak"],
                log_vmax_by_vmpeak=halos_jax["logvmax_frac"],
                upid=halos_jax["upid"],
                idx_to_deposit=idx_to_deposit,
                mass_bin_low=mass_bin_edges[0],
                mass_bin_high=mass_bin_edges[1],
                theta=jax.device_put(theta, jax.devices()[0])
    )

    w_list = []
    dw_list = []
    for d in range(n_devices):
        cp.cuda.Device(d).use()
        w_list.append(cp.from_dlpack(jax.dlpack.to_dlpack(w, copy=True)))
        dw_list.append(cp.from_dlpack(jax.dlpack.to_dlpack(dw, copy=True)))

    wprp, wprp_grad = wprp_mpi_comp_and_reduce(
                        x1=halos_cp["halo_x"],
                        y1=halos_cp["halo_y"],
                        z1=halos_cp["halo_z"],
                        w1=w_list,
                        w1_jac=dw_list,
                        inside_subvol=halos_cp["_inside_subvol"],
                        rpbins_squared=halos_cp["rpbins_squared"],
                        zmax=zmax,
                        boxsize=box_length,
                        kernel_func=wprp_mpi_kernel_cuda
    )

    error = 0.5 * (np.sum(((wprp - wprp_goal) / wprp_err)**2))
    error_grad = 0.5 * np.sum((2 * wprp_grad * (wprp - wprp_goal)) / (wprp_err**2),
                              axis=1)

    return error, error_grad


@custom_vjp
def get_potential(theta):
    val, grad = jax.pure_callback(
                    potential,
                    (np.array(1.0, dtype=np.float64), np.ones(14, dtype=np.float64)),
                    theta
    )

    return val


def vjp_fwd(theta):
    val, grad = jax.pure_callback(
                    potential,
                    (np.array(1.0, dtype=np.float64), np.ones(14, dtype=np.float64)),
                    theta
    )

    return val, grad


def vjp_bwd(grad, tan):
    # jax expects a tuple here
    return (grad * tan,)


get_potential.defvjp(vjp_fwd, vjp_bwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="hmc_wprp_all.py",
        description="HMC on the full correlation function"
    )
    parser.add_argument(
        "--halo-file",
        type=str,
        default="/home/jwick/data/value_added_orphan_complete_bpl_1.002310.h5"
    )
    parser.add_argument(
        "--particle-file",
        type=str,
        default="/home/jwick/data/hlist_1.00231.particles.halotools_v0p4.hdf5"
    )
    parser.add_argument(
        "-w", "--wprp-file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--mass-bin-low",
        type=float,
        default=10.6
    )
    parser.add_argument(
        "--mass-bin-high",
        type=float,
        default=100.0
    )
    parser.add_argument(
        "-t", "--theta-prior",
        type=str,
        default=None
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default="./"
    )
    parser.add_argument(
        "--hmcut",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--prior-width",
        type=float,
        default=0.05
    )
    parser.add_argument(
        "--hmc-niter",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--hmc-nwarmup",
        type=int,
        default=500
    )
    parser.add_argument(
        "-c", "--hmc-checkpoint",
        type=str,
        default=None
    )
    args = parser.parse_args()

    # 1) setup
    outdir = args.outdir
    if outdir[-1] != "/":
        outdir.append("/")

    theta_default = get_default_params()
    lower_bounds, upper_bounds = get_param_bounds()
    param_names = get_param_names()

    wprp_info = {}
    if RANK == 0:
        # load wprp, error, rpbins
        with h5py.File(args.wprp_file, "r") as f:
            wprp_goal = f["wprp"][...].astype(np.float64)

            rpbins = f["rpbins"][...].astype(np.float64)
            if rpbins[0] != 0:
                rpbins = np.concatenate([np.array([0.0]), rpbins], dtype=np.float64)

            # optional for purposes of the demo, really you should provide this
            if "wprp_error" not in f.keys():
                wprp_err = 0.1 * wprp_goal
            else:
                wprp_err = f["wprp_error"][...].astype(np.float64)

        assert len(wprp_goal) == len(rpbins) - 2
        wprp_info = {
                        "wprp": wprp_goal,
                        "wprp_err": wprp_err,
                        "rpbins": rpbins
        }

        # load hmc info if a checkpoint is provided
        do_hmc_adapt = args.hmc_checkpoint is None
        hmc_ss = None
        hmc_imm = None  # will be a dictionary
        hmc_key = jax.random.PRNGKey(42)
        if not do_hmc_adapt:
            hmc_imm = {}
            hmc_init_pos = {}
            with h5py.File(args.hmc_checkpoint, "r") as f:
                hmc_ss = f["step_size"][...]

                # PRNGKey is how the docs suggest to resume chains
                hmc_key = jax.numpy.array([
                            f["checkpoint_prng/0"][...],
                            f["checkpoint_prng/1"][...]], dtype=np.uint32
                )

                # the format for this is apparently a tuple of names as the key
                # and a single array of values
                params = tuple(param_names)
                vallist = []
                for n in param_names:
                    retrieval_str = "inverse_mass_matrix/"+n
                    vallist.append(f[retrieval_str][...])
            hmc_imm = {params: np.array(vallist, dtype=np.float64)}

    wprp_info = COMM.bcast(wprp_info, root=0)
    wprp_goal = wprp_info["wprp"]
    wprp_err = wprp_info["wprp_err"]
    rpbins = wprp_info["rpbins"]

    # load bolshoi data
    box_length = 250.0  # Mpc
    buff_wprp = max(rpbins) + 1  # Mpc
    zmax = 20.0

    # jax weights prefer this as an array
    mass_bin_edges = np.array([args.mass_bin_low, args.mass_bin_high], dtype=np.float64)

    theta_init = np.copy(theta_default)
    if args.theta_prior is not None:
        with h5py.File(args.theta_prior, "r") as f:
            theta_init = f["theta"][...].astype(np.float64)
    if RANK == 0:
        print("theta prior:", theta_init, flush=True)

    n_params = len(theta_init)
    n_rpbins = len(rpbins) - 2
    # note this works with nvidia devices, use dpnp for intel gpu count
    n_devices = jax.local_device_count()

    hmcut = args.hmcut
    halos, _ = load_and_chop_data_bolshoi_planck(
                args.particle_file,
                args.halo_file,
                box_length,
                buff_wprp,
                host_mpeak_cut=hmcut
    )

    idx_to_deposit = _calculate_indx_to_deposit(halos["upid"], halos["halo_id"])

    # jax copy (weights_ and cupy copies (wprp) for each GPU
    halos_jax = OrderedDict()
    halos_cp = OrderedDict()
    for k in halos.keys():
        halos_jax[k] = jnp.array(halos[k], dtype=halos[k].dtype)

        halos_cp[k] = []
        for d in range(n_devices):
            cp.cuda.Device(d).use()
            halos_cp[k].append(cp.array(halos[k], dtype=halos[k].dtype))

    halos_cp["rpbins_squared"] = []
    for d in range(n_devices):
        cp.cuda.Device(d).use()
        halos_cp["rpbins_squared"].append(cp.array(rpbins**2, dtype=cp.float64))

    # 2) HMC
    # defined inside main so we can use the default argument
    def model(
            theta_init=theta_init,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds
    ):
        # priors on parameters
        sigmas = (upper_bounds - lower_bounds) * args.prior_width

        smhm_0_dist = dist.Normal(theta_init[0], sigmas[0])
        smhm_0_dist.support = dist.constraints.interval(lower_bounds[0],
                                                        upper_bounds[0])
        smhm_0 = numpyro.sample(param_names[0], smhm_0_dist)

        smhm_1_dist = dist.Normal(theta_init[1], sigmas[1])
        smhm_1_dist.support = dist.constraints.interval(lower_bounds[1],
                                                        upper_bounds[1])
        smhm_1 = numpyro.sample(param_names[1], smhm_1_dist)

        smhm_2_dist = dist.Normal(theta_init[2], sigmas[2])
        smhm_2_dist.support = dist.constraints.interval(lower_bounds[2],
                                                        upper_bounds[2])
        smhm_2 = numpyro.sample(param_names[2], smhm_2_dist)

        smhm_3_dist = dist.Normal(theta_init[3], sigmas[3])
        smhm_3_dist.support = dist.constraints.interval(lower_bounds[3],
                                                        upper_bounds[3])
        smhm_3 = numpyro.sample(param_names[3], smhm_3_dist)

        smhm_4_dist = dist.Normal(theta_init[4], sigmas[4])
        smhm_4_dist.support = dist.constraints.interval(lower_bounds[4],
                                                        upper_bounds[4])
        smhm_4 = numpyro.sample(param_names[4], smhm_4_dist)

        smhm_sigma_0_dist = dist.Normal(theta_init[5], sigmas[5])
        smhm_sigma_0_dist.support = dist.constraints.interval(
                                        lower_bounds[5], upper_bounds[5])
        smhm_sigma_0 = numpyro.sample(param_names[5], smhm_sigma_0_dist)

        smhm_sigma_1_dist = dist.Normal(theta_init[6], sigmas[6])
        smhm_sigma_1_dist.support = dist.constraints.interval(
                                        lower_bounds[6], upper_bounds[6])
        smhm_sigma_1 = numpyro.sample(param_names[6], smhm_sigma_1_dist)

        smhm_sigma_2_dist = dist.Normal(theta_init[7], sigmas[7])
        smhm_sigma_2_dist.support = dist.constraints.interval(
                                        lower_bounds[7], upper_bounds[7])
        smhm_sigma_2 = numpyro.sample(param_names[7], smhm_sigma_2_dist)

        smhm_sigma_3_dist = dist.Normal(theta_init[8], sigmas[8])
        smhm_sigma_3_dist.support = dist.constraints.interval(
                                        lower_bounds[8], upper_bounds[8])
        smhm_sigma_3 = numpyro.sample(param_names[8], smhm_sigma_3_dist)

        disruption_0_dist = dist.Normal(theta_init[9], sigmas[9])
        disruption_0_dist.support = dist.constraints.interval(
                                        lower_bounds[9], upper_bounds[9])
        disruption_0 = numpyro.sample(param_names[9], disruption_0_dist)

        disruption_1_dist = dist.Normal(theta_init[10], sigmas[10])
        disruption_1_dist.support = dist.constraints.interval(
                                        lower_bounds[10], upper_bounds[10])
        disruption_1 = numpyro.sample(param_names[10], disruption_1_dist)

        disruption_2_dist = dist.Normal(theta_init[11], sigmas[11])
        disruption_2_dist.support = dist.constraints.interval(
                                        lower_bounds[11], upper_bounds[11])
        disruption_2 = numpyro.sample(param_names[11], disruption_2_dist)

        disruption_3_dist = dist.Normal(theta_init[12], sigmas[12])
        disruption_3_dist.support = dist.constraints.interval(
                                        lower_bounds[12], upper_bounds[12])
        disruption_3 = numpyro.sample(param_names[12], disruption_3_dist)

        disruption_4_dist = dist.Normal(theta_init[13], sigmas[13])
        disruption_4_dist.support = dist.constraints.interval(
                                        lower_bounds[13], upper_bounds[13])
        disruption_4 = numpyro.sample(param_names[13], disruption_4_dist)

        theta = jnp.array([
                    smhm_0, smhm_1, smhm_2, smhm_3, smhm_4,
                    smhm_sigma_0, smhm_sigma_1, smhm_sigma_2, smhm_sigma_3,
                    disruption_0, disruption_1, disruption_2, disruption_3, disruption_4
        ], dtype=jnp.float64)

        U = get_potential(theta)
        numpyro.factor("log_prob", -1.0 * U)

    # here we split the ranks, rank 0 does HMC, others just a compute loop
    if RANK > 0:
        while True:
            theta = np.empty(n_params, dtype=np.float64)
            COMM.Bcast([theta, MPI.DOUBLE], root=0)

            if theta[0] < 0:
                break

            w, dw = compute_weight_and_jac(
                        logmpeak=halos_jax["logmpeak"],
                        loghost_mpeak=halos_jax["loghost_mpeak"],
                        log_vmax_by_vmpeak=halos_jax["logvmax_frac"],
                        upid=halos_jax["upid"],
                        idx_to_deposit=idx_to_deposit,
                        mass_bin_low=mass_bin_edges[0],
                        mass_bin_high=mass_bin_edges[1],
                        theta=jax.device_put(theta, jax.devices()[0])
            )

            w_list = []
            dw_list = []
            for d in range(n_devices):
                cp.cuda.Device(d).use()
                w_list.append(cp.from_dlpack(jax.dlpack.to_dlpack(w, copy=True)))
                dw_list.append(cp.from_dlpack(jax.dlpack.to_dlpack(dw, copy=True)))

            _, _ = wprp_mpi_comp_and_reduce(
                    x1=halos_cp["halo_x"],
                    y1=halos_cp["halo_y"],
                    z1=halos_cp["halo_z"],
                    w1=w_list,
                    w1_jac=dw_list,
                    inside_subvol=halos_cp["_inside_subvol"],
                    rpbins_squared=halos_cp["rpbins_squared"],
                    zmax=zmax,
                    boxsize=box_length,
                    kernel_func=wprp_mpi_kernel_cuda
            )
    else:
        num_warmup = args.hmc_nwarmup
        num_samples = args.hmc_niter
        if hmc_ss is not None:
            nuts_kernel = NUTS(model, step_size=hmc_ss, inverse_mass_matrix=hmc_imm,
                               adapt_mass_matrix=do_hmc_adapt,
                               adapt_step_size=do_hmc_adapt)
        else:
            nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=1)
        mcmc.run(hmc_key)
        mcmc.print_summary()

        wu_stepsize = mcmc.last_state.adapt_state.step_size
        wu_imm = mcmc.last_state.adapt_state.inverse_mass_matrix
        print("III:", wu_imm, flush=True)
        imm_keys = [*wu_imm][0]
        imm_vals = [*wu_imm.values()][0]
        final_state = mcmc.last_state.rng_key

        # we're done with HMC, broadcast the stop condition
        stop = -1 * np.ones_like(theta_init)
        COMM.Bcast(stop, root=0)

        # let's export
        fpath_positions = outdir+"positions.csv"
        mcmc_positions = mcmc.get_samples()
        positions_df = pd.DataFrame.from_dict(mcmc_positions)
        if do_hmc_adapt:
            positions_df.to_csv(fpath_positions, mode="w", header=True, index=False)
        else:
            positions_df.to_csv(fpath_positions, mode="a", header=False, index=False)

        # hdf5 for checkpointing
        last_pos = positions_df.iloc[-1]
        fpath_checkpoint = outdir+"checkpoint_hmc.hdf5"
        with h5py.File(fpath_checkpoint, "w") as f:
            # save chain prng state
            grp_pos = f.create_group("checkpoint_prng")
            grp_pos.create_dataset("0", data=final_state[0], dtype=np.uint32)
            grp_pos.create_dataset("1", data=final_state[1], dtype=np.uint32)

            # save warmup info
            f.create_dataset("step_size", data=wu_stepsize, dtype="f")
            grp_imm = f.create_group("inverse_mass_matrix")
            for i, k in enumerate(imm_keys):
                grp_imm.create_dataset(k, data=imm_vals[i], dtype="f")

        # and make a figure
        # less specific labels bc the full names are a bit too long for the figure
        corner_labels = [
            "smhm_0", "smhm_1", "smhm_2", "smhm_3", "smhm_4",
            "sigma_0", "sigma_1", "sigma_2", "sigma_3",
            "merge_0", "merge_1", "merge_2", "merge_3", "merge_4"
        ]

        # reload the csv of positions so that we have any prior runs too
        mcmc_positions = pd.read_csv(fpath_positions)
        positions_np = np.vstack([
                        mcmc_positions[param_names[0]].to_numpy(),
                        mcmc_positions[param_names[1]].to_numpy(),
                        mcmc_positions[param_names[2]].to_numpy(),
                        mcmc_positions[param_names[3]].to_numpy(),
                        mcmc_positions[param_names[4]].to_numpy(),

                        mcmc_positions[param_names[5]].to_numpy(),
                        mcmc_positions[param_names[6]].to_numpy(),
                        mcmc_positions[param_names[7]].to_numpy(),
                        mcmc_positions[param_names[8]].to_numpy(),

                        mcmc_positions[param_names[9]].to_numpy(),
                        mcmc_positions[param_names[10]].to_numpy(),
                        mcmc_positions[param_names[11]].to_numpy(),
                        mcmc_positions[param_names[12]].to_numpy(),
                        mcmc_positions[param_names[13]].to_numpy()
        ]).T

        fig = corner.corner(
                positions_np,
                labels=corner_labels,
                show_titles=True,

                levels=(1-np.exp(-1), 1-np.exp(-2)),
                smooth=1.0,
                color="C2",

                plot_density=False,
                plot_datapoints=False,
                fill_contours=True
        )
        plt.savefig(outdir+"corner_hmc.png")
