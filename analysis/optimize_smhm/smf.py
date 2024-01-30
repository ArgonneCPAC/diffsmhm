import numpy as np
from jax.numpy import histogram as jnp_hist

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1


from diffsmhm.diff_stats.cuda.tw_kernels import (
    tw_kern_mstar_bin_weights_and_derivs_cuda
)
from diff_sm import (
    compute_sm_and_jac,
    compute_sm_sigma_and_jac
)

def smf_mpi_comp_and_reduce(
    hm,
    host_hm,
    log_vmax_by_vmpeak,
    upid,
    idx_to_deposit,
    inside_subvol,
    bin_edges,
    theta,
    threads=32,
    blocks=512
):
    # not using `compute_weight_and_jac` bc only need to calculate sm once
    sm, sm_jac = compute_sm_and_jac(
                                hm,
                                host_hm,
                                log_vmax_by_vmpeak,
                                upid,
                                idx_to_deposit,
                                theta
    )
    sm_sigma, sm_sigma_jac = compute_sm_sigma_and_jac(hm, theta)

    sm = sm[inside_subvol]
    sm_jac = sm_jac[:,inside_subvol]
    sm_sigma = sm_sigma[inside_subvol]
    sm_sigma_jac = sm_sigma_jac[:, inside_subvol]
    
    n_bins = len(bin_edges)-1
    n_params = len(theta)
    n_halos = len(hm)

    hist = np.zeros(n_bins, dtype=np.float64)
    hist_grad = np.zeros((n_params, n_bins), dtype=np.float64)
    for i in range(n_bins):
        w = np.zeros(n_halos, dtype=np.float64)
        dw = np.zeros((n_params, n_halos), dtype=np.float64)

        tw_kern_mstar_bin_weights_and_derivs_cuda[blocks, threads](
                                            sm,
                                            sm_jac,
                                            sm_sigma,
                                            sm_sigma_jac,
                                            bin_edges[i], bin_edges[i+1],
                                            w,
                                            dw
        )

        hist[i] = np.sum(w)
        hist_grad[:, i] = np.sum(dw, axis=1)

    # reduce
    hist_red = np.zeros_like(hist)
    COMM.Reduce(hist, hist_red, op=MPI.SUM, root=0)

    hist_grad_red = np.zeros_like(hist_grad)
    COMM.Reduce(hist_grad, hist_grad_red, op=MPI.SUM, root=0)    

    if RANK == 0:
        return hist_red, hist_grad_red
    else:
        return None, None

