import numpy as np
import numba


@numba.njit(fastmath=True, parallel=True)
def compute_rr_rrgrad(w, dw, volratio):
    w_tot = np.sum(w)
    w2_tot = np.sum(w ** 2)
    n_eff = w_tot ** 2 / w2_tot
    dw_tot = np.sum(dw, axis=1)
    wdw_tot = np.sum(dw * w, axis=1)

    # finally get rr and drr
    return compute_rr_rrgrad_eff(w_tot, dw_tot, wdw_tot, n_eff, volratio)


@numba.njit(fastmath=True)
def compute_rr_rrgrad_eff(w_tot, dw_tot, wdw_tot, n_eff, volratio):
    dw_tot = np.reshape(dw_tot, (-1, 1, 1))
    wdw_tot = np.reshape(wdw_tot, (-1, 1, 1))
    rr = w_tot ** 2 * (1 - 1.0 / n_eff) * volratio
    rr_grad = (
        2
        * (w_tot * dw_tot - wdw_tot)
        * volratio
    )
    return rr, rr_grad
