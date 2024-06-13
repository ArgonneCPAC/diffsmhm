import os

import numpy as np
import Corrfunc
import psutil

from diffsmhm.diff_stats.mpi.types import WprpMPIData
from .wprp_utils import compute_rr_rrgrad


def wprp_serial_cpu(
    *,
    x1,
    y1,
    z1,
    w1,
    w1_jac,
    rpbins_squared,
    zmax,
    boxsize,
):
    """Compute wp(rp) w/ derivs for a periodic volume.

    Parameters
    ----------
    x1, y1, z1, w1 : array-like, shape (n_pts,)
        The arrays of positions and weights for the first set of points.
    w1_jac : array-lke, shape (n_grads, n_pts,)
        The array of weight gradients for the first set of points.
    rpbins_squared : array-like, shape (n_rpbins+1,)
        Array of the squared bin edges in the `rp` direction. Note that
        this array is one longer than the number of bins in `rp` direction.
    zmax : float
        The maximum separation in the `pi` (or typically `z`) direction. Output
        bins in `z` direction are unit width, so make sure this is a whole number.
    boxsize : float
        The size of the total periodic volume of the simulation.

    Returns
    -------
    wprp : array-like, shape (n_rpbins,)
        The projected correlation function.
    wprp_grad : array-like, shape (n_grads, n_rpbins)
        The gradients of the projected correlation function.
    """

    # check for first rpbin being zero for consistency with mpi kernels
    assert np.allclose(rpbins_squared[0], 0)

    n_grads = w1_jac.shape[0]
    n_rp = rpbins_squared.shape[0] - 1
    n_pi = int(zmax)

    # dd
    res = Corrfunc.theory.DDrppi(
        1,
        int(os.environ.get("OMP_NUM_THREADS", psutil.cpu_count(logical=False))),
        zmax,
        np.sqrt(rpbins_squared),
        x1,
        y1,
        z1,
        weights1=w1,
        periodic=True,
        boxsize=boxsize,
        weight_type="pair_product",
    )
    dd = (
        res["weightavg"].reshape((n_rp, n_pi)) * res["npairs"].reshape((n_rp, n_pi)) / 2
    )

    # now do the grad terms
    dd_grad = np.zeros((n_grads, n_rp, n_pi))
    for g in range(n_grads):
        res = Corrfunc.theory.DDrppi(
            0,
            int(os.environ.get("OMP_NUM_THREADS", psutil.cpu_count(logical=False))),
            zmax,
            np.sqrt(rpbins_squared),
            x1,
            y1,
            z1,
            weights1=w1,
            X2=x1,
            Y2=y1,
            Z2=z1,
            weights2=w1_jac[g, :],
            periodic=True,
            boxsize=boxsize,
            weight_type="pair_product",
        )
        dd_grad[g, :, :] += (
            res["weightavg"].reshape((n_rp, n_pi))
            * res["npairs"].reshape((n_rp, n_pi))
            / 2
        )

        res = Corrfunc.theory.DDrppi(
            False,
            int(os.environ.get("OMP_NUM_THREADS", psutil.cpu_count(logical=False))),
            zmax,
            np.sqrt(rpbins_squared),
            x1,
            y1,
            z1,
            weights1=w1_jac[g, :],
            X2=x1,
            Y2=y1,
            Z2=z1,
            weights2=w1,
            periodic=True,
            boxsize=boxsize,
            weight_type="pair_product",
        )
        dd_grad[g, :, :] += (
            res["weightavg"].reshape((n_rp, n_pi))
            * res["npairs"].reshape((n_rp, n_pi))
            / 2
        )

    # now do norm by RR and compute proper grad

    # this is the volume of the shell
    dpi = 1.0  # here to make the code clear, always true
    volfac = np.pi * (rpbins_squared[2:] - rpbins_squared[:-1])
    volratio = volfac[:, None] * np.ones(n_pi) * dpi / boxsize ** 3

    # finally get rr and drr
    rr, rr_grad = compute_rr_rrgrad(w1, w1_jac, volratio)

    # now produce value and derivs
    xirppi = dd / rr - 1
    xirppi_grad = (
        dd_grad / rr[None, :, :] - dd[None, :, :] / rr[None, :, :] ** 2 * rr_grad
    )

    # integrate over pi
    wprp = 2.0 * dpi * np.sum(xirppi, axis=-1)
    wprp_grad = 2.0 * dpi * np.sum(xirppi_grad, axis=-1)

    return wprp, wprp_grad


def wprp_mpi_kernel_cpu(
    *, x1, y1, z1, w1, w1_jac, inside_subvol, rpbins_squared, zmax, boxsize,
):
    """The per-process CPU kernel for MPI-parallel wprp computations.

    Parameters
    ----------
    x1, y1, z1, w1 : [array-like], shape [(n_pts,), (n_pts,), ...]
        The lists of arrays of positions and weights for the first set of points.
        As with all lists in this method, arrays in each respective list will be
        concatenated for calculation.
    w1_jac : [array-lke], shape [(n_grads, n_pts,), ...]
        The list of array of weight gradients for the first set of points.
    inside_subvol : [array-like], shape [(n_pts,), ...]
        The list of boolean arrays with `True` when the point is inside the subvolume
        and `False` otherwise.
    rpbins_squared : [array-like], shape [(n_rpbins+1,), ...]
        The list of arrays of the squared bin edges in the `rp` direction. Note that
        these arrays are one longer than the number of bins in `rp` direction. Also
        note that if multiple arrays are passed in this list they must be identical.
    zmax : float
        The maximum separation in the `pi` (or typically `z`) direction. Output
        bins in `z` direction are unit width, so make sure this is a whole number.
    boxsize : float
        The size of the total periodic volume of the simulation.

    Returns
    -------
    wprp_mpi_data : named tuple of type WprpMPIData
        A named tuple of the data needed for the MPI reduction and final summary stats.
    """
    # assert that all rpbins arrays are the same
    for i, _ in enumerate(rpbins_squared[:-1]):
        assert np.allclose(rpbins_squared[i], rpbins_squared[i+1])

    # concatenate input arrays
    x1_all = np.concatenate(x1)
    y1_all = np.concatenate(y1)
    z1_all = np.concatenate(z1)
    w1_all = np.concatenate(w1)
    w1_jac_all = np.concatenate(w1_jac, axis=1)
    inside_subvol_all = np.concatenate(inside_subvol)

    n_grads = w1_jac_all.shape[0]
    n_rp = rpbins_squared[0].shape[0] - 1
    n_pi = int(zmax)

    # dd
    res = Corrfunc.theory.DDrppi(
        False,
        int(os.environ.get("OMP_NUM_THREADS", psutil.cpu_count(logical=False))),
        zmax,
        np.sqrt(rpbins_squared[0]),
        x1_all[inside_subvol_all],
        y1_all[inside_subvol_all],
        z1_all[inside_subvol_all],
        weights1=w1_all[inside_subvol_all],
        X2=x1_all,
        Y2=y1_all,
        Z2=z1_all,
        weights2=w1_all,
        periodic=False,
        weight_type="pair_product",
    )
    _dd = (
        res["weightavg"].reshape((n_rp, n_pi)) * res["npairs"].reshape((n_rp, n_pi))
    ).astype(np.float64)

    # now do the grad terms
    _dd_grad = np.zeros((n_grads, n_rp, n_pi), dtype=np.float64)
    for g in range(n_grads):
        res = Corrfunc.theory.DDrppi(
            False,
            int(os.environ.get("OMP_NUM_THREADS", psutil.cpu_count(logical=False))),
            zmax,
            np.sqrt(rpbins_squared[0]),
            x1_all[inside_subvol_all],
            y1_all[inside_subvol_all],
            z1_all[inside_subvol_all],
            weights1=w1_all[inside_subvol_all],
            X2=x1_all,
            Y2=y1_all,
            Z2=z1_all,
            weights2=w1_jac_all[g, :],
            periodic=False,
            weight_type="pair_product",
        )
        _dd_grad[g, :, :] += (
            res["weightavg"].reshape((n_rp, n_pi))
            * res["npairs"].reshape((n_rp, n_pi))
        )
        res = Corrfunc.theory.DDrppi(
            False,
            int(os.environ.get("OMP_NUM_THREADS", psutil.cpu_count(logical=False))),
            zmax,
            np.sqrt(rpbins_squared[0]),
            x1_all[inside_subvol_all],
            y1_all[inside_subvol_all],
            z1_all[inside_subvol_all],
            weights1=w1_jac_all[g, inside_subvol_all],
            X2=x1_all,
            Y2=y1_all,
            Z2=z1_all,
            weights2=w1_all,
            periodic=False,
            weight_type="pair_product",
        )
        _dd_grad[g, :, :] += (
            res["weightavg"].reshape((n_rp, n_pi))
            * res["npairs"].reshape((n_rp, n_pi))
        )

    # now do reductions
    _w_tot = np.atleast_1d(np.sum(w1_all[inside_subvol_all]))
    _w2_tot = np.atleast_1d(np.sum(w1_all[inside_subvol_all]**2))
    _wdw_tot = np.sum(w1_jac_all[:, inside_subvol_all] * w1_all[inside_subvol_all],
                      axis=1)
    _dw_tot = np.sum(w1_jac_all[:, inside_subvol_all], axis=1)

    return WprpMPIData(
        dd=_dd / 2.0,
        dd_jac=_dd_grad / 2.0,
        w_tot=_w_tot,
        w2_tot=_w2_tot,
        ww_jac_tot=_wdw_tot,
        w_jac_tot=_dw_tot,
        rpbins_squared=rpbins_squared[0]
    )
