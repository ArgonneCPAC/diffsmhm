import numpy as np
from numba import cuda

from diffsmhm.diff_stats.mpi.types import WprpMPIData
from diffsmhm.diff_stats.cpu.wprp_utils import compute_rr_rrgrad_eff


@cuda.jit(fastmath=False)
def _sum_nomask(w, res, ind):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = w.shape[0]
    tot = 0.0
    for i in range(start, n, stride):
        if w[i] > 0:
            tot += w[i]

    cuda.atomic.add(res, ind, tot)


@cuda.jit(fastmath=False)
def _sum2_nomask(w, res, ind):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = w.shape[0]
    tot = 0.0
    for i in range(start, n, stride):
        if w[i] > 0:
            tot += w[i]*w[i]

    cuda.atomic.add(res, ind, tot)


@cuda.jit(fastmath=False)
def _sum_prod_nomask(w1, w2, res, ind):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = w1.shape[0]
    tot = 0.0
    for i in range(start, n, stride):
        if w1[i] > 0:
            tot += w1[i]*w2[i]

    cuda.atomic.add(res, ind, tot)


@cuda.jit(fastmath=False)
def _sum_prod_at_ind_nomask(w1, w2, res, atind, ind):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = w1.shape[0]
    tot = 0.0
    for i in range(start, n, stride):
        if w1[i] > 0:
            tot += w1[i]*w2[atind, i]

    cuda.atomic.add(res, ind, tot)


@cuda.jit(fastmath=False)
def _sum_at_ind_nomask(w1, w2, res, atind, ind):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = w1.shape[0]
    tot = 0.0
    for i in range(start, n, stride):
        if w1[i] > 0:
            tot += w2[atind, i]

    cuda.atomic.add(res, ind, tot)


@cuda.jit(fastmath=False)
def _count_weighted_pairs_rppi_with_derivs_periodic_cuda(
    x1, y1, z1, w1, dw1, rpbins_squared, n_pi, result, result_grad,
    boxsize, boxsize_2,
):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    n_rp = rpbins_squared.shape[0]

    # this shape is (ngrads, nbins) to attempt to keep things local in memory
    ngrads = dw1.shape[0]

    # the if statements in this kernel are wrong if w1 == 0 but dw1 !=0
    # this happens at two points on the real line for a smooth triweight
    # kernel, so hopefully we can ignore it
    for i in range(start, n1, stride):
        if w1[i] > 0:
            px = x1[i]
            py = y1[i]
            pz = z1[i]
            pw = w1[i]

            for j in range(i+1, n1):
                if w1[j] > 0:
                    qx = x1[j]
                    qy = y1[j]
                    qz = z1[j]
                    qw = w1[j]

                    absdz = abs(pz - qz)
                    if absdz > boxsize_2:
                        absdz = boxsize - absdz

                    if absdz < n_pi:
                        dx = abs(px - qx)
                        if dx > boxsize_2:
                            dx = boxsize - dx
                        dy = abs(py - qy)
                        if dy > boxsize_2:
                            dy = boxsize - dy

                        dsq = dx * dx + dy * dy

                        zbin = int(absdz)

                        wprod = pw * qw
                        k = n_rp - 1
                        while dsq <= rpbins_squared[k]:
                            cuda.atomic.add(
                                result,
                                (k - 1) * n_pi + zbin,
                                wprod,
                            )
                            k = k - 1
                            if k <= 0:
                                break

                        for g in range(ngrads):
                            wprod_grad = (dw1[g, i] * qw) + (pw * dw1[g, j])
                            k = n_rp - 1
                            while dsq <= rpbins_squared[k]:
                                cuda.atomic.add(
                                    result_grad,
                                    g * (n_rp-1) * n_pi + (k - 1) * n_pi + zbin,
                                    wprod_grad,
                                )
                                k = k - 1
                                if k <= 0:
                                    break


def wprp_serial_cuda(
    *,
    x1,
    y1,
    z1,
    w1,
    w1_jac,
    rpbins_squared,
    zmax,
    boxsize,
    threads=32,
    blocks=512,
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
    threads : int, optional
        The # of threads per block. Default is set to 32.
    blocks : int, optional
        The # of blocks on the GPU. Default is set to 512.

    Returns
    -------
    wprp : array-like, shape (n_rpbins,)
        The projected correlation function.
    wprp_grad : array-like, shape (n_grads, n_rpbins)
        The gradients of the projected correlation function.
    """
    assert not np.allclose(rpbins_squared[0], 0)
    _rpbins_squared = np.concatenate([[0], rpbins_squared], axis=0)

    n_grads = w1_jac.shape[0]
    n_rp = _rpbins_squared.shape[0] - 1
    n_pi = int(zmax)

    result = cuda.to_device(np.zeros(n_rp * n_pi, dtype=np.float64))
    result_grad = cuda.to_device(
        np.zeros(n_grads * n_rp * n_pi, dtype=np.float64)
    )
    boxsize_2 = boxsize / 2.0

    _count_weighted_pairs_rppi_with_derivs_periodic_cuda[blocks, threads](
        x1, y1, z1, w1, w1_jac,
        _rpbins_squared,
        n_pi,
        result,
        result_grad,
        boxsize,
        boxsize_2
    )

    res = result.copy_to_host().reshape((n_rp, n_pi))
    res_grad = result_grad.copy_to_host().reshape((n_grads, n_rp, n_pi))

    # TODO - do this with cupy if it is available?
    sums = cuda.to_device(np.zeros(2 + 2*n_grads, dtype=np.float64))
    _sum_nomask[blocks, threads](w1, sums, 0)
    _sum2_nomask[blocks, threads](w1, sums, 1)
    for g in range(n_grads):
        _sum_prod_at_ind_nomask[blocks, threads](
            w1, w1_jac, sums, g, 2+g
        )
        _sum_at_ind_nomask[blocks, threads](
            w1, w1_jac, sums, g, 2+n_grads+g
        )
    sums = sums.copy_to_host()

    # convert to differential
    n_rp = rpbins_squared.shape[0] - 1
    dd = res[1:] - res[:-1]
    dd_grad = res_grad[:, 1:] - res_grad[:, :-1]

    # now do norm by RR and compute proper grad
    # this is the volume of the shell
    dpi = 1.0  # here to make the code clear, always true
    volfac = np.pi * (rpbins_squared[1:] - rpbins_squared[:-1])
    volratio = volfac[:, None] * np.ones(n_pi) * dpi / boxsize ** 3

    # finally get rr and drr
    n_eff = sums[0] ** 2 / sums[1]
    rr, rr_grad = compute_rr_rrgrad_eff(
        sums[0],
        sums[2+n_grads:2+2*n_grads],
        sums[2:2+n_grads],
        n_eff,
        volratio,
    )

    # now produce value and derivs
    xirppi = dd / rr - 1
    xirppi_grad = (
        dd_grad / rr[None, :, :] - dd[None, :, :] / rr[None, :, :] ** 2 * rr_grad
    )

    # integrate over pi
    wprp = 2.0 * dpi * np.sum(xirppi, axis=-1)
    wprp_grad = 2.0 * dpi * np.sum(xirppi_grad, axis=-1)

    return wprp, wprp_grad


@cuda.jit(fastmath=False)
def _sum_mask(w, mask, res, ind):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = w.shape[0]
    tot = 0.0
    for i in range(start, n, stride):
        if w[i] > 0 and mask[i]:
            tot += w[i]

    cuda.atomic.add(res, ind, tot)


@cuda.jit(fastmath=False)
def _sum2_mask(w, mask, res, ind):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = w.shape[0]
    tot = 0.0
    for i in range(start, n, stride):
        if w[i] > 0 and mask[i]:
            tot += w[i]*w[i]

    cuda.atomic.add(res, ind, tot)


@cuda.jit(fastmath=False)
def _sum_prod_mask(w1, w2, mask, res, ind):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = w1.shape[0]
    tot = 0.0
    for i in range(start, n, stride):
        if w1[i] > 0 and mask[i]:
            tot += w1[i]*w2[i]

    cuda.atomic.add(res, ind, tot)


@cuda.jit(fastmath=False)
def _sum_prod_at_ind_mask(w1, w2, mask, res, atind, ind):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = w1.shape[0]
    tot = 0.0
    for i in range(start, n, stride):
        if w1[i] > 0 and mask[i]:
            tot += w1[i]*w2[atind, i]

    cuda.atomic.add(res, ind, tot)


@cuda.jit(fastmath=False)
def _sum_at_ind_mask(w1, w2, mask, res, atind, ind):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = w1.shape[0]
    tot = 0.0
    for i in range(start, n, stride):
        if w1[i] > 0 and mask[i]:
            tot += w2[atind, i]

    cuda.atomic.add(res, ind, tot)


@cuda.jit(fastmath=False)
def _count_weighted_pairs_rppi_with_derivs_cuda(
    x1, y1, z1, w1, dw1, inside_subvol, rpbins_squared, n_pi, result,
    result_grad,
):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n1 = x1.shape[0]
    n_rp = rpbins_squared.shape[0]

    # this shape is (ngrads, nbins) to attempt to keep things local in memory
    ngrads = dw1.shape[0]

    # the if statements in this kernel are wrong if w1 == 0 but dw1 !=0
    # this happens at two points on the real line for a smooth triweight
    # kernel, so hopefully we can ignore it
    for i in range(start, n1, stride):
        if w1[i] > 0 and inside_subvol[i]:
            px = x1[i]
            py = y1[i]
            pz = z1[i]
            pw = w1[i]

            for j in range(n1):
                if w1[j] > 0:
                    qx = x1[j]
                    qy = y1[j]
                    qz = z1[j]
                    qw = w1[j]

                    absdz = abs(pz - qz)

                    if absdz < n_pi:
                        dx = px - qx
                        dy = py - qy

                        dsq = dx * dx + dy * dy

                        zbin = int(absdz)

                        wprod = pw * qw
                        k = n_rp - 1
                        while dsq <= rpbins_squared[k]:
                            cuda.atomic.add(
                                result,
                                (k - 1) * n_pi + zbin,
                                wprod,
                            )
                            k = k - 1
                            if k <= 0:
                                break

                        for g in range(ngrads):
                            wprod_grad = (dw1[g, i] * qw) + (pw * dw1[g, j])
                            k = n_rp - 1
                            while dsq <= rpbins_squared[k]:
                                cuda.atomic.add(
                                    result_grad,
                                    g * (n_rp-1) * n_pi + (k - 1) * n_pi + zbin,
                                    wprod_grad,
                                )
                                k = k - 1
                                if k <= 0:
                                    break


def wprp_mpi_kernel_cuda(
    *,
    x1,
    y1,
    z1,
    w1,
    w1_jac,
    inside_subvol,
    rpbins_squared,
    zmax,
    boxsize,
    threads=32,
    blocks=512,
):
    """The per-process CUDA kernel for MPI-parallel wprp computations.

    Parameters
    ----------
    x1, y1, z1, w1 : array-like, shape (n_pts,)
        The arrays of positions and weights for the first set of points.
    w1_jac : array-lke, shape (n_grads, n_pts,)
        The array of weight gradients for the first set of points.
    inside_subvol : array-like, shape (n_pts,)
        A boolean array with `True` when the point is inside the subvolume
        and `False` otherwise.
    rpbins_squared : array-like, shape (n_rpbins+1,)
        Array of the squared bin edges in the `rp` direction. Note that
        this array is one longer than the number of bins in `rp` direction.
    zmax : float
        The maximum separation in the `pi` (or typically `z`) direction. Output
        bins in `z` direction are unit width, so make sure this is a whole number.
    boxsize : float
        The size of the total periodic volume of the simulation.
    threads : int, optional
        The # of threads per block. Default is set to 32.
    blocks : int, optional
        The # of blocks on the GPU. Default is set to 512.

    Returns
    -------
    wprp_mpi_data : named tuple of type WprpMPIData
        A named tuple of the data needed for the MPI reduction and final summary stats.
    """
    assert not np.allclose(rpbins_squared[0], 0)
    _rpbins_squared = np.concatenate([[0], rpbins_squared], axis=0)

    n_grads = w1_jac.shape[0]
    n_rp = _rpbins_squared.shape[0] - 1
    n_pi = int(zmax)

    result = cuda.to_device(np.zeros(n_rp * n_pi, dtype=np.float64))
    result_grad = cuda.to_device(
        np.zeros(n_grads * n_rp * n_pi, dtype=np.float64)
    )
    _count_weighted_pairs_rppi_with_derivs_cuda[blocks, threads](
        x1, y1, z1, w1, w1_jac, inside_subvol,
        _rpbins_squared, n_pi,
        result, result_grad,
    )
    res = result.copy_to_host().reshape((n_rp, n_pi))
    res_grad = result_grad.copy_to_host().reshape((n_grads, n_rp, n_pi))

    # TODO - do this with cupy if it is available?
    sums = cuda.to_device(np.zeros(2 + 2*n_grads, dtype=np.float64))
    _sum_mask[blocks, threads](w1, inside_subvol, sums, 0)
    _sum2_mask[blocks, threads](w1, inside_subvol, sums, 1)
    for g in range(n_grads):
        _sum_prod_at_ind_mask[blocks, threads](
            w1, w1_jac, inside_subvol, sums, g, 2+g
        )
        _sum_at_ind_mask[blocks, threads](
            w1, w1_jac, inside_subvol, sums, g, 2+n_grads+g
        )
    sums = sums.copy_to_host()

    # convert to differential
    n_rp = rpbins_squared.shape[0] - 1
    res = res[1:] - res[:-1]
    res_grad = res_grad[:, 1:] - res_grad[:, :-1]

    # correct for double counting
    dd = res / 2
    dd_grad = res_grad / 2

    return WprpMPIData(
        dd=dd,
        dd_jac=dd_grad,
        w_tot=np.atleast_1d(sums[0]),
        w2_tot=np.atleast_1d(sums[1]),
        ww_jac_tot=sums[2:2+n_grads],
        w_jac_tot=sums[2+n_grads:2+2*n_grads],
    )
