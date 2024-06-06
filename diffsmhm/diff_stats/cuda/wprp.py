import numpy as np
import cupy as cp

from numba import cuda

from diffsmhm.diff_stats.mpi.types import WprpMPIData
from diffsmhm.diff_stats.cpu.wprp_utils import compute_rr_rrgrad_eff


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
def _count_weighted_pairs_rppi_with_derivs_periodic_cuda(
    x1, y1, z1, w1, dw1, rpbins_squared, n_pi, result, result_grad,
    boxsize, boxsize_2
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
    # check if cupy is available
    xp = cp.get_array_module(x1)
    can_cupy = xp is cp

    assert not xp.allclose(rpbins_squared[0], 0)
    _rpbins_squared = xp.concatenate([xp.array([0]), rpbins_squared], axis=0)

    n_grads = w1_jac.shape[0]
    n_rp = _rpbins_squared.shape[0] - 1
    n_pi = int(zmax)

    if can_cupy:
        result = xp.zeros(n_rp * n_pi, dtype=np.float64)
        result_grad = xp.zeros(n_grads * n_rp * n_pi, dtype=xp.float64)
    else:
        result = cuda.to_device(xp.zeros(n_rp * n_pi, dtype=xp.float64))
        result_grad = cuda.to_device(
            xp.zeros(n_grads * n_rp * n_pi, dtype=xp.float64)
        )

    boxsize_2 = boxsize / 2.0

    _count_weighted_pairs_rppi_with_derivs_periodic_cuda[blocks, threads](
        x1, y1, z1, w1, w1_jac,
        _rpbins_squared,
        n_pi,
        result,
        result_grad,
        boxsize,
        boxsize_2,
    )

    if can_cupy:
        res = result.reshape((n_rp, n_pi))
        res_grad = result_grad.reshape((n_grads, n_rp, n_pi))
    else:
        res = result.copy_to_host().reshape((n_rp, n_pi))
        res_grad = result_grad.copy_to_host().reshape((n_grads, n_rp, n_pi))

    sums = xp.zeros(2 + 2*n_grads, dtype=xp.float64)
    sums[0] = xp.sum(w1)
    sums[1] = xp.sum(w1**2)
    for g in range(n_grads):
        sums[2+g] = xp.sum(w1 * w1_jac[g, :])
        sums[2+n_grads+g] = xp.sum(w1_jac[g, :])

    # let's bring everything back to cpu/numpy
    if can_cupy:
        res = np.array(res.get())
        res_grad = np.array(res_grad.get())
        rpbins_squared = np.array(rpbins_squared.get())
        sums = np.array(sums.get())

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


# outer loop can be limited in scope with start_idx end_idx which allows
# for the wprp computation to be split
@cuda.jit(fastmath=False)
def _count_weighted_pairs_rppi_with_derivs_cuda(
    x1, y1, z1, w1, dw1, inside_subvol, rpbins_squared, n_pi,
    result, result_grad,
    start_idx, end_idx
):
    # start position depends on thread position and designated chunk
    start = start_idx + cuda.grid(1)
    stride = cuda.gridsize(1)

    # end position for designated chunk (outer loop)
    n1 = end_idx
    # end is all objects for inner loop
    n2 = x1.shape[0]

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

            for j in range(n2):
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
        The arrays of positions and weights for the first set of points. If using
        multiple GPUs per rank these should be lists of cupy arrays that correspond
        to the data on each device.
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
    # check if cupy is available
    # this is here bc github CI doesn't work with cupy; no cupy cpu simulator
    xp = cp.get_array_module(x1[0])
    can_cupy = xp is cp

    assert not xp.allclose(rpbins_squared[0], 0)
    _rpbins_squared = []
    for d, _ in enumerate(rpbins_squared):
        if can_cupy:
            cp.cuda.Device(d).use()
        _rpbins_squared.append(xp.concatenate([xp.array([0]),
                                               rpbins_squared[d]], axis=0))

    n_grads = w1_jac[0].shape[0]
    n_rp = _rpbins_squared[0].shape[0] - 1
    n_pi = int(zmax)

    # use multiple GPUs if available
    n_devices = 1
    if can_cupy:
        n_devices = len(cuda.gpus)

    # split the data into chunks for each device
    avg, rem = divmod(len(x1[0]), n_devices)
    device_count = [avg + 1 if p < rem else avg for p in range(n_devices)]
    device_displ = [sum(device_count[:p]) for p in range(n_devices)]
    # lists of results for combination later
    result_all = []
    result_grad_all = []
    for d in range(n_devices):
        # get this chunk range
        count = device_count[d]
        displ = device_displ[d]
        start_idx = displ
        end_idx = displ + count

        # data should already be copied to relevant gpus
        if can_cupy:
            cp.cuda.Device(d).use()

        result_d = xp.zeros(n_rp * n_pi, dtype=np.float64)
        result_grad_d = xp.zeros(n_grads * n_rp * n_pi, dtype=xp.float64)

        # launch kernel
        _count_weighted_pairs_rppi_with_derivs_cuda[blocks, threads](
            x1[d], y1[d], z1[d], w1[d], w1_jac[d], inside_subvol[d],
            _rpbins_squared[d], n_pi,
            result_d, result_grad_d,
            start_idx, end_idx
        )

        # add chunked result to list
        result_all.append(result_d)
        result_grad_all.append(result_grad_d)

    # now add the distributed calculation
    if can_cupy:
        cp.cuda.Device(0).use()
    res = xp.copy(result_all[0])
    res_grad = xp.copy(result_grad_all[0])
    for d in range(1, n_devices):
        # let's be explicit about moving data to one device
        rd = xp.copy(result_all[d])
        rd_grad = xp.copy(result_grad_all[d])

        res += rd
        res_grad += rd_grad

    res = xp.reshape(res, (n_rp, n_pi))
    res_grad = xp.reshape(res_grad, (n_grads, n_rp, n_pi))

    sums = xp.zeros(2 + 2*n_grads, dtype=xp.float64)
    sums[0] = xp.sum(w1[0][inside_subvol[0]])
    sums[1] = xp.sum(w1[0][inside_subvol[0]]**2)
    for g in range(n_grads):
        sums[2+g] = xp.sum(w1[0][inside_subvol[0]] * w1_jac[0][g, inside_subvol[0]])
        sums[2+n_grads+g] = xp.sum(w1_jac[0][g, inside_subvol[0]])

    # convert to differential
    n_rp = rpbins_squared[0].shape[0] - 1
    res = res[1:] - res[:-1]
    res_grad = res_grad[:, 1:] - res_grad[:, :-1]

    # correct for double counting
    dd = res / 2
    dd_grad = res_grad / 2

    # return as cpu data
    if can_cupy:
        sums_np = cp.asnumpy(sums)
        return WprpMPIData(
            dd=cp.asnumpy(dd),
            dd_jac=cp.asnumpy(dd_grad),
            w_tot=np.atleast_1d(sums_np[0]),
            w2_tot=np.atleast_1d(sums_np[1]),
            ww_jac_tot=sums_np[2:2+n_grads],
            w_jac_tot=sums_np[2+n_grads:2+2*n_grads],
            rpbins_squared=cp.asnumpy(rpbins_squared[0])
        )
    else:
        return WprpMPIData(
            dd=dd,
            dd_jac=dd_grad,
            w_tot=np.atleast_1d(sums[0]),
            w2_tot=np.atleast_1d(sums[1]),
            ww_jac_tot=sums[2:2+n_grads],
            w_jac_tot=sums[2+n_grads:2+2*n_grads],
            rpbins_squared=rpbins_squared[0]
        )
