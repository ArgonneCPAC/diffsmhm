import numpy as np
import cupy as cp

from numba import cuda

import math

from diffsmhm.diff_stats.mpi.types import SigmaMPIData


def _copy_periodic_points_3D(x, y, z, boxsize, buffer_length):
    # copy particles within buffer_length of an edge in XY
    n_points = len(x)

    # setup buffer point arrays
    x_copied_points = []
    y_copied_points = []
    z_copied_points = []

    for p in range(n_points):
        # if not near edge continue
        if (
            not (x[p] < buffer_length or x[p] > boxsize-buffer_length) and
            not (y[p] < buffer_length or y[p] > boxsize-buffer_length) and
            not (z[p] < buffer_length or z[p] > boxsize-buffer_length)
        ):
            continue

        # 6 edges

        # x low edge
        if x[p] < buffer_length:
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p])
            z_copied_points.append(z[p])
        # x high edge
        elif x[p] > boxsize-buffer_length:
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p])
            z_copied_points.append(z[p])

        # y low edge
        if y[p] < buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p])
        # y high edge
        elif y[p] > boxsize-buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p])

        # z low edge
        if z[p] < buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p])
            z_copied_points.append(z[p] + boxsize)
        # z high edge
        elif z[p] > boxsize-buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p])
            z_copied_points.append(z[p] - boxsize)

        # 8 corners

        # "bottom" corners (z low)
        # x low, y low
        if (
                x[p] < buffer_length and
                y[p] < buffer_length and
                z[p] < buffer_length
        ):
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p] + boxsize)
        # x low, y high
        elif (
                x[p] < buffer_length and
                y[p] > boxsize-buffer_length and
                z[p] < buffer_length
        ):
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p] + boxsize)
        # x high, y low
        elif (
                x[p] > boxsize-buffer_length and
                y[p] < buffer_length and
                z[p] < buffer_length
        ):
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p] + boxsize)
        # x high, y high
        elif (
                x[p] > boxsize-buffer_length and
                y[p] > boxsize-buffer_length and
                z[p] < buffer_length
        ):
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p] + boxsize)

        # "top" corners (z high)
        # x low, y low
        elif (
                x[p] < buffer_length and
                y[p] < buffer_length and
                z[p] > boxsize-buffer_length
        ):
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p] - boxsize)
        # x low, y high
        elif (
                x[p] < buffer_length and
                y[p] > boxsize-buffer_length and
                z[p] > boxsize-buffer_length
        ):
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p] - boxsize)
        # x high, y low
        elif (
                x[p] > boxsize-buffer_length and
                y[p] < buffer_length and
                z[p] > boxsize-buffer_length
        ):
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p] - boxsize)
        # x high, y high
        elif (
            x[p] > boxsize-buffer_length and
            y[p] > boxsize-buffer_length and
            z[p] > boxsize-buffer_length
        ):
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p] - boxsize)

        # 12 egdges

        # x low, y low
        if x[p] < buffer_length and y[p] < buffer_length:
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p])
        # x high, y low
        if x[p] > boxsize-buffer_length and y[p] < buffer_length:
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p])
        # x low, y high
        if x[p] < buffer_length and y[p] > boxsize-buffer_length:
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p])
        # x high, y high
        if x[p] > boxsize-buffer_length and y[p] > boxsize-buffer_length:
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p])

        # y low, z low
        if y[p] < buffer_length and z[p] < buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p] + boxsize)
        # y high, z low
        if y[p] > boxsize-buffer_length and z[p] < buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p] + boxsize)
        # y low, z high
        if y[p] < buffer_length and z[p] > boxsize-buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p] + boxsize)
            z_copied_points.append(z[p] - boxsize)
        # y high, z high
        if y[p] > boxsize-buffer_length and z[p] > boxsize-buffer_length:
            x_copied_points.append(x[p])
            y_copied_points.append(y[p] - boxsize)
            z_copied_points.append(z[p] - boxsize)

        # x low, z low
        if x[p] < buffer_length and z[p] < buffer_length:
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p])
            z_copied_points.append(z[p] + boxsize)
        # x high, z low
        if x[p] > boxsize-buffer_length and z[p] < buffer_length:
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p])
            z_copied_points.append(z[p] + boxsize)
        # x low, z high
        if x[p] < buffer_length and z[p] > boxsize-buffer_length:
            x_copied_points.append(x[p] + boxsize)
            y_copied_points.append(y[p])
            z_copied_points.append(z[p] - boxsize)
        # x high, z high
        if x[p] > boxsize-buffer_length and z[p] > boxsize-buffer_length:
            x_copied_points.append(x[p] - boxsize)
            y_copied_points.append(y[p])
            z_copied_points.append(z[p] - boxsize)

    # combine and return
    x_periodic = np.append(x, x_copied_points)
    y_periodic = np.append(y, y_copied_points)
    z_periodic = np.append(z, z_copied_points)

    return x_periodic, y_periodic, z_periodic


@cuda.jit(fastmath=False)
def _count_particles(
    xh, yh, zh,
    wh, wh_jac,
    mask,
    n_grads,
    xp, yp, zp,
    start_idx, end_idx,
    rpbins,
    zmax,
    result, result_grad
):

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    n_halos = len(xh)
    n_bins = len(rpbins) - 1

    # for each halo
    for i in range(start, n_halos, stride):
        if mask[i]:
            # for each particle in specified range
            for j in range(start_idx, end_idx):
                # ensure Z distance is within range
                if abs(zh[i] - zp[j]) > zmax:
                    continue

                # calculate XY distance
                pdist = math.sqrt((xh[i]-xp[j])*(xh[i]-xp[j]) +
                                  (yh[i]-yp[j])*(yh[i]-yp[j]))

                for r in range(n_bins):
                    if pdist >= rpbins[r] and pdist < rpbins[r+1]:
                        # add weight from halo
                        cuda.atomic.add(result, r, wh[i])
                        # and for gradients
                        for g in range(n_grads):
                            cuda.atomic.add(result_grad, (g, r), wh_jac[g, i])
                        break


def sigma_serial_cuda(
    *,
    xh, yh, zh, wh,
    wh_jac,
    mask,
    xp, yp, zp,
    rpbins,
    zmax,
    boxsize,
    threads=32,
    blocks=512
):
    """
    sigma_serial_cuda(...)
        Compute sigma with derivatives for a periodic volume.

    Parameters
    ----------
    xh, yh, zh, wh : array-like, shape(n_halos,)
        The arrays of positions and weights for the halos.
    wh_jac : array-like, shape(n_grads, n_halos,)
        The array of weight gradients for the first set of points.
    mask : array-like, shape (n_halos,)
        A boolean array with `True` for halos to be included and `False` for halos
        to be ignored. Generally used to mask out zero weights. Passed as a
        parameter to avoid copying masked data with each kernel call.
    xp, yp, zp : array-like, shape (n_particles,)
        The arrays of positions for the particles.
    rpbins : array-like, shape (n_rpbins+1,)
        Array of radial bin edges, note that this array is one longer than the
        number of bins in the 'rp' (xy radial) direction.
    zmax : float
        Maximum separation in the z direction. Particles with z distance less
        than zmax from a given halo are included in surface density calculations
        for that halo.
    boxsize: float
        Length of the periodic volume, not used in the cuda kernel but included
        for consistency with CPU versions.

    Returns
    -------
    sigma : array-like, shape(n_rpbins,)
        The surface density at each radial bin.
    sigma_grad : array-like, shape(n_rpbins,)
        The surface density gradients at each radial bin.
    """

    # check if cupy is available
    qp = cp.get_array_module(xp)

    # ensure smallest bin is zero
    assert qp.allclose(rpbins[0], 0.0)

    # set up sizes
    n_grads = wh_jac.shape[0]

    n_rpbins = len(rpbins) - 1
    rads = qp.pi * (qp.square(rpbins[1:], dtype=qp.float64) -
                    qp.square(rpbins[:-1], dtype=qp.float64))

    rpmax = rpbins[-1]
    periodic_buffer = max(rpmax, zmax)

    # handle periodicity
    xp_p, yp_p, zp_p = _copy_periodic_points_3D(xp, yp, zp, boxsize, periodic_buffer)

    # set up device arrays
    sigma_device = qp.zeros(n_rpbins, dtype=qp.float64)
    sigma_grad_1st_device = qp.zeros((n_grads, n_rpbins),
                                     dtype=qp.float64)

    # do the actual counting on GPU
    _count_particles[blocks, threads](
                                      xh, yh, zh, wh, wh_jac,
                                      mask,
                                      n_grads,
                                      xp_p, yp_p, zp_p,
                                      0, len(xp_p),
                                      rpbins,
                                      zmax,
                                      sigma_device,
                                      sigma_grad_1st_device
    )

    sigma_exp = qp.array(sigma_device)
    sigma_grad_1st = sigma_grad_1st_device.reshape((n_grads, n_rpbins))

    # normalize by surface area
    sigma_exp /= rads
    sigma_grad_1st /= rads

    # normalize sigma by weights sum
    weights_sum = qp.sum(wh, dtype=qp.float64)
    sigma_exp /= weights_sum

    # second term of gradient
    grad_sum = qp.sum(wh_jac, axis=1, dtype=qp.float64).reshape(n_grads, 1)
    sigma_row = sigma_exp.reshape(1, n_rpbins)
    sigma_grad_2nd = qp.matmul(grad_sum, sigma_row, dtype=qp.float64)

    # subtract gradient terms
    sigma_grad = sigma_grad_1st - sigma_grad_2nd

    # normalize gradient by weights sum
    sigma_grad /= weights_sum

    # return
    return sigma_exp, sigma_grad


def sigma_mpi_kernel_cuda(
    *,
    xh, yh, zh, wh,
    wh_jac,
    mask,
    xp, yp, zp,
    rpbins,
    zmax,
    boxsize,
    threads=32,
    blocks=512
):
    """
    sigma_mpi_kernel_cuda(...)
        Per-process CUDA kernel for MPI-parallel sigma computations.

    Parameters
    ----------
    xh, yh, zh, wh : array-like, shape (n_halos,)
        The arrays of positions and weights for the halos.
    wh_jac : array-like, shape (n_grads, n_halos,)
        The array of weight gradients for the first set of points.
    mask : array-like, shape (n_halos,)
        A boolean array with `True` for halos to be included and `False` for halos
        to be ignored. Generally used to mask out zero weights. Passed as a
        parameter to avoid copying masked data with each kernel call.
    xp, yp, zp : array-like, shape (n_particles,)
        The arrays of positions for the particles.
    rpbins : array-like, shape (n_rpbins+1,)
        Array of radial bin edges, Note that this array is one longer than the
        number of bins in the 'rp' (xy radial) direction.
    zmax : float
        Maximum separation in the z direction. Particles with z distance less
        than zmax from a given halo are included in surface density calculations
        for that halo.
    boxsize: float
        Length of the periodic volume, not used in the cuda kernel but included
        for consistency with CPU versions.

    Returns
    -------
    sigma_mpi_data : named typle of type SigmaMPIData
        A named tuple of the data needed for the MPI reduction and final summary stats.
    """
    # check if cupy is available
    # bc cupy has no "simulation" mode to use on github CI
    qp = cp.get_array_module(xp[0])
    can_cupy = qp is cp

    # use multiple GPUs is available
    n_devices = 1
    if can_cupy:
        n_devices = len(cuda.gpus)

    # check that first bin starts at zero
    for d in range(n_devices):
        assert qp.allclose(rpbins[d][0], 0)

    # set up sizes
    n_grads = wh_jac[0].shape[0]
    n_rpbins = len(rpbins[0]) - 1

    # split the particle data into chunks for each device
    # this made more of a performance difference than splitting by halos
    avg, rem = divmod(len(xp[0]), n_devices)
    device_count = [avg + 1 if p < rem else avg for p in range(n_devices)]
    device_displ = [sum(device_count[:p]) for p in range(n_devices)]
    # list of results for combination later
    result_all = []
    result_grad_all = []
    for d in range(n_devices):
        # get this chunk range
        count = device_count[d]
        displ = device_displ[d]
        start_idx = displ
        end_idx = displ + count

        # data must already be copied to relevant gpus
        if can_cupy:
            cp.cuda.Device(d).use()

        sigma_d = qp.zeros(n_rpbins, dtype=qp.float64)
        sigma_grad_1st_d = qp.zeros((n_grads, n_rpbins),
                                    dtype=qp.float64)

        # launch kernel
        _count_particles[blocks, threads](
                                            xh[d], yh[d], zh[d],
                                            wh[d], wh_jac[d],
                                            mask[d],
                                            n_grads,
                                            xp[d], yp[d], zp[d],
                                            start_idx, end_idx,
                                            rpbins[d],
                                            zmax,
                                            sigma_d,
                                            sigma_grad_1st_d
                                         )

        # add chunked result to list
        result_all.append(sigma_d)
        result_grad_all.append(sigma_grad_1st_d)

    # now add the distributed calculation
    if can_cupy:
        cp.cuda.Device(0).use()
    sigma_exp = qp.copy(result_all[0])
    sigma_grad_1st = qp.copy(result_grad_all[0])
    for d in range(1, n_devices):
        # let's be explicit about moving data between devices
        rd = qp.copy(result_all[d])
        rd_grad = qp.copy(result_grad_all[d])

        sigma_exp += rd
        sigma_grad_1st += rd_grad

    sigma_grad_1st = sigma_grad_1st.reshape((n_grads, n_rpbins))

    # do radial normalization
    sigma_exp /= qp.pi * (rpbins[0][1:]**2 - rpbins[0][:-1]**2)

    # return data on cpu
    if can_cupy:
        return SigmaMPIData(
                sigma=qp.asnumpy(sigma_exp.get()),
                sigma_grad_1st=qp.asnumpy(sigma_grad_1st.get()),
                w_tot=qp.asnumpy(qp.sum(wh[0][mask[0]])),
                w_jac_tot=qp.asnumpy(qp.sum(wh_jac[0][:, mask[0]], axis=1))
        )

    else:
        return SigmaMPIData(
                sigma=sigma_exp,
                sigma_grad_1st=sigma_grad_1st,
                w_tot=np.sum(wh[0][mask[0]]),
                w_jac_tot=np.sum(wh_jac[0][:, mask[0]], axis=1)
        )
