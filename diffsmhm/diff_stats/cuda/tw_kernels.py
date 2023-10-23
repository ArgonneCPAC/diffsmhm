from numba import cuda


@cuda.jit(device=True, fastmath=False)
def tw_cuml_kern_cuda(x, m, h):
    """CDF of the triweight kernel.

    Parameters
    ----------
    x : array-like or scalar
        The value at which to evaluate the kernel.
    m : array-like or scalar
        The mean of the kernel.
    h : array-like or scalar
        The approximate 1-sigma width of the kernel.

    Returns
    -------
    kern_cdf : array-like or scalar
        The value of the kernel CDF.
    """
    y = (x - m) / h
    if y < -3:
        return 0
    elif y > 3:
        return 1
    else:
        val = (
            -5 * y ** 7 / 69984
            + 7 * y ** 5 / 2592
            - 35 * y ** 3 / 864
            + 35 * y / 96
            + 1 / 2
        )
        return val


@cuda.jit(device=True, fastmath=False)
def tw_kern_cuda(x, m, h):
    """Triweight kernel.

    Parameters
    ----------
    x : array-like or scalar
        The value at which to evaluate the kernel.
    m : array-like or scalar
        The mean of the kernel.
    h : array-like or scalar
        The approximate 1-sigma width of the kernel.

    Returns
    -------
    kern : array-like or scalar
        The value of the kernel.
    """
    z = (x - m) / h
    if z < -3 or z > 3:
        return 0
    else:
        return 35 / 96 * (1 - (z / 3) ** 2) ** 3 / h


@cuda.jit(fastmath=False)
def tw_kern_mstar_bin_weights_and_derivs_cuda(
    log10mstar, log10mstar_jac, sigma, sigma_jac, log10mstar_low, log10mstar_high,
    w, w_jac,
):
    """Compute the bin weights+derivs given stellar mass, scatter, bin limits,
    and derivs.

    Parameters
    ----------
    log10mstar : ndarray, shape (n_halos,)
        The stellar mass in base-10 log units.
    log10mstar_jac : ndarray, shape (n_params, n_halos)
        The gradients of the stellar mass in base-10 log units wrt the parameters.
    sigma : ndarray, shape (n_halos,)
        The scatter in the base-10 log stellar mass.
    sigma_jac : ndarray, shape (n_params, n_halos)
        The gradients in the scatter in the base-10 log stellar mass.
    log10mstar_low : float
        The lower edge of the stellar mass bin.
    log10mstar_high : float
        The upper edge of the stellar mass bin.
    w : ndarray, shape (n_halos,)
        The array to fill for weights for the bin.
    w_jac : ndarray, shape (n_params, n_halos)
        The array to fill for gradients of the weights wrt the params.
    """
    n_params = log10mstar_jac.shape[0]
    n_data = log10mstar.shape[0]

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, n_data, stride):
        x = log10mstar[i]
        sig = sigma[i]
        last_cdf = tw_cuml_kern_cuda(log10mstar_low, x, sig)
        last_cdf_deriv = tw_kern_cuda(log10mstar_low, x, sig)

        new_cdf = tw_cuml_kern_cuda(log10mstar_high, x, sig)
        new_cdf_deriv = tw_kern_cuda(log10mstar_high, x, sig)

        w[i] = new_cdf - last_cdf

        # do the derivs
        for k in range(n_params):
            fac1 = log10mstar_jac[k, i] - x / sig * sigma_jac[k, i]
            fac2 = sigma_jac[k, i] / sig
            w_jac[k, i] = last_cdf_deriv * (
                fac1 + log10mstar_low * fac2
            ) - new_cdf_deriv * (fac1 + log10mstar_high * fac2)
