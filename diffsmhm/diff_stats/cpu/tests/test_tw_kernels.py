import numpy as np

import pytest

from diffsmhm.diff_stats.cpu.tw_kernels import (
    tw_cuml_kern_cpu, tw_kern_cpu, tw_kern_mstar_bin_weights_and_derivs_cpu,
)


@pytest.mark.mpi_skip
def test_tw_kern_cpu():
    h = 2.0
    m = 1.0
    x = np.linspace(-10, 10, 1000)
    dx = x[1] - x[0]

    msk = x < m - 3*h
    assert np.allclose(tw_kern_cpu(x[msk], m, h), 0)

    msk = x > m + 3*h
    assert np.allclose(tw_kern_cpu(x[msk], m, h), 0)

    kv = tw_kern_cpu(x, m, h)
    assert np.allclose(np.sum(kv) * dx, 1.0)

    assert np.all(kv <= tw_kern_cpu(m, m, h))


@pytest.mark.mpi_skip
def test_tw_cuml_kern_cpu():
    h = 2.0
    m = 1.0
    x = np.linspace(-10, 10, 1000)

    msk = x < m - 3*h
    assert np.allclose(tw_cuml_kern_cpu(x[msk], m, h), 0)

    msk = x > m + 3*h
    assert np.allclose(tw_cuml_kern_cpu(x[msk], m, h), 1)

    kv = tw_cuml_kern_cpu(x, m, h)
    assert np.all(kv <= 1.0)


@pytest.mark.mpi_skip
def test_tw_kern_mstar_bin_weights_and_derivs_cpu():
    eps = 1e-6
    log10mstar = np.array([10.0, 9.5])
    log10mstar_jac = np.array([[1.0, 2.0], [2.0, 1.0]])
    sigma = np.array([0.1, 0.2])
    sigma_jac = np.array([[1.0, 3.0], [3.0, 1.0]])
    log10mstar_low = 9.9
    log10mstar_high = 10.05

    def _new_fill():
        return np.zeros(2, dtype=np.float64), np.zeros((2, 2), dtype=np.float64)

    w, w_grad = _new_fill()
    tw_kern_mstar_bin_weights_and_derivs_cpu(
        log10mstar, log10mstar_jac,
        sigma, sigma_jac,
        log10mstar_low, log10mstar_high,
        w, w_grad,
    )

    w_p0, blah = _new_fill()
    tw_kern_mstar_bin_weights_and_derivs_cpu(
        log10mstar + log10mstar_jac[0, :]*eps, log10mstar_jac,
        sigma + sigma_jac[0, :]*eps, sigma_jac,
        log10mstar_low, log10mstar_high,
        w_p0, blah,
    )

    w_m0, blah = _new_fill()
    tw_kern_mstar_bin_weights_and_derivs_cpu(
        log10mstar - log10mstar_jac[0, :]*eps, log10mstar_jac,
        sigma - sigma_jac[0, :]*eps, sigma_jac,
        log10mstar_low, log10mstar_high,
        w_m0, blah,
    )
    gval = (w_p0 - w_m0)/2.0/eps
    assert np.allclose(w_grad[0, :], gval)

    w_p1, blah = _new_fill()
    tw_kern_mstar_bin_weights_and_derivs_cpu(
        log10mstar + log10mstar_jac[1, :]*eps, log10mstar_jac,
        sigma + sigma_jac[1, :]*eps, sigma_jac,
        log10mstar_low, log10mstar_high,
        w_p1, blah,
    )

    w_m1, blah = _new_fill()
    tw_kern_mstar_bin_weights_and_derivs_cpu(
        log10mstar - log10mstar_jac[1, :]*eps, log10mstar_jac,
        sigma - sigma_jac[1, :]*eps, sigma_jac,
        log10mstar_low, log10mstar_high,
        w_m1, blah,
    )
    assert np.allclose(w_grad[1, :], (w_p1 - w_m1)/2.0/eps)
