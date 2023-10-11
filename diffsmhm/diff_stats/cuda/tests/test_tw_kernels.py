import numpy as np
from numpy.testing import assert_allclose
from numba import cuda

import pytest

from diffsmhm.diff_stats.cpu.tw_kernels import (
    tw_kern_mstar_bin_weights_and_derivs_cpu,
)
from diffsmhm.diff_stats.cuda.tw_kernels import (
    tw_kern_mstar_bin_weights_and_derivs_cuda,
)

from .conftest import SKIP_CUDA_TESTS


@pytest.mark.mpi_skip
@pytest.mark.skipif(
    SKIP_CUDA_TESTS,
    reason="numba not in CUDA simulator mode or no CUDA-capable GPU is available",
)
def test_tw_kern_mstar_bin_weights_and_derivs_cuda():
    log10mstar = np.array([10.0, 9.5])
    log10mstar_jac = np.array([[1.0, 2.0], [2.0, 1.0]])
    sigma = np.array([0.1, 0.2])
    sigma_jac = np.array([[1.0, 3.0], [3.0, 1.0]])
    log10mstar_low = 9.9
    log10mstar_high = 10.05

    def _new_fill():
        return np.zeros(2, dtype=np.float64), np.zeros((2, 2), dtype=np.float64)

    w_cpu, w_grad_cpu = _new_fill()
    tw_kern_mstar_bin_weights_and_derivs_cpu(
        log10mstar, log10mstar_jac,
        sigma, sigma_jac,
        log10mstar_low, log10mstar_high,
        w_cpu, w_grad_cpu,
    )

    w_cuda, w_grad_cuda = _new_fill()
    w_cuda = cuda.to_device(w_cuda)
    w_grad_cuda = cuda.to_device(w_grad_cuda)
    tw_kern_mstar_bin_weights_and_derivs_cuda[2, 2](
        log10mstar, log10mstar_jac,
        sigma, sigma_jac,
        log10mstar_low, log10mstar_high,
        w_cuda, w_grad_cuda,
    )
    w_cuda = w_cuda.copy_to_host()
    w_grad_cuda = w_grad_cuda.copy_to_host()

    assert_allclose(w_cpu, w_cuda)
    assert_allclose(w_grad_cpu, w_grad_cuda)
