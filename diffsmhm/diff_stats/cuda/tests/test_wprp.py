import numpy as np

from numpy.testing import assert_allclose

import pytest

from diffsmhm.diff_stats.cpu.wprp import (
    wprp_serial_cpu
)
from diffsmhm.diff_stats.cuda.wprp import (
    wprp_serial_cuda
)
from diffsmhm.testing import gen_mstar_data

from .conftest import SKIP_CUDA_TESTS


@pytest.mark.mpi_skip
@pytest.mark.skipif(
    SKIP_CUDA_TESTS,
    reason="numba not in CUDA simulator mode or no CUDA-capable GPU is available",
)
def test_wprp_serial_cuda_smoke():
    data = gen_mstar_data(seed=42)
    bins = np.logspace(0.1, 15, 10)
    bins = np.concatenate(np.array([0.0]), bins)

    nrp = data["rp_bins"].shape[0] - 1
    wprp, wprp_grad = wprp_serial_cuda(
        x1=data["x"],
        y1=data["y"],
        z1=data["z"],
        w1=data["w"],
        w1_jac=data["w_jac"],
        rpbins_squared=bins**2,
        zmax=data["zmax"],
        boxsize=data["boxsize"],
    )

    assert wprp.shape == (nrp,)
    assert wprp_grad.shape == (data["npars"], nrp)
    assert np.all(np.isfinite(wprp))
    assert np.all(np.isfinite(wprp_grad))
    assert np.any(wprp != 0)
    assert np.any(wprp_grad != 0)


@pytest.mark.mpi_skip
@pytest.mark.skipif(
    SKIP_CUDA_TESTS,
    reason="numba not in CUDA simulator mode or no CUDA-capable GPU is available",
)
def test_wprp_serial_cuda():
    data = gen_mstar_data(seed=42)
    bins = np.logspace(0.1, 15, 10)
    bins = np.concatenate(np.array([0.0]), bins)

    wprp_cuda, wprp_grad_cuda = wprp_serial_cuda(
        x1=data["x"],
        y1=data["y"],
        z1=data["z"],
        w1=data["w"],
        w1_jac=data["w_jac"],
        rpbins_squared=bins**2,
        zmax=data["zmax"],
        boxsize=data["boxsize"],
    )

    wprp_cpu, wprp_grad_cpu = wprp_serial_cpu(
        x1=data["x"],
        y1=data["y"],
        z1=data["z"],
        w1=data["w"],
        w1_jac=data["w_jac"],
        rpbins_squared=bins**2,
        zmax=data["zmax"],
        boxsize=data["boxsize"],
    )

    assert_allclose(wprp_cuda, wprp_cpu)
    assert_allclose(wprp_grad_cuda, wprp_grad_cpu)
