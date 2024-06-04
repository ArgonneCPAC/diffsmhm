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

from diffsmhm.diff_stats.cupy_utils import get_array_backend


@pytest.mark.mpi_skip
@pytest.mark.skipif(
    SKIP_CUDA_TESTS,
    reason="numba not in CUDA simulator mode or no CUDA-capable GPU is available",
)
def test_wprp_serial_cuda_smoke():
    xp = get_array_backend()

    data = gen_mstar_data(seed=42)
    wgt_mask = data["w"] > 0
    dwgt_mask = np.sum(np.abs(data["w_jac"]), axis=0) > 0
    full_mask = wgt_mask & dwgt_mask

    nrp = data["rp_bins"].shape[0] - 1
    wprp, wprp_grad = wprp_serial_cuda(
        x1=xp.asarray(data["x"]),
        y1=xp.asarray(data["y"]),
        z1=xp.asarray(data["z"]),
        w1=xp.asarray(data["w"]),
        w1_jac=xp.asarray(data["w_jac"]),
        mask=xp.array(full_mask),
        rpbins_squared=xp.asarray(data["rp_bins"]**2),
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
    xp = get_array_backend()

    data = gen_mstar_data(seed=42)
    wgt_mask = data["w"] > 0
    dwgt_mask = np.sum(np.abs(data["w_jac"]), axis=0) > 0
    full_mask = wgt_mask & dwgt_mask

    wprp_cuda, wprp_grad_cuda = wprp_serial_cuda(
        x1=xp.asarray(data["x"]),
        y1=xp.asarray(data["y"]),
        z1=xp.asarray(data["z"]),
        w1=xp.asarray(data["w"]),
        w1_jac=xp.asarray(data["w_jac"]),
        mask=xp.array(full_mask),
        rpbins_squared=xp.asarray(data["rp_bins"]**2),
        zmax=data["zmax"],
        boxsize=data["boxsize"],
    )

    wprp_cpu, wprp_grad_cpu = wprp_serial_cpu(
        x1=data["x"],
        y1=data["y"],
        z1=data["z"],
        w1=data["w"],
        w1_jac=data["w_jac"],
        mask=full_mask,
        rpbins_squared=data["rp_bins"]**2,
        zmax=data["zmax"],
        boxsize=data["boxsize"],
    )

    assert_allclose(wprp_cuda, wprp_cpu)
    assert_allclose(wprp_grad_cuda, wprp_grad_cpu)
