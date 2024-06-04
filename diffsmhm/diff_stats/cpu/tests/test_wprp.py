import numpy as np

from numpy.testing import assert_allclose

import pytest

from diffsmhm.diff_stats.cpu.wprp import (
    wprp_serial_cpu
)
from diffsmhm.testing import gen_mstar_data


@pytest.mark.mpi_skip
def test_wprp_serial_cpu_smoke():
    data = gen_mstar_data(seed=42)

    nrp = data["rp_bins"].shape[0] - 1
    mask = data["w"] > 0.0
    wprp, wprp_grad = wprp_serial_cpu(
        x1=data["x"],
        y1=data["y"],
        z1=data["z"],
        w1=data["w"],
        w1_jac=data["w_jac"],
        mask=mask,
        rpbins_squared=data["rp_bins"]**2,
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
def test_wprp_serial_cpu_derivs():
    data = gen_mstar_data(seed=42)
    mask = data["w"] > 0.0

    wprp, wprp_grad = wprp_serial_cpu(
        x1=data["x"],
        y1=data["y"],
        z1=data["z"],
        w1=data["w"],
        w1_jac=data["w_jac"],
        mask=mask,
        rpbins_squared=data["rp_bins"]**2,
        zmax=data["zmax"],
        boxsize=data["boxsize"],
    )

    eps = 1e-6
    for pind in range(data["npars"]):
        w_p = data["w"] + data["w_jac"][pind, :] * eps
        wprp_p, _ = wprp_serial_cpu(
            x1=data["x"],
            y1=data["y"],
            z1=data["z"],
            w1=w_p,
            w1_jac=data["w_jac"],
            mask=mask,
            rpbins_squared=data["rp_bins"]**2,
            zmax=data["zmax"],
            boxsize=data["boxsize"],
        )

        w_m = data["w"] - data["w_jac"][pind, :] * eps
        wprp_m, _ = wprp_serial_cpu(
            x1=data["x"],
            y1=data["y"],
            z1=data["z"],
            w1=w_m,
            w1_jac=data["w_jac"],
            mask=mask,
            rpbins_squared=data["rp_bins"]**2,
            zmax=data["zmax"],
            boxsize=data["boxsize"],
        )

        grad = (wprp_p - wprp_m)/2.0/eps
        assert_allclose(wprp_grad[pind, :], grad)
        assert np.any(grad != 0)
        assert np.any(wprp_grad[pind, :] != 0)
