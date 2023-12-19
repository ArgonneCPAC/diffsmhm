import numpy as np
from numpy.testing import assert_allclose

import pytest

from diffsmhm.diff_stats.cpu.sigma import (
    sigma_serial_cpu,
    delta_sigma_from_sigma
)
from diffsmhm.testing import gen_mstar_data


@pytest.mark.mpi_skip
def test_sigma_serial_cpu_smoke():
    rng = np.random.RandomState(seed=42)

    # generate halo and particle locations
    boxsize = 1.0
    n_halos = 10

    n_pars = 4

    xh = rng.uniform(0.0, boxsize, n_halos)
    yh = rng.uniform(0.0, boxsize, n_halos)
    zh = rng.uniform(0.0, boxsize, n_halos)
    wh = rng.uniform(0.0, 1.0, n_halos)
    dwh = rng.uniform(0.0, 1.0, (n_pars, n_halos))

    n_particles = 1000

    xp = rng.uniform(0.0, boxsize, n_particles)
    yp = rng.uniform(0.0, boxsize, n_particles)
    zp = rng.uniform(0.0, boxsize, n_particles)

    # radial bins
    n_bins = 5
    bins = np.linspace(0.1, 0.6, n_bins+1)

    # do calculation
    sigma, sigma_grad = sigma_serial_cpu(
        xh=xh, yh=yh, zh=zh, wh=wh, wh_jac=dwh,
        xp=xp, yp=yp, zp=zp,
        rpbins=bins, boxsize=boxsize
    )

    # check calculation
    assert sigma.shape == (n_bins,)
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma >= 0)

    assert sigma_grad.shape == (n_pars, n_bins)
    assert np.all(np.isfinite(sigma_grad))
    assert np.all(sigma_grad != 0)


@pytest.mark.mpi_skip
def test_sigma_cpu_serial_derivs():
    boxsize = 100.0
    n_halos = 100
    n_particles = 1000

    rseed = 42
    rng = np.random.RandomState(seed=rseed)

    halos = gen_mstar_data(boxsize=boxsize, npts=n_halos, seed=rseed)

    parts_x = rng.uniform(0.0, boxsize, n_particles)
    parts_y = rng.uniform(0.0, boxsize, n_particles)
    parts_z = rng.uniform(0.0, boxsize, n_particles)
    sigma, sigma_grad = sigma_serial_cpu(
        xh=halos["x"],
        yh=halos["y"],
        zh=halos["z"],
        wh=halos["w"],
        wh_jac=halos["w_jac"],
        xp=parts_x,
        yp=parts_y,
        zp=parts_z,
        rpbins=halos["rp_bins"],
        boxsize=boxsize
    )

    eps = 1e-6
    for pind in range(halos["npars"]):
        w_p = halos["w"] + halos["w_jac"][pind, :] * eps
        sigma_p, _ = sigma_serial_cpu(
            xh=halos["x"],
            yh=halos["y"],
            zh=halos["z"],
            wh=w_p,
            wh_jac=halos["w_jac"],
            xp=parts_x,
            yp=parts_y,
            zp=parts_z,
            rpbins=halos["rp_bins"],
            boxsize=boxsize
        )

        w_m = halos["w"] - halos["w_jac"][pind, :] * eps
        sigma_m, _ = sigma_serial_cpu(
            xh=halos["x"],
            yh=halos["y"],
            zh=halos["z"],
            wh=w_m,
            wh_jac=halos["w_jac"],
            xp=parts_x,
            yp=parts_y,
            zp=parts_z,
            rpbins=halos["rp_bins"],
            boxsize=boxsize
        )

        grad = (sigma_p - sigma_m)/2.0/eps
        assert_allclose(sigma_grad[pind, :], grad)
        assert np.any(grad != 0)
        assert np.any(sigma_grad[pind, :] != 0)


@pytest.mark.mpi_skip
def test_delta_sigma_from_sigma():
    rpbins = np.array([1, 2, 3, 4, 5], dtype=np.double)
    sigma = np.array([40, 30, 20, 10], dtype=np.double)

    delta_sigma = delta_sigma_from_sigma(rpbins, sigma)

    delta_sigma_exp = np.array([
                        -40,
                        10/np.pi - 30,
                        70/(9*np.pi) - 20,
                        90/(16*np.pi) - 10
                      ], dtype=np.double)

    assert_allclose(delta_sigma_exp, delta_sigma)
