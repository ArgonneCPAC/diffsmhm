import numpy as np
import cupy as cp

import pytest

from diffsmhm.diff_stats.cpu.sigma import sigma_serial_cpu
from diffsmhm.diff_stats.cuda.sigma import sigma_serial_cuda

from diffsmhm.diff_stats.mpi.tests.test_sigma import _gen_data


@pytest.mark.mpi_skip
def test_sigma_serial_cuda():
    try:
        _ = cp.array([1])
        qp = cp
    except RuntimeError:
        qp = np

    lbox = 100.0

    n_bins = 10
    rpmax = 5.0
    zmax = 40.0

    seed = 42

    n_halos = 100
    n_particles = 1000

    n_pars = 3

    rpbins = np.linspace(0.0, rpmax, n_bins+1)

    # get data
    halo_cat, particle_cat = _gen_data(n_halos, n_particles, n_pars, lbox, seed)

    halo_dw1 = np.stack([halo_cat["dw1_%d" % h] for h in range(n_pars)], axis=0)

    sigma_cuda, sigma_grad_cuda = sigma_serial_cuda(
        xh=qp.asarray(halo_cat["x"]),
        yh=qp.asarray(halo_cat["y"]),
        zh=qp.asarray(halo_cat["z"]),
        wh=qp.asarray(halo_cat["w1"]),
        wh_jac=qp.asarray(halo_dw1),
        xp=qp.asarray(particle_cat["x"]),
        yp=qp.asarray(particle_cat["y"]),
        zp=qp.asarray(particle_cat["z"]),
        rpbins=qp.asarray(rpbins),
        zmax=zmax,
        boxsize=lbox
    )

    sigma_cpu, sigma_grad_cpu = sigma_serial_cpu(
        xh=halo_cat["x"],
        yh=halo_cat["y"],
        zh=halo_cat["z"],
        wh=halo_cat["w1"],
        wh_jac=halo_dw1,
        xp=particle_cat["x"],
        yp=particle_cat["y"],
        zp=particle_cat["z"],
        rpbins=rpbins,
        zmax=zmax,
        boxsize=lbox
    )

    qp.testing.assert_allclose(sigma_cpu, sigma_cuda)
    qp.testing.assert_allclose(sigma_grad_cpu, sigma_grad_cuda)
