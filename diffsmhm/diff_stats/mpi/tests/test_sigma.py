from collections import OrderedDict

from numpy.testing import assert_allclose
import numpy as np
import cupy as cp
import jax.numpy as jnp
import pytest

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1

if COMM is not None:
    import mpipartition

from diffsmhm.diff_stats.cuda.sigma import (
    sigma_mpi_kernel_cuda
)
from diffsmhm.diff_stats.cpu.sigma import (
    sigma_serial_cpu,
    sigma_mpi_kernel_cpu
)
from diffsmhm.diff_stats.mpi.sigma import (
    sigma_mpi_comp_and_reduce
)

from diffsmhm.testing import gen_mstar_data
from diffsmhm.loader import wrap_to_local_volume_inplace


def _gen_data(n_halos, n_particles, n_pars, lbox, seed):
    rng = np.random.RandomState(seed)

    halo_catalog = OrderedDict()
    particle_catalog = OrderedDict()

    if RANK == 0:
        _halo_catalog = gen_mstar_data(seed=seed, npts=n_halos, boxsize=lbox,
                                       npars=n_pars)
        halo_catalog["x"] = _halo_catalog["x"]
        halo_catalog["y"] = _halo_catalog["y"]
        halo_catalog["z"] = _halo_catalog["z"]
        halo_catalog["w1"] = _halo_catalog["w"]
        for i in range(3):
            halo_catalog["dw1_%d" % i] = _halo_catalog["w_jac"][i, :]

        particle_catalog["x"] = rng.uniform(0.0, lbox, n_particles)
        particle_catalog["y"] = rng.uniform(0.0, lbox, n_particles)
        particle_catalog["z"] = rng.uniform(0.0, lbox, n_particles)
    else:
        halo_catalog["x"] = np.array([], dtype=np.float64)
        halo_catalog["y"] = np.array([], dtype=np.float64)
        halo_catalog["z"] = np.array([], dtype=np.float64)
        halo_catalog["w1"] = np.array([], dtype=np.float64)
        for i in range(3):
            halo_catalog["dw1_%d" % i] = np.array([], dtype=np.float64)

        particle_catalog["x"] = np.array([], dtype=np.float64)
        particle_catalog["y"] = np.array([], dtype=np.float64)
        particle_catalog["z"] = np.array([], dtype=np.float64)

    return halo_catalog, particle_catalog


def _distribute_data(partition, lbox, data, lov):
    data_rank = mpipartition.distribute(partition, lbox, data, ["x", "y", "z"])
    data_rank["rank"] = np.zeros_like(data_rank["x"], dtype=np.int32) + RANK

    data_rank = mpipartition.overload(partition, lbox, data_rank, lov, ["x", "y", "z"])
    data_rank["_inside_subvol"] = data_rank["rank"] == RANK

    center = lbox * (
        np.array(partition.extent) / 2.0
        + np.array(partition.origin)
    )

    wrap_to_local_volume_inplace(data_rank["x"], center[0], lbox)
    wrap_to_local_volume_inplace(data_rank["y"], center[1], lbox)
    wrap_to_local_volume_inplace(data_rank["z"], center[2], lbox)

    return data_rank


@pytest.mark.mpi
def test_sigma_mpi_comp_and_reduce_cpu():
    lbox = 250.0

    n_bins = 10
    rpmax = 10

    seed = 42

    n_halos = 100
    n_particles = 1000

    n_pars = 3

    zmax = 40.0
    lov = max(rpmax, zmax)

    rpbins = np.linspace(0.0, rpmax, n_bins+1)

    # get data
    halo_cat_orig, particle_cat_orig = _gen_data(n_halos, n_particles, n_pars,
                                                 lbox, seed)

    # distribute and overload
    # note: halos don't need to be overloaded for this measurement, but
    # 	    we do it here to test handling the case where they are
    partition = mpipartition.Partition()

    halo_catalog = _distribute_data(partition, lbox, halo_cat_orig, lov)
    particle_catalog = _distribute_data(partition, lbox, particle_cat_orig, lov)

    # stack gradients
    halo_dw1 = np.stack([halo_catalog["dw1_%d" % h] for h in range(n_pars)], axis=0)

    sigma_mpi, sigma_grad_mpi = sigma_mpi_comp_and_reduce(
        xh=halo_catalog["x"],
        yh=halo_catalog["y"],
        zh=halo_catalog["z"],
        wh=halo_catalog["w1"],
        wh_jac=halo_dw1,
        xp=particle_catalog["x"],
        yp=particle_catalog["y"],
        zp=particle_catalog["z"],
        inside_subvol=halo_catalog["_inside_subvol"],
        rpbins=rpbins,
        zmax=zmax,
        boxsize=lbox,
        kernel_func=sigma_mpi_kernel_cpu
    )

    if RANK == 0:
        # stack gradients for non-distributed catalog
        halo_dw1_orig = np.stack(
                                  [halo_cat_orig["dw1_%d" % h] for h in range(n_pars)],
                                  axis=0
                                )

        # call serial version to check mpi
        sigma_serial, sigma_grad_serial = sigma_serial_cpu(
            xh=halo_cat_orig["x"],
            yh=halo_cat_orig["y"],
            zh=halo_cat_orig["z"],
            wh=halo_cat_orig["w1"],
            wh_jac=halo_dw1_orig,
            xp=particle_cat_orig["x"],
            yp=particle_cat_orig["y"],
            zp=particle_cat_orig["z"],
            rpbins=rpbins,
            zmax=zmax,
            boxsize=lbox
        )

        ok = True
        try:
            assert_allclose(sigma_mpi, sigma_serial)
            assert_allclose(sigma_grad_mpi, sigma_grad_serial)
        except AssertionError:
            ok = False
    else:
        ok = None
        assert sigma_mpi is None
        assert sigma_grad_mpi is None

    ok = COMM.bcast(ok, root=0)
    if not ok and RANK == 0:
        assert_allclose(sigma_mpi, sigma_serial)


@pytest.mark.mpi
def test_sigma_mpi_comp_and_reduce_cuda():
    try:
        _ = cp.array([1])
        xp = cp
        can_cupy = True
    except RuntimeError:
        xp = np
        can_cupy = False

    lbox = 250.0
    n_bins = 10
    rpmax = 5
    n_halos = 100
    n_particles = 1000
    n_pars = 3

    seed = 42

    rpbins = xp.linspace(0.0, rpmax, n_bins+1)

    zmax = 40.0
    lov = max(zmax, rpmax)

    # get data
    halo_cat_orig, particle_cat_orig = _gen_data(n_halos, n_particles, n_pars,
                                                 lbox, seed)

    # distribute and overload
    partition = mpipartition.Partition()

    halo_catalog = _distribute_data(partition, lbox, halo_cat_orig, lov)
    particle_catalog = _distribute_data(partition, lbox, particle_cat_orig, lov)

    halo_dw1 = np.stack([halo_catalog["dw1_%d" % h] for h in range(n_pars)], axis=0)

    # make jax version that will be on gpu if available 
    halo_catalog_jax = OrderedDict()
    for k in halo_catalog.keys():
        halo_catalog_jax[k] = jnp.copy(halo_catalog[k])
    particle_catalog_jax = OrderedDict()
    for k in particle_catalog.keys():
        particle_catalog_jax[k] = jnp.copy(particle_catalog[k])

    sigma_mpi, sigma_grad_mpi = sigma_mpi_comp_and_reduce(
        xh=xp.asarray(halo_catalog_jax["x"]).astype(xp.float64),
        yh=xp.asarray(halo_catalog_jax["y"]).astype(xp.float64),
        zh=xp.asarray(halo_catalog_jax["z"]).astype(xp.float64),
        wh=xp.asarray(halo_catalog_jax["w1"]).astype(xp.float64),
        wh_jac=xp.asarray(halo_dw1).astype(xp.float64),
        xp=xp.asarray(particle_catalog_jax["x"]).astype(xp.float64),
        yp=xp.asarray(particle_catalog_jax["y"]).astype(xp.float64),
        zp=xp.asarray(particle_catalog_jax["z"]).astype(xp.float64),
        inside_subvol=xp.asarray(halo_catalog_jax["_inside_subvol"]),
        boxsize=lbox,
        rpbins=rpbins,
        zmax=zmax,
        kernel_func=sigma_mpi_kernel_cuda
    )

    if RANK == 0:
        if can_cupy:
            rpbins_cpu = np.array(rpbins.get())
        else:
            rpbins_cpu = np.copy(rpbins)

        # stack gradients of original, non-distributed catalog
        halo_dw1_orig = np.stack(
                                  [halo_cat_orig["dw1_%d" % h] for h in range(n_pars)],
                                  axis=0
                                )

        # call serial version to check mpi
        sigma_serial, sigma_grad_serial = sigma_serial_cpu(
            xh=halo_cat_orig["x"],
            yh=halo_cat_orig["y"],
            zh=halo_cat_orig["z"],
            wh=halo_cat_orig["w1"],
            wh_jac=halo_dw1_orig,
            xp=particle_cat_orig["x"],
            yp=particle_cat_orig["y"],
            zp=particle_cat_orig["z"],
            rpbins=rpbins_cpu,
            zmax=zmax,
            boxsize=lbox
        )

        ok = True
        try:
            assert_allclose(sigma_mpi, sigma_serial)
            assert_allclose(sigma_grad_mpi, sigma_grad_serial)
        except AssertionError:
            ok = False
    else:
        ok = None
        assert sigma_mpi is None
        assert sigma_grad_mpi is None

    ok = COMM.bcast(ok, root=0)
    if not ok and RANK == 0:
        assert_allclose(sigma_mpi, sigma_serial)
