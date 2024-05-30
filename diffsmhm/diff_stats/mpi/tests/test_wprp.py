import os

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

from diffsmhm.testing import gen_mstar_data
from diffsmhm.diff_stats.mpi.wprp import wprp_mpi_comp_and_reduce
from diffsmhm.diff_stats.cpu.wprp import (
    wprp_mpi_kernel_cpu,
    wprp_serial_cpu,
)
from diffsmhm.diff_stats.cuda.wprp import (
    wprp_mpi_kernel_cuda,
)
from diffsmhm.loader import wrap_to_local_volume_inplace
from diffsmhm.diff_stats.cuda.tests.conftest import (
    SKIP_CUDA_TESTS
)


def _gen_data(**kwargs):
    halo_catalog = OrderedDict()
    if RANK == 0:
        _halo_catalog = gen_mstar_data(**kwargs)
        halo_catalog["x"] = _halo_catalog["x"]
        halo_catalog["y"] = _halo_catalog["y"]
        halo_catalog["z"] = _halo_catalog["z"]
        halo_catalog["w1"] = _halo_catalog["w"]
        for i in range(3):
            halo_catalog["dw1_%d" % i] = _halo_catalog["w_jac"][i, :]
    else:
        halo_catalog["x"] = np.array([], dtype=np.float64)
        halo_catalog["y"] = np.array([], dtype=np.float64)
        halo_catalog["z"] = np.array([], dtype=np.float64)
        halo_catalog["w1"] = np.array([], dtype=np.float64)
        for i in range(3):
            halo_catalog["dw1_%d" % i] = np.array([], dtype=np.float64)

    return halo_catalog


def _distribute_data(data, lbox, lov):
    partition = mpipartition.Partition()
    data = mpipartition.distribute(partition, lbox, data, ["x", "y", "z"])
    data["rank"] = np.zeros_like(data["x"], dtype=np.int32) + RANK

    data = mpipartition.overload(partition, lbox, data, lov, ["x", "y", "z"])
    data["_inside_subvol"] = data["rank"] == RANK

    center = lbox * (
        np.array(partition.extent) / 2.0
        + np.array(partition.origin)
    )

    wrap_to_local_volume_inplace(data["x"], center[0], lbox)
    wrap_to_local_volume_inplace(data["y"], center[1], lbox)
    wrap_to_local_volume_inplace(data["z"], center[2], lbox)

    return data


@pytest.mark.mpi
def test_wprp_mpi_comp_and_reduce_cpu():
    lbox = 120
    zmax = 10
    nbins = 10
    rpmax = 15
    seed = 42
    npts = 50000
    rpbins_squared = np.logspace(-1, np.log10(rpmax), nbins + 1) ** 2
    halo_catalog = _gen_data(
        seed=seed,
        boxsize=lbox,
        npts=npts,
        rpmax=rpmax,
        zmax=zmax,
        nbins=nbins,
    )
    halo_catalog = _distribute_data(halo_catalog, lbox, rpmax)

    _dw1 = np.stack([halo_catalog["dw1_%d" % g] for g in range(3)], axis=0)
    wprp, wprp_grad = wprp_mpi_comp_and_reduce(
        x1=halo_catalog["x"].astype(np.float64),
        y1=halo_catalog["y"].astype(np.float64),
        z1=halo_catalog["z"].astype(np.float64),
        w1=halo_catalog["w1"].astype(np.float64),
        w1_jac=_dw1.astype(np.float64),
        inside_subvol=halo_catalog["_inside_subvol"],
        rpbins_squared=rpbins_squared,
        zmax=zmax,
        boxsize=lbox,
        kernel_func=wprp_mpi_kernel_cpu,
    )

    if RANK == 0:
        # compare to serial computation
        orig_halo_catalog = _gen_data(
            seed=seed,
            boxsize=lbox,
            npts=npts,
            rpmax=rpmax,
            zmax=zmax,
            nbins=nbins,
        )
        dw1 = np.stack([orig_halo_catalog["dw1_%d" % g] for g in range(3)], axis=0)
        (
            wprp_serial,
            wprp_grad_serial,
        ) = wprp_serial_cpu(
            x1=orig_halo_catalog["x"].astype(np.float64),
            y1=orig_halo_catalog["y"].astype(np.float64),
            z1=orig_halo_catalog["z"].astype(np.float64),
            w1=orig_halo_catalog["w1"].astype(np.float64),
            w1_jac=dw1,
            rpbins_squared=rpbins_squared,
            zmax=zmax,
            boxsize=lbox,
        )
        ok = True
        try:
            assert_allclose(wprp, wprp_serial)
            assert_allclose(wprp_grad, wprp_grad_serial)
        except AssertionError:
            ok = False
    else:
        ok = None
        assert wprp is None
        assert wprp_grad is None

    ok = COMM.bcast(ok, root=0)
    if not ok and RANK == 0:
        assert_allclose(wprp, wprp_serial)
        assert_allclose(wprp_grad, wprp_grad_serial)

    assert ok, "Tests failed - see rank 0 for details!"


@pytest.mark.mpi
@pytest.mark.skipif(
    SKIP_CUDA_TESTS,
    reason="numba not in CUDA simulator mode or no CUDA-capable GPU is available",
)
def test_wprp_mpi_comp_and_reduce_cuda():
    try:
        _ = cp.array([1])
        xp = cp
        can_cupy = True
    except RuntimeError:
        xp = np
        can_cupy = False

    lbox = 120
    zmax = 12
    nbins = 10
    rpmax = 15
    seed = 42
    rpbins_squared = xp.logspace(-1, xp.log10(rpmax), nbins + 1) ** 2

    if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
        npts = 500
    else:
        npts = 100000 #5000000
    halo_catalog = _gen_data(
        seed=seed,
        boxsize=lbox,
        npts=npts,
        rpmax=rpmax,
        zmax=zmax,
        nbins=nbins,
    )
    halo_catalog = _distribute_data(halo_catalog, lbox, rpmax)

    # make jax version that will be on gpu is available
    halo_catalog_jax = OrderedDict()
    for k in halo_catalog.keys():
        halo_catalog_jax[k] = jnp.copy(halo_catalog[k])

    _dw1 = jnp.stack([halo_catalog_jax["dw1_%d" % g] for g in range(3)], axis=0)
    wprp, wprp_grad = wprp_mpi_comp_and_reduce(
        x1=xp.asarray(halo_catalog_jax["x"]).astype(xp.float64),
        y1=xp.asarray(halo_catalog_jax["y"]).astype(xp.float64),
        z1=xp.asarray(halo_catalog_jax["z"]).astype(xp.float64),
        w1=xp.asarray(halo_catalog_jax["w1"]).astype(xp.float64),
        w1_jac=xp.asarray(_dw1).astype(xp.float64),
        inside_subvol=xp.asarray(halo_catalog_jax["_inside_subvol"]),
        rpbins_squared=rpbins_squared,
        zmax=zmax,
        boxsize=lbox,
        kernel_func=wprp_mpi_kernel_cuda,
    )

    if RANK == 0:
        if can_cupy:
            rpbins_squared_cpu = np.array(rpbins_squared.get())
        else:
            rpbins_squared_cpu = np.copy(rpbins_squared)

        # compare to serial computation
        orig_halo_catalog = _gen_data(
            seed=seed,
            boxsize=lbox,
            npts=npts,
            rpmax=rpmax,
            zmax=zmax,
            nbins=nbins,
        )
        dw1 = np.stack([orig_halo_catalog["dw1_%d" % g] for g in range(3)], axis=0)
        (
            wprp_serial,
            wprp_grad_serial,
        ) = wprp_serial_cpu(
            x1=orig_halo_catalog["x"].astype(np.float64),
            y1=orig_halo_catalog["y"].astype(np.float64),
            z1=orig_halo_catalog["z"].astype(np.float64),
            w1=orig_halo_catalog["w1"].astype(np.float64),
            w1_jac=dw1,
            rpbins_squared=rpbins_squared_cpu,
            zmax=zmax,
            boxsize=lbox,
        )
        ok = True
        try:
            assert_allclose(wprp, wprp_serial)
            assert_allclose(wprp_grad, wprp_grad_serial)
        except AssertionError:
            ok = False
    else:
        ok = None
        assert wprp is None
        assert wprp_grad is None

    ok = COMM.bcast(ok, root=0)
    if not ok and RANK == 0:
        assert_allclose(wprp, wprp_serial)
        assert_allclose(wprp_grad, wprp_grad_serial)

    assert ok, "Tests failed - see rank 0 for details!"
