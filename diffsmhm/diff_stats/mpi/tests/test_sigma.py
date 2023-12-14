from numpy.testing import assert_allclose
import numpy as np
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

from diffsmhm.diff_stats.cuda.sigma import(
    sigma_mpi_kernel_cuda
)
from diffsmhm.diff_stats.cpu.sigma import(
    sigma_cpu_serial,
    sigma_mpi_kernel_cpu
)
from diffsmhm.diff_stats.mpi.sigma import(
    sigma_mpi_comp_and_reduce
)

from diffsmhm.testing import gen_mstar_data
from diffsmhm.loader import wrap_to_local_volume_inplace


def _gen_data(n_halos, n_particles, n_pars, lbox, seed):
    rng = np.random.RandomState(seed=seed)

    halo_catalog = dict()
    particle_catalog = dict()
    if RANK == 0:
        _halo_catalog = gen_mstar_data(seed=seed, npts=n_halos, boxsize=lbox,
                                       npars=n_pars)
        halo_catalog["x"] = _halo_catalog["x"]
        halo_catalog["y"] = _halo_catalog["y"]
        halo_catalog["z"] = _halo_catalog["z"]
        halo_catalog["w1"] = _halo_catalog["w"]
        for i in range(3):
            halo_catalog["dw1_%d" % i] = _halo_catalog["w_jac"][i, :]

        particle_catalog["x"] = np.random.uniform(0.0, lbox, n_particles)
        particle_catalog["y"] = np.random.uniform(0.0, lbox, n_particles)
        particle_catalog["z"] = np.random.uniform(0.0, lbox, n_particles)
    else:
        halo_catalog["x"] = np.array([], dtype=np.double)
        halo_catalog["y"] = np.array([], dtype=np.double)
        halo_catalog["z"] = np.array([], dtype=np.double)
        halo_catalog["w1"] = np.array([], dtype=np.double)
        for i in range(3):
            halo_catalog["dw1_%d" % i]= np.array([], dtype=np.double)

        particle_catalog["x"] = np.array([], dtype=np.double)
        particle_catalog["y"] = np.array([], dtype=np.double)
        particle_catalog["z"] = np.array([], dtype=np.double)

    return halo_catalog, particle_catalog


def _distribute_data(partition, lbox, data, lov):
    data = mpipartition.distribute(partition, lbox, data, ["x","y"])
    data["rank"] = np.zeros_like(data["x"], dtype=np.int32) + RANK

    data = mpipartition.overload(partition, lbox, data, lov, ["x","y"])
    data["_inside_subvol"] = data["rank"] == RANK

    center = lbox * (
        np.array(partition.extent) / 2.0
        + np.array(partition.origin)
    )

    # currently sigma is only 2D so cannot wrap 'z'
    wrap_to_local_volume_inplace(data["x"], center[0], lbox)
    wrap_to_local_volume_inplace(data["y"], center[1], lbox)
    #wrap_to_local_volume_inplace(data["z"], center[2], lbox)

    return data


def test_sigma_mpi_comp_and_reduce_cpu():
    lbox = 100.0

    n_bins = 10
    rpmax = 5

    seed = 42

    n_halos = 100
    n_particles = 1000

    n_pars = 3

    lov = rpmax

    rpbins = np.linspace(0.1, rpmax, n_bins+1)

    # get data
    halo_cat_orig, particle_cat_orig = _gen_data(n_halos, n_particles, n_pars, 
                                                 lbox, seed
                                                 )

    # distribute and overload
    # note: halos don't need to be overloaded for this measurement, but 
    # 	    we do it here to test handling the case where they are
    partition = mpipartition.Partition(2)

    halo_catalog = _distribute_data(partition, lbox, halo_cat_orig, lov)
    particle_catalog = _distribute_data(partition, lbox, particle_cat_orig, lov)

    # now we stack gradients
    halo_dw1 = np.stack([halo_catalog["dw1_%d" % h] for h in range(n_pars)], axis=0)

    sigma_mpi, sigma_grad_mpi = sigma_mpi_comp_and_reduce(
        xh = halo_catalog["x"], 
        yh = halo_catalog["y"],
        zh = halo_catalog["z"], 
        wh = halo_catalog["w1"], 
        wh_jac = halo_dw1,
        xp = particle_catalog["x"],
        yp = particle_catalog["y"],
        zp = particle_catalog["z"], 
        inside_subvol = halo_catalog["_inside_subvol"],
        rpbins = rpbins,
        kernel_func = sigma_mpi_kernel_cpu
    )

    if RANK == 0:
        # stack gradients for non-distributed catalog
        halo_dw1_orig = np.stack([halo_cat_orig["dw1_%d" % h] for h in range(n_pars)], axis=0)

        # call serial version to check mpi 
        sigma_serial, sigma_grad_serial = sigma_cpu_serial(
            xh = halo_cat_orig["x"],
            yh= halo_cat_orig["y"],
            zh = halo_cat_orig["z"],
            wh = halo_cat_orig["w1"],
            wh_jac = halo_dw1_orig,
            xp = particle_cat_orig["x"],
            yp = particle_cat_orig["y"],
            zp = particle_cat_orig["z"],
            rpbins = rpbins,
            box_length=lbox
		)

        ok = True
        try:
            assert_allclose(sigma_mpi, sigma_serial)
            assert_allclose(sigma_grad_mpi, sigma_grad_serial)
        except:
            ok = False
    else:
        ok = None
        assert sigma_mpi is None	
        assert sigma_grad_mpi is None

    ok = COMM.bcast(ok, root=0)
    if not ok and RANK == 0:
        assert_allclose(sigma_mpi, sigma_serial)


def test_sigma_mpi_comp_and_reduce_cuda():
    lbox = 100

    n_bins = 10
    rpmax = 5

    seed = 42

    n_halos = 100
    n_particles = 1000

    n_pars = 3

    lov = rpmax
    rpbins = np.linspace(0.1, rpmax, n_bins+1)

    # get data
    halo_cat_orig, particle_cat_orig = _gen_data(n_halos, n_particles, n_pars,
                                                    lbox, seed)

    # distribute and overload
    partition = mpipartition.Partition(2)

    halo_catalog = _distribute_data(partition, lbox, halo_cat_orig, lov)
    particle_catalog = _distribute_data(partition, lbox, particle_cat_orig, lov)

    halo_dw1 = np.stack([halo_catalog["dw1_%d" % h] for h in range(n_pars)], axis=0)

    sigma_mpi, sigma_grad_mpi = sigma_mpi_comp_and_reduce(
        xh = halo_catalog["x"], 
        yh = halo_catalog["y"],
        zh = halo_catalog["z"], 
        wh = halo_catalog["w1"], 
        wh_jac = halo_dw1,
        xp = particle_catalog["x"],
        yp = particle_catalog["y"],
        zp = particle_catalog["z"], 
        inside_subvol = halo_catalog["_inside_subvol"],
        rpbins = rpbins,
        kernel_func = sigma_mpi_kernel_cuda
    )

    if RANK == 0:
        # stack gradients of original, non-distributed catalog
        halo_dw1_orig = np.stack([halo_cat_orig["dw1_%d" % h] for h in range(n_pars)], axis=0)

        # call serial version to check mpi 
        sigma_serial, sigma_grad_serial = sigma_cpu_serial(
            xh = halo_cat_orig["x"],
            yh = halo_cat_orig["y"],
            zh = halo_cat_orig["z"],
            wh = halo_cat_orig["w1"],
            wh_jac = halo_dw1_orig,
            xp = particle_cat_orig["x"],
            yp = particle_cat_orig["y"],
            zp = particle_cat_orig["z"],
            rpbins = rpbins,
            box_length=lbox
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

