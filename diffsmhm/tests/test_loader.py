import numpy as np
from numpy.testing import assert_allclose
import pytest

import h5py
import os

from diffsmhm.loader import (
    _compute_host_index,
    _munge_halos,
    wrap_to_local_volume_inplace,
    find_and_write_most_massive_hosts
)

from diffsmhm.tests.testing_catalogs.make_mock_halos import make_test_catalogs

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1


@pytest.mark.mpi_skip
def test_wrap_to_local_volume_inplace():
    pos = np.array([0, 10, 30, 9, 20])
    cen = 9
    lbox = 20
    wrap_to_local_volume_inplace(pos, cen, lbox)
    assert_allclose(pos, np.array([-9, 1, 1, 0, -9]))


@pytest.mark.mpi_skip
def test_compute_host_index():
    halos = {
        "upid": np.array([-1, -1, 10, 3, -1, 5, 5, 5], dtype=np.int64),
        "halo_id": np.array([3, 10, 11, 12, 5, 13, 14, 15], dtype=np.int64),
    }

    host_index = _compute_host_index(halos)

    assert np.array_equal(
        host_index, np.array([0, 1, 1, 0, 4, 4, 4, 4], dtype=np.int64)
    )


@pytest.mark.mpi_skip
def test_munge_halos():
    rng = np.random.RandomState(seed=10)
    halos = dict()
    orig_halos = dict()
    for key in ["mpeak", "host_mpeak", "vmax_frac", "x", "y", "z"]:
        halos[key] = rng.uniform(size=3)
        orig_halos[key] = halos[key].copy()

    _munge_halos(halos)

    assert "x" not in halos
    assert "y" not in halos
    assert "z" not in halos
    assert np.array_equal(halos["logmpeak"], np.log10(orig_halos["mpeak"]))
    assert np.array_equal(halos["loghost_mpeak"], np.log10(orig_halos["host_mpeak"]))
    assert np.array_equal(halos["logvmax_frac"], np.log10(orig_halos["vmax_frac"]))
    assert np.array_equal(halos["halo_x"], orig_halos["x"])
    assert np.array_equal(halos["halo_y"], orig_halos["y"])
    assert np.array_equal(halos["halo_z"], orig_halos["z"])


@pytest.mark.mpi_skip
def test_find_and_write_most_massive_hosts_tree_walk():
    if RANK == 0:
        make_test_catalogs(1)
    COMM.Barrier()

    script_dir = os.path.realpath(os.path.dirname(__file__))
    testfile = script_dir+"/testing_catalogs/mock_halos_1.h5"
    mmhid, mmh_x, mmh_y, mmh_z, mmh_dist = find_and_write_most_massive_hosts(
                                                testfile, export=True
                                           )

    expected_mmhid = [-1, 1, 1, 1]

    expected_mmh_x = [10, 10, 10, 10]
    expected_mmh_y = [-10, -10, -10, -10]
    expected_mmh_z = [10, 10, 10, 10]

    if RANK == 0:
        with h5py.File(testfile, "r") as f:
            expected_mmh_dist = np.sqrt(np.power(f["x"][...] - expected_mmh_x, 2) +
                                        np.power(f["y"][...] - expected_mmh_y, 2) +
                                        np.power(f["z"][...] - expected_mmh_z, 2)
                                        )

        ok = True

        try:
            assert np.allclose(expected_mmhid, mmhid)

            assert np.allclose(expected_mmh_x, mmh_x)
            assert np.allclose(expected_mmh_y, mmh_y)
            assert np.allclose(expected_mmh_z, mmh_z)

            assert np.allclose(expected_mmh_dist, mmh_dist)

            f = h5py.File(testfile, "r")
            assert np.allclose(f["mmhid"], expected_mmhid)
            assert np.allclose(f["mmh_x"], expected_mmh_x)
            assert np.allclose(f["mmh_y"], expected_mmh_y)
            assert np.allclose(f["mmh_z"], expected_mmh_z)
            assert np.allclose(f["mmh_dist"], expected_mmh_dist)
            f.close()

            os.remove(testfile)

        except AssertionError:
            ok = False
    else:
        ok = None

    ok = COMM.bcast(ok, root=0)
    assert ok, "Tests Failed - see rank 0 for details."


@pytest.mark.mpi_skip
def test_find_and_write_most_massive_hosts_pid_host():
    if RANK == 0:
        make_test_catalogs(2)
    COMM.Barrier()

    script_dir = os.path.realpath(os.path.dirname(__file__))
    testfile = script_dir+"/testing_catalogs/mock_halos_2.h5"
    mmhid, mmh_x, mmh_y, mmh_z, mmh_dist = find_and_write_most_massive_hosts(
                                                testfile, export=True
                                           )

    expected_mmhid = [-1, 1, 1]

    expected_mmh_x = [10, 10, 10]
    expected_mmh_y = [10, 10, 10]
    expected_mmh_z = [10, 10, 10]

    if RANK == 0:
        with h5py.File(testfile, "r") as f:
            expected_mmh_dist = np.sqrt(np.power(f["x"][...] - expected_mmh_x, 2) +
                                        np.power(f["y"][...] - expected_mmh_y, 2) +
                                        np.power(f["z"][...] - expected_mmh_z, 2)
                                        )

        ok = True

        try:
            assert np.allclose(expected_mmhid, mmhid)

            assert np.allclose(expected_mmh_x, mmh_x)
            assert np.allclose(expected_mmh_y, mmh_y)
            assert np.allclose(expected_mmh_z, mmh_z)

            assert np.allclose(expected_mmh_dist, mmh_dist)

            f = h5py.File(testfile, "r")
            assert np.allclose(f["mmhid"], expected_mmhid)
            assert np.allclose(f["mmh_x"], expected_mmh_x)
            assert np.allclose(f["mmh_y"], expected_mmh_y)
            assert np.allclose(f["mmh_z"], expected_mmh_z)
            assert np.allclose(f["mmh_dist"], expected_mmh_dist)
            f.close()

            os.remove(testfile)

        except AssertionError:
            ok = False
    else:
        ok = None

    ok = COMM.bcast(ok, root=0)
    assert ok, "Test Failed - see rank 0 for details"


def test_find_and_write_most_massive_hosts_2_structs():
    if RANK == 0:
        make_test_catalogs(3)
    COMM.Barrier()

    script_dir = os.path.realpath(os.path.dirname(__file__))
    testfile = script_dir+"/testing_catalogs/mock_halos_3.h5"

    mmhid, mmh_x, mmh_y, mmh_z, mmh_dist = find_and_write_most_massive_hosts(
                                                testfile, export=False
                                           )

    expected_mmhid = [-1, 1, 1, 1, 1, 1, -1, 7, 7, 7]

    expected_mmh_x = [10, 10, 10, 10, 10, 10, -10, -10, -10, -10]
    expected_mmh_y = [11, 11, 11, 11, 11, 11, -11, -11, -11, -11]
    expected_mmh_z = [12, 12, 12, 12, 12, 12, -12, -12, -12, -12]

    if RANK == 0:
        with h5py.File(testfile, "r") as f:
            expected_mmh_dist = np.sqrt(np.power(f["x"][...] - expected_mmh_x, 2) +
                                        np.power(f["y"][...] - expected_mmh_y, 2) +
                                        np.power(f["z"][...] - expected_mmh_z, 2)
                                        )

        ok = True

        try:
            assert np.allclose(expected_mmhid, mmhid)

            assert np.allclose(expected_mmh_x, mmh_x)
            assert np.allclose(expected_mmh_y, mmh_y)
            assert np.allclose(expected_mmh_z, mmh_z)

            assert np.allclose(expected_mmh_dist, mmh_dist)

            os.remove(testfile)

        except AssertionError:
            ok = False
    else:
        ok = None

    ok = COMM.bcast(ok, root=0)
    assert ok, "Test Failed - see rank 0 for details."


def test_find_and_write_most_massive_hosts_multiple_pid():
    if RANK == 0:
        make_test_catalogs(4)
    COMM.Barrier()

    script_dir = os.path.realpath(os.path.dirname(__file__))
    testfile = script_dir + "/testing_catalogs/mock_halos_4.h5"

    mmhid, mmh_x, mmh_y, mmh_z, mmh_dist = find_and_write_most_massive_hosts(
                                                testfile, export=False
                                           )

    expected_mmhid = [-1, 1, -1, 1, 3]

    expected_mmh_x = [2, 2, 6, 2, 6]
    expected_mmh_y = [7, 7, 9, 7, 9]
    expected_mmh_z = [1, 1, 8, 1, 8]

    if RANK == 0:
        with h5py.File(testfile, "r") as f:
            expected_mmh_dist = np.sqrt(np.power(f["x"][...] - expected_mmh_x, 2) +
                                        np.power(f["y"][...] - expected_mmh_y, 2) +
                                        np.power(f["z"][...] - expected_mmh_z, 2)
                                        )

        ok = True

        try:
            assert np.allclose(expected_mmhid, mmhid)

            assert np.allclose(expected_mmh_x, mmh_x)
            assert np.allclose(expected_mmh_y, mmh_y)
            assert np.allclose(expected_mmh_z, mmh_z)

            assert np.allclose(expected_mmh_dist, mmh_dist)

            os.remove(testfile)

        except AssertionError:
            ok = False
    else:
        ok = None

    ok = COMM.bcast(ok, root=0)
    assert ok, "Test Failed - see rank 0 for details."


def test_find_and_write_most_massive_hosts_loop():
    if RANK == 0:
        make_test_catalogs(5)
    COMM.Barrier()

    script_dir = os.path.realpath(os.path.dirname(__file__))
    testfile = script_dir+"/testing_catalogs/mock_halos_5.h5"

    mmhid, mmh_x, mmh_y, mmh_z, mmh_dist = find_and_write_most_massive_hosts(
                                                testfile, export=False
                                           )

    expected_mmhid = [-1, 1, -1, 3, 3]

    if RANK == 0:
        ok = True
        try:
            assert np.allclose(expected_mmhid, mmhid)
            os.remove(testfile)
        except AssertionError:
            ok = False
    else:
        ok = None

    ok = COMM.bcast(ok, root=0)
    assert ok, "Test Failed - see rank 0 for details."
