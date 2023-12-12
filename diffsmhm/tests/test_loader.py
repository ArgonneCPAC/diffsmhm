import numpy as np
from numpy.testing import assert_allclose
import pytest

import h5py
import tempfile

from diffsmhm.loader import (
    _compute_host_index,
    wrap_to_local_volume_inplace,
    find_and_write_most_massive_hosts,
    load_and_chop_data_bolshoi_planck
)

from diffsmhm.tests.make_mock_halos import (
    make_test_catalogs_find_and_write,
    make_test_catalogs_loader_with_mmh,
    make_test_catalogs_loader_without_mmh
)

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
def test_find_and_write_most_massive_hosts_tree_walk():
    # make tempdir
    with tempfile.TemporaryDirectory() as tdir:
        if RANK == 0:
            make_test_catalogs_find_and_write(tdir, "tree_walk")
        COMM.Barrier()

        testfile = tdir+"/mock_halos_tree_walk.h5"
        mmhid, mmh_x, mmh_y, mmh_z, mmh_dist = find_and_write_most_massive_hosts(
                                                    testfile, export=True
                                               )

        if RANK == 0:
            with h5py.File(testfile, "r") as f:
                expected_mmhid = f["expected_mmhid"][...]

                expected_mmh_x = f["expected_mmh_x"][...]
                expected_mmh_y = f["expected_mmh_y"][...]
                expected_mmh_z = f["expected_mmh_z"][...]

                expected_mmh_dist = f["expected_mmh_dist"][...]

            ok = True

            try:
                assert np.allclose(expected_mmhid, mmhid)

                assert np.allclose(expected_mmh_x, mmh_x)
                assert np.allclose(expected_mmh_y, mmh_y)
                assert np.allclose(expected_mmh_z, mmh_z)

                assert np.allclose(expected_mmh_dist, mmh_dist)

                with h5py.File(testfile, "r") as f:
                    assert np.allclose(f["mmhid"], expected_mmhid)
                    assert np.allclose(f["mmh_x"], expected_mmh_x)
                    assert np.allclose(f["mmh_y"], expected_mmh_y)
                    assert np.allclose(f["mmh_z"], expected_mmh_z)
                    assert np.allclose(f["mmh_dist"], expected_mmh_dist)

            except AssertionError:
                ok = False
        else:
            ok = None

        ok = COMM.bcast(ok, root=0)
        assert ok, "Tests Failed - see rank 0 for details."


@pytest.mark.mpi_skip
def test_find_and_write_most_massive_hosts_upid_reassign():
    # make tempdir
    with tempfile.TemporaryDirectory() as tdir:
        if RANK == 0:
            make_test_catalogs_find_and_write(tdir, "upid_reassign")
        COMM.Barrier()

        testfile = tdir+"/mock_halos_upid_reassign.h5"
        mmhid, mmh_x, mmh_y, mmh_z, mmh_dist = find_and_write_most_massive_hosts(
                                                testfile, export=True
                                               )

        if RANK == 0:
            with h5py.File(testfile, "r") as f:
                expected_mmhid = f["expected_mmhid"][...]

                expected_mmh_x = f["expected_mmh_x"][...]
                expected_mmh_y = f["expected_mmh_y"][...]
                expected_mmh_z = f["expected_mmh_z"][...]

                expected_mmh_dist = f["expected_mmh_dist"][...]

            ok = True

            try:
                assert np.allclose(expected_mmhid, mmhid)

                assert np.allclose(expected_mmh_x, mmh_x)
                assert np.allclose(expected_mmh_y, mmh_y)
                assert np.allclose(expected_mmh_z, mmh_z)

                assert np.allclose(expected_mmh_dist, mmh_dist)

                with h5py.File(testfile, "r") as f:
                    assert np.allclose(f["mmhid"], expected_mmhid)
                    assert np.allclose(f["mmh_x"], expected_mmh_x)
                    assert np.allclose(f["mmh_y"], expected_mmh_y)
                    assert np.allclose(f["mmh_z"], expected_mmh_z)
                    assert np.allclose(f["mmh_dist"], expected_mmh_dist)

            except AssertionError:
                ok = False
        else:
            ok = None

        ok = COMM.bcast(ok, root=0)
        assert ok, "Test Failed - see rank 0 for details"


@pytest.mark.mpi_skip
def test_find_and_write_most_massive_hosts_2_structs():
    # make temp dir
    with tempfile.TemporaryDirectory() as tdir:
        if RANK == 0:
            make_test_catalogs_find_and_write(tdir, "two_structs")
        COMM.Barrier()

        testfile = tdir+"/mock_halos_two_structs.h5"

        mmhid, mmh_x, mmh_y, mmh_z, mmh_dist = find_and_write_most_massive_hosts(
                                                    testfile, export=False
                                               )

        if RANK == 0:
            with h5py.File(testfile, "r") as f:
                expected_mmhid = f["expected_mmhid"][...]

                expected_mmh_x = f["expected_mmh_x"][...]
                expected_mmh_y = f["expected_mmh_y"][...]
                expected_mmh_z = f["expected_mmh_z"][...]

                expected_mmh_dist = f["expected_mmh_dist"][...]

            ok = True

            try:
                assert np.allclose(expected_mmhid, mmhid)

                assert np.allclose(expected_mmh_x, mmh_x)
                assert np.allclose(expected_mmh_y, mmh_y)
                assert np.allclose(expected_mmh_z, mmh_z)

                assert np.allclose(expected_mmh_dist, mmh_dist)

            except AssertionError:
                ok = False
        else:
            ok = None

        ok = COMM.bcast(ok, root=0)
        assert ok, "Test Failed - see rank 0 for details."


@pytest.mark.mpi_skip
def test_find_and_write_most_massive_hosts_multiple_pid():
    with tempfile.TemporaryDirectory() as tdir:
        if RANK == 0:
            make_test_catalogs_find_and_write(tdir, "mult_pid")
        COMM.Barrier()

        testfile = tdir + "/mock_halos_mult_subs.h5"

        mmhid, mmh_x, mmh_y, mmh_z, mmh_dist = find_and_write_most_massive_hosts(
                                                    testfile, export=False
                                               )

        if RANK == 0:
            with h5py.File(testfile, "r") as f:
                expected_mmhid = f["expected_mmhid"][...]

                expected_mmh_x = f["expected_mmh_x"][...]
                expected_mmh_y = f["expected_mmh_y"][...]
                expected_mmh_z = f["expected_mmh_z"][...]

                expected_mmh_dist = f["expected_mmh_dist"][...]

            ok = True

            try:
                assert np.allclose(expected_mmhid, mmhid)

                assert np.allclose(expected_mmh_x, mmh_x)
                assert np.allclose(expected_mmh_y, mmh_y)
                assert np.allclose(expected_mmh_z, mmh_z)

                assert np.allclose(expected_mmh_dist, mmh_dist)

            except AssertionError:
                ok = False
        else:
            ok = None

        ok = COMM.bcast(ok, root=0)
        assert ok, "Test Failed - see rank 0 for details."


@pytest.mark.mpi_skip
def test_find_and_write_most_massive_hosts_loop():
    with tempfile.TemporaryDirectory() as tdir:
        if RANK == 0:
            make_test_catalogs_find_and_write(tdir, "loop")
        COMM.Barrier()

        testfile = tdir+"/mock_halos_loop.h5"

        mmhid, mmh_x, mmh_y, mmh_z, mmh_dist = find_and_write_most_massive_hosts(
                                                    testfile, export=False
                                               )

        if RANK == 0:
            with h5py.File(testfile, "r") as f:
                expected_mmhid = f["expected_mmhid"][...]

                expected_mmh_x = f["expected_mmh_x"][...]
                expected_mmh_y = f["expected_mmh_y"][...]
                expected_mmh_z = f["expected_mmh_z"][...]

                expected_mmh_dist = f["expected_mmh_dist"][...]
            ok = True
            try:
                assert np.allclose(expected_mmhid, mmhid)

                assert np.allclose(expected_mmh_x, mmh_x)
                assert np.allclose(expected_mmh_y, mmh_y)
                assert np.allclose(expected_mmh_z, mmh_z)

                assert np.allclose(expected_mmh_dist, mmh_dist)
            except AssertionError:
                ok = False
        else:
            ok = None

        ok = COMM.bcast(ok, root=0)
        assert ok, "Test Failed - see rank 0 for details."


@pytest.mark.mpi
def test_load_and_chop_data_bolshoi_planck_mmh_known():
    # test parameters
    n_halos = 1000
    n_parts = 10000
    boxsize = 250.0
    mmh_dist = 10.0

    # make temp dir
    with tempfile.TemporaryDirectory() as tdir:

        halofile, partfile = make_test_catalogs_loader_with_mmh(
                                                       tdir,
                                                       n_halos, n_parts,
                                                       boxsize, mmh_dist
                                                               )

        halofile = COMM.bcast(halofile, root=0)
        partfile = COMM.bcast(partfile, root=0)

        halos, particles = load_and_chop_data_bolshoi_planck(
                                                part_file=partfile,
                                                halo_file=halofile,
                                                box_length=boxsize,
                                                buff_wprp=mmh_dist)

        # check logs were calculated
        assert_allclose(halos["logmpeak"], np.log10(np.ones_like(halos["mpeak"])))
        assert_allclose(halos["loghost_mpeak"],
                        np.log10(np.ones_like(halos["host_mpeak"])))
        assert_allclose(halos["logvmax_frac"],
                        np.log10(np.ones_like(halos["vmax_frac"])))

        # check change of "x"/"y"/"z" to "halo_x"/"halo_y"/"halo_z"
        assert "x" not in halos.keys()
        assert "y" not in halos.keys()
        assert "z" not in halos.keys()

        assert "halo_x" in halos.keys()
        assert "halo_y" in halos.keys()
        assert "halo_x" in halos.keys()

        # check structure overload
        with h5py.File(halofile, "r") as f:
            halo_id_all = np.array(f["halo_id"][...], dtype="i8")
            halo_mmhid_all = np.array(f["mmhid"][...], dtype="i8")

        present_hosts = np.unique(halos["mmhid"])
        needed_subs = halo_id_all[np.isin(halo_mmhid_all, present_hosts)]

        missing_subs = np.setdiff1d(needed_subs, halos["halo_id"])
        assert len(missing_subs) == 0

        # particles
        # check id created
        assert "part_id" in particles.keys()

        assert "x" in particles.keys()
        assert "y" in particles.keys()
        assert "z" in particles.keys()


@pytest.mark.mpi
def test_load_and_chop_data_bolshoi_planck_mmh_unknown():
    # test parameters
    n_structs = 10
    n_parts = 10000
    boxsize = 250.0
    ol = 10.0

    # generate catalog
    with tempfile.TemporaryDirectory() as tdir:
        halofile, partfile = "", ""
        if RANK == 0:
            halofile, partfile = make_test_catalogs_loader_without_mmh(
                                    tdir,
                                    n_structs, n_parts,
                                    boxsize
                                 )

        COMM.Barrier()

        halofile = COMM.bcast(halofile, root=0)
        partfile = COMM.bcast(partfile, root=0)

        # do load_and_chop
        halos, particles = load_and_chop_data_bolshoi_planck(
                               part_file=partfile,
                               halo_file=halofile,
                               box_length=boxsize,
                               buff_wprp=ol
                           )

        # check mmhid's are correct and structure overload
        with h5py.File(halofile, "r") as f:
            assert_allclose(f["mmhid"][...], f["exp_mmhid"][...])
            assert_allclose(f["mmh_x"][...], f["exp_mmh_x"][...])

            halo_id_all = np.array(f["halo_id"][...], dtype="i8")
            halo_mmhid_all = np.array(f["mmhid"][...], dtype="i8")

        present_hosts = np.unique(halos["mmhid"])
        needed_subs = halo_id_all[np.isin(halo_mmhid_all, present_hosts)]

        missing_subs = np.setdiff1d(needed_subs, halos["halo_id"])
        assert len(missing_subs) == 0
