import numpy as np
import h5py

import os


def _write_file(fname, hid, upid, pid,
                mpeak, host_mpeak,
                x, y, z, hx, hy, hz,
                host_dist):

    with h5py.File(fname, "w") as f:
        f.create_dataset("halo_id", data=hid)
        f.create_dataset("upid", data=upid)
        f.create_dataset("pid", data=pid)

        f.create_dataset("mpeak", data=mpeak)
        f.create_dataset("host_mpeak", data=host_mpeak)

        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("z", data=z)

        f.create_dataset("host_x", data=hx)
        f.create_dataset("host_y", data=hy)
        f.create_dataset("host_z", data=hz)

        f.create_dataset("host_dist", data=host_dist)


def make_test_catalogs(t):
    # define file names
    script_dir = os.path.realpath(os.path.dirname(__file__))
    filenames = [
                    script_dir+"/mock_halos_1.h5",
                    script_dir+"/mock_halos_2.h5",
                    script_dir+"/mock_halos_3.h5",
                    script_dir+"/mock_halos_4.h5",
                    script_dir+"/mock_halos_5.h5"
                ]

    # remove any existing files
    for fname in filenames:
        if os.path.exists(fname):
            os.remove(fname)

    if t == 1:
        # testing only tree walk
        mock_halo_id = np.array([1, 2, 3, 4], dtype="i8")
        mock_upid = np.array([-1, 1, 2, 3], dtype="i8")
        mock_pid = np.array([-1, -1, -1, -1], dtype="i8")

        mock_mpeak = 10**np.array([4, 3, 2, 1], dtype="f4")
        mock_host_mpeak = 10**np.array([4, 4, 3, 2], dtype="f4")

        mock_x = np.array([10, 1, 2, 3], dtype="f4")
        mock_y = np.array([-10, -1, -2, -3], dtype="f4")
        mock_z = np.array([10, 1, 2, 3], dtype="f4")

        mock_host_x = np.array([10, 10, 1, 2], dtype="f4")
        mock_host_y = np.array([-10, -10, -1, -2], dtype="f4")
        mock_host_z = np.array([10, 10, 1, 2], dtype="f4")

        mock_host_dist = np.sqrt(
                                np.power(mock_x - mock_host_x, 2) +
                                np.power(mock_y - mock_host_y, 2) +
                                np.power(mock_z - mock_host_z, 2)
                                )

        _write_file(filenames[0], mock_halo_id, mock_upid, mock_pid,
                    mock_mpeak, mock_host_mpeak,
                    mock_x, mock_y, mock_z,
                    mock_host_x, mock_host_y, mock_host_z,
                    mock_host_dist
                    )

    if t == 2:
        # testing only upid reassignment
        mock_halo_id = np.array([1, 2, 3], dtype="i8")
        mock_upid = np.array([-1, 1, -1], dtype="i8")
        mock_pid = np.array([-1, 3, -1], dtype="i8")

        mock_mpeak = 10**np.array([3, 1, 2], dtype="i8")
        mock_host_mpeak = 10**np.array([3, 3, 2], dtype="i8")

        mock_x = np.array([10, 1, 2], dtype="i8")
        mock_y = np.array([10, 1, 2], dtype="i8")
        mock_z = np.array([10, 1, 2], dtype="i8")

        mock_host_x = np.array([10, 10, 2], dtype="i8")
        mock_host_y = np.array([10, 10, 2], dtype="i8")
        mock_host_z = np.array([10, 10, 2], dtype="i8")

        mock_host_dist = np.sqrt(
                                np.power(mock_x - mock_host_x, 2) +
                                np.power(mock_y - mock_host_y, 2) +
                                np.power(mock_z - mock_host_z, 2)
                                )

        _write_file(filenames[1], mock_halo_id, mock_upid, mock_pid,
                    mock_mpeak, mock_host_mpeak,
                    mock_x, mock_y, mock_z,
                    mock_host_x, mock_host_y, mock_host_z,
                    mock_host_dist
                    )

    if t == 3:
        # test tree walk and upid correction with two separate structures
        mock_halo_id = np.array([1, 2, 3, 4, 5, 6,  7, 8, 9, 10], dtype="i8")
        mock_upid = np.array([-1, 1, -1, 1, 2, 1,  -1, 7, 8, -1], dtype="i8")
        mock_pid = np.array([-1, -1, -1, -1, -1, 3,  -1, -1, 10, -1], dtype="i8")

        mock_mpeak = 10**np.array([4, 2, 3, 2, 1, 2,  4, 3, 1, 2], dtype="i8")
        mock_host_mpeak = 10**np.array([4, 4, 3, 4, 2, 4,  4, 4, 3, 2], dtype="i8")

        mock_x = np.array([10, 2, 3, 4, 5, 6,  -10, -8, -9, -10], dtype="f4")
        mock_y = np.array([11, 2, 3, 4, 5, 6,  -11, -8, -9, -10], dtype="f4")
        mock_z = np.array([12, 2, 3, 4, 5, 6,  -12, -8, -9, -10], dtype="f4")

        mock_host_x = np.array([10, 10, 3, 10, 2, 10,  -10, -10, -8, -10], dtype="f4")
        mock_host_y = np.array([11, 11, 3, 11, 2, 11,  -11, -11, -8, -10], dtype="f4")
        mock_host_z = np.array([12, 12, 3, 12, 2, 12,  -12, -12, -8, -10], dtype="f4")

        mock_host_dist = np.sqrt(
                                    np.power(mock_x - mock_host_x, 2) +
                                    np.power(mock_y - mock_host_y, 2) +
                                    np.power(mock_z - mock_host_z, 2)
                                )

        _write_file(filenames[2], mock_halo_id, mock_upid, mock_pid,
                    mock_mpeak, mock_host_mpeak,
                    mock_x, mock_y, mock_z,
                    mock_host_x, mock_host_y, mock_host_z,
                    mock_host_dist
                    )

    if t == 4:
        # test multiple galaxies having a host as their pid
        mock_halo_id = np.array([1, 2, 3, 4, 5], dtype="i8")
        mock_upid = np.array([-1, -1, -1, 1, 3], dtype="i8")
        mock_pid = np.array([-1, -1, -1, 2, 2], dtype="i8")

        mock_mpeak = 10**np.array([5, 3, 4, 2, 2], dtype="i8")
        mock_host_mpeak = 10**np.array([5, 3, 4, 5, 4], dtype="i8")

        mock_x = np.array([2, 4, 6, 2, 6], dtype="f4")
        mock_y = np.array([7, 8, 9, 7, 9], dtype="f4")
        mock_z = np.array([1, 6, 8, 1, 8], dtype="f4")

        mock_host_x = np.array([2, 4, 6, 2, 6], dtype="f4")
        mock_host_y = np.array([7, 8, 9, 7, 9], dtype="f4")
        mock_host_z = np.array([1, 6, 8, 1, 8], dtype="f4")

        mock_host_dist = np.sqrt(
                                    np.power(mock_x - mock_host_x, 2) +
                                    np.power(mock_y - mock_host_y, 2) +
                                    np.power(mock_z - mock_host_z, 2)
                                )

        _write_file(filenames[3], mock_halo_id, mock_upid, mock_pid,
                    mock_mpeak, mock_host_mpeak,
                    mock_x, mock_y, mock_z,
                    mock_host_x, mock_host_y, mock_host_z,
                    mock_host_dist
                    )

    if t == 5:
        # test against loop conditions
        mock_halo_id = np.array([1, 2, 3, 4, 5], dtype="i8")
        mock_upid = np.array([-1, 1, -1, 3, 4], dtype="i8")
        mock_pid = np.array([-1, 1, -1, 3, 3], dtype="i8")

        mock_mpeak = 10**np.array([5, 2, 5, 4, 3], dtype="i8")
        mock_host_mpeak = 10**np.array([5, 5, 5, 5, 4], dtype="i8")

        mock_x = np.array([1, 1, 3, 2, 4], dtype="f4")
        mock_y = np.array([1, 2, 1, 2, 2], dtype="f4")
        mock_z = np.array([1, 1, 1, 1, 1], dtype="f4")

        mock_host_x = np.array([1, 1, 3, 3, 2], dtype="f4")
        mock_host_y = np.array([1, 1, 1, 1, 2], dtype="f4")
        mock_host_z = np.array([1, 1, 1, 1, 1], dtype="f4")

        mock_host_dist = np.sqrt(
                                    np.power(mock_x - mock_host_x, 2) +
                                    np.power(mock_y - mock_host_y, 2) +
                                    np.power(mock_z - mock_host_z, 2)
                                )

        _write_file(filenames[4], mock_halo_id, mock_upid, mock_pid,
                    mock_mpeak, mock_host_mpeak,
                    mock_x, mock_y, mock_z,
                    mock_host_x, mock_host_y, mock_host_z,
                    mock_host_dist
                    )
