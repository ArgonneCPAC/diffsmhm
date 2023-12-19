import numpy as np
import h5py

from collections import OrderedDict

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


def _write_file(fname, hid, upid, pid,
                mpeak, host_mpeak,
                x, y, z, hx, hy, hz,
                exp_mmhid,
                exp_mmh_x, exp_mmh_y, exp_mmh_z
                ):

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

        host_dist = np.sqrt(np.power(x - hx, 2) +
                            np.power(y - hy, 2) +
                            np.power(z - hz, 2)
                            )

        f.create_dataset("host_dist", data=host_dist)

        f.create_dataset("expected_mmhid", data=exp_mmhid)

        f.create_dataset("expected_mmh_x", data=exp_mmh_x)
        f.create_dataset("expected_mmh_y", data=exp_mmh_y)
        f.create_dataset("expected_mmh_z", data=exp_mmh_z)

        exp_mmh_dist = np.sqrt(np.power(x - exp_mmh_x, 2) +
                               np.power(y - exp_mmh_y, 2) +
                               np.power(z - exp_mmh_z, 2)
                               )

        f.create_dataset("expected_mmh_dist", data=exp_mmh_dist)


def make_test_catalogs_find_and_write(tdir, t):
    filenames = [
                    tdir+"/mock_halos_tree_walk.h5",
                    tdir+"/mock_halos_upid_reassign.h5",
                    tdir+"/mock_halos_two_structs.h5",
                    tdir+"/mock_halos_mult_subs.h5",
                    tdir+"/mock_halos_loop.h5"
                ]

    if t == "tree_walk":
        # testing only tree walk
        # ie following upid chain to find most massive host
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

        expected_mmhid = [-1, 1, 1, 1]

        expected_mmh_x = [10, 10, 10, 10]
        expected_mmh_y = [-10, -10, -10, -10]
        expected_mmh_z = [10, 10, 10, 10]

        _write_file(filenames[0], mock_halo_id, mock_upid, mock_pid,
                    mock_mpeak, mock_host_mpeak,
                    mock_x, mock_y, mock_z,
                    mock_host_x, mock_host_y, mock_host_z,
                    expected_mmhid,
                    expected_mmh_x, expected_mmh_y, expected_mmh_z
                    )

    if t == "upid_reassign":
        # testing only upid reassignment
        # ie when a halo identified as independent should be associated with a host
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

        expected_mmhid = [-1, 1, 1]

        expected_mmh_x = [10, 10, 10]
        expected_mmh_y = [10, 10, 10]
        expected_mmh_z = [10, 10, 10]

        _write_file(filenames[1], mock_halo_id, mock_upid, mock_pid,
                    mock_mpeak, mock_host_mpeak,
                    mock_x, mock_y, mock_z,
                    mock_host_x, mock_host_y, mock_host_z,
                    expected_mmhid,
                    expected_mmh_x, expected_mmh_y, expected_mmh_z
                    )

    if t == "two_structs":
        # test tree walk and upid correction with two separate structures
        # primary purpose of this test is to split across mpi ranks
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

        expected_mmhid = [-1, 1, 1, 1, 1, 1, -1, 7, 7, 7]

        expected_mmh_x = [10, 10, 10, 10, 10, 10, -10, -10, -10, -10]
        expected_mmh_y = [11, 11, 11, 11, 11, 11, -11, -11, -11, -11]
        expected_mmh_z = [12, 12, 12, 12, 12, 12, -12, -12, -12, -12]

        _write_file(filenames[2], mock_halo_id, mock_upid, mock_pid,
                    mock_mpeak, mock_host_mpeak,
                    mock_x, mock_y, mock_z,
                    mock_host_x, mock_host_y, mock_host_z,
                    expected_mmhid,
                    expected_mmh_x, expected_mmh_y, expected_mmh_z
                    )

    if t == "mult_pid":
        # test multiple galaxies having a host as their pid
        # this ensures the code selects the correct halo as the new upid
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

        expected_mmhid = [-1, 1, -1, 1, 3]

        expected_mmh_x = [2, 2, 6, 2, 6]
        expected_mmh_y = [7, 7, 9, 7, 9]
        expected_mmh_z = [1, 1, 8, 1, 8]

        _write_file(filenames[3], mock_halo_id, mock_upid, mock_pid,
                    mock_mpeak, mock_host_mpeak,
                    mock_x, mock_y, mock_z,
                    mock_host_x, mock_host_y, mock_host_z,
                    expected_mmhid,
                    expected_mmh_x, expected_mmh_y, expected_mmh_z
                    )

    if t == "loop":
        # test against loop conditions:
        # 1) subhalo has the same host as it's upid and pid (here halos 1-2)
        # 2) One subhalo has another as it's upid (here halos 3-5)
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

        expected_mmhid = [-1, 1, -1, 3, 3]

        expected_mmh_x = [1, 1, 3, 3, 3]
        expected_mmh_y = [1, 1, 1, 1, 1]
        expected_mmh_z = [1, 1, 1, 1, 1]

        _write_file(filenames[4], mock_halo_id, mock_upid, mock_pid,
                    mock_mpeak, mock_host_mpeak,
                    mock_x, mock_y, mock_z,
                    mock_host_x, mock_host_y, mock_host_z,
                    expected_mmhid,
                    expected_mmh_x, expected_mmh_y, expected_mmh_z
                    )


def make_test_catalogs_loader_with_mmh(tdir, n_halos, n_parts, boxsize, mmh_dist):
    # define filenames
    halo_file = tdir+"/halos.hdf5"
    part_file = tdir+"/parts.hdf5"

    # setup partition and generation parameters
    labels = "xyz"
    n_struct = int(n_halos/4)
    partition = mpipartition.Partition()

    np.random.seed(RANK)

    # generate data within our partition
    halos = {
        x: np.random.uniform(0, 1, n_halos) * partition.extent[i] + partition.origin[i]
        for i, x in enumerate(labels)
    }
    halos["x"] *= boxsize
    halos["y"] *= boxsize
    halos["z"] *= boxsize

    # unique ID
    halos["halo_id"] = np.arange(n_halos, dtype="i8") + RANK * n_halos
    # mmh tag
    halos["mmhid"] = np.random.randint(0, n_struct, n_halos) + RANK * n_halos

    halos["mmh_x"] = halos["x"][halos["mmhid"] - RANK * n_halos]
    halos["mmh_y"] = halos["y"][halos["mmhid"] - RANK * n_halos]
    halos["mmh_z"] = halos["z"][halos["mmhid"] - RANK * n_halos]

    halos["mmhid"][0:n_struct] = -1

    # gather for writing to file
    all_halos = OrderedDict()
    for k in halos.keys():
        all_halos[k] = np.concatenate(partition.comm.allgather(halos[k]))
        assert len(all_halos[k]) == n_halos * N_RANKS

    if RANK == 0:
        # necesary fields are id, pos, mmh & pos, mpeak, host mpeak, vmax frac
        # we need to add mpeak, host_mpeak, vmax_frac

        # quantities that have logs calculated
        mpeak = np.ones_like(all_halos["x"], dtype=np.double)
        host_mpeak = np.ones_like(all_halos["x"], dtype=np.double)
        vmax_frac = np.ones_like(all_halos["x"], dtype=np.double)

        # write to file
        with h5py.File(halo_file, "w") as f:

            f.create_dataset("x", data=all_halos["x"], dtype="f4")
            f.create_dataset("y", data=all_halos["y"], dtype="f4")
            f.create_dataset("z", data=all_halos["z"], dtype="f4")

            f.create_dataset("halo_id", data=all_halos["halo_id"], dtype="i8")

            f.create_dataset("mpeak", data=mpeak, dtype="f4")
            f.create_dataset("host_mpeak", data=host_mpeak, dtype="f4")
            f.create_dataset("vmax_frac", data=vmax_frac, dtype="f4")

            f.create_dataset("mmhid", data=all_halos["mmhid"], dtype="i8")
            f.create_dataset("mmh_x", data=all_halos["mmh_x"], dtype="f4")
            f.create_dataset("mmh_y", data=all_halos["mmh_y"], dtype="f4")
            f.create_dataset("mmh_z", data=all_halos["mmh_z"], dtype="f4")

            # these quantities won't be used but need to be present
            other_keys = ["vx", "vy", "vz",
                          "upid",
                          "host_x", "host_y", "host_z", "host_dist"]
            for key in other_keys:
                if key in ["upid"]:
                    dt = "i8"
                else:
                    dt = "f4"
                f.create_dataset(key, data=np.zeros_like(all_halos["x"]), dtype=dt)

        # now the particle catalog; just needs positions and id
        particle_x = np.random.uniform(0.0, boxsize, size=n_parts)
        particle_y = np.random.uniform(0.0, boxsize, size=n_parts)
        particle_z = np.random.uniform(0.0, boxsize, size=n_parts)

        # write to file
        with h5py.File(part_file, "w") as f:
            dg = f.create_group("data")

            dg.create_dataset("x", data=particle_x, dtype="f4")
            dg.create_dataset("y", data=particle_y, dtype="f4")
            dg.create_dataset("z", data=particle_z, dtype="f4")

    # return filepaths
    # only rank 0 returns a string bc tempdir will be different for each rank
    if RANK == 0:
        return halo_file, part_file
    else:
        return None, None


def make_test_catalogs_loader_without_mmh(tdir, n_clones, n_parts, boxsize):
    # filenames
    halo_file = tdir+"/halos.hdf5"
    part_file = tdir+"/parts.hdf5"

    # spawn a bunch of 7 halo structs that span ~100 units
    syst_size = 7

    # define system to clone
    syst_x = np.array([0.00, -45.0,  45.0,  0.00,  0.00,  0.00,  0.00], dtype="f4")
    syst_y = np.array([0.00,  0.00,  00.0, -45.0,  45.0,  0.00,  0.00], dtype="f4")
    syst_z = np.array([0.00,  0.00,  00.0,  0.00,  0.00, -45.0,  45.0], dtype="f4")

    syst_upid = np.array([-1, 1, -1, 1, 4, 2, 2], dtype="i8")
    syst_pid = np.array([-1, 1, -1, 3, 4, 2, 2], dtype="i8")

    syst_mpeak = np.array([4, 2, 3, 2, 1, 1, 1], dtype="f4")
    syst_host_mpeak = np.array([4, 4, 3, 4, 2, 2, 2], dtype="f4")

    syst_host_x = np.array([00.0,  00.0,  45.0,  00.0,  0.00, -45.0, -45.0], dtype="f4")
    syst_host_y = np.array([00.0,  00.0,  00.0,  00.0, -45.0,  00.0,  00.0], dtype="f4")
    syst_host_z = np.array([00.0,  00.0,  00.0,  00.0,  00.0,  00.0,  00.0], dtype="f4")

    syst_mmh_exp = np.array([-1, 1, 1, 1, 1, 1, 1], dtype="i8")

    # set up empty arrays to add systems to
    all_x = np.array([], dtype="f4")
    all_y = np.array([], dtype="f4")
    all_z = np.array([], dtype="f4")

    all_upid = np.array([], dtype="i8")
    all_pid = np.array([], dtype="i8")
    all_mmh_exp = np.array([], dtype="i8")

    # this is important for finding mmh and then distibuting properly
    all_host_x = np.array([], dtype="f4")
    all_host_y = np.array([], dtype="f4")
    all_host_z = np.array([], dtype="f4")

    all_mpeak = np.array([], dtype="i8")
    all_host_mpeak = np.array([], dtype="i8")

    all_mmh_exp = np.array([], dtype="i8")
    all_mmh_x_exp = np.array([], dtype="f4")

    # add clones to arrays
    for i in range(n_clones):
        x = np.random.uniform(0.0, boxsize)
        y = np.random.uniform(0.0, boxsize)
        z = np.random.uniform(0.0, boxsize)

        all_x = np.append(all_x, syst_x + x)
        all_y = np.append(all_y, syst_y + y)
        all_z = np.append(all_z, syst_z + z)

        all_mmh_x_exp = np.append(all_mmh_x_exp, np.full(syst_size, x, dtype="f4"))

        id_scale = syst_size * i

        upid_to_add = np.copy(syst_upid)
        upid_to_add[upid_to_add > 0] += id_scale
        all_upid = np.append(all_upid, upid_to_add)

        pid_to_add = np.copy(syst_pid)
        pid_to_add[pid_to_add > 0] += id_scale
        all_pid = np.append(all_pid, pid_to_add)

        all_host_x = np.append(all_host_x, syst_host_x + x)
        all_host_y = np.append(all_host_y, syst_host_y + y)
        all_host_z = np.append(all_host_z, syst_host_z + z)

        all_mmh_exp = np.append(all_mmh_exp, syst_mmh_exp)
        syst_mmh_exp[1:] += syst_size

        all_mpeak = np.append(all_mpeak, syst_mpeak)
        all_host_mpeak = np.append(all_host_mpeak, syst_host_mpeak)

    # id is done based on length
    all_halo_id = np.arange(1, n_clones*syst_size+1)

    # other fields that aren't used but are expected
    other_important_keys = ["vx", "vy", "vz", "vmax_frac", "host_dist"]

    # write to file
    with h5py.File(halo_file, "w") as f:
        f.create_dataset("exp_mmhid", data=all_mmh_exp, dtype="i8")
        f.create_dataset("exp_mmh_x", data=all_mmh_x_exp, dtype="f4")

        f.create_dataset("halo_id", data=all_halo_id, dtype="i8")
        f.create_dataset("upid", data=all_upid, dtype="i8")
        f.create_dataset("pid", data=all_pid, dtype="i8")

        f.create_dataset("x", data=all_x, dtype="f4")
        f.create_dataset("y", data=all_y, dtype="f4")
        f.create_dataset("z", data=all_z, dtype="f4")

        f.create_dataset("host_x", data=all_host_x, dtype="f4")
        f.create_dataset("host_y", data=all_host_y, dtype="f4")
        f.create_dataset("host_z", data=all_host_z, dtype="f4")

        f.create_dataset("mpeak", data=all_mpeak, dtype="f4")
        f.create_dataset("host_mpeak", data=all_host_mpeak, dtype="f4")

        for key in other_important_keys:
            f.create_dataset(key, data=np.ones_like(all_halo_id), dtype="f4")

    # particle catalog
    particle_x = np.random.uniform(0.0, boxsize, size=n_parts)
    particle_y = np.random.uniform(0.0, boxsize, size=n_parts)
    particle_z = np.random.uniform(0.0, boxsize, size=n_parts)

    # write to file
    with h5py.File(part_file, "w") as f:
        # particle catalogs have a top level group
        dg = f.create_group("data")

        dg.create_dataset("x", data=particle_x, dtype="f4")
        dg.create_dataset("y", data=particle_y, dtype="f4")
        dg.create_dataset("z", data=particle_z, dtype="f4")

    # return filepaths
    if RANK == 0:
        return halo_file, part_file
    else:
        return None, None
