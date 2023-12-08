import numpy as np
import h5py

try:
    from mpi4py import MPI
    
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1


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
        expected_mmh_z =  [10, 10, 10, 10]

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


def make_test_catalogs_loader(tdir, n_halos, n_parts, boxsize, mmh_dist):
    # define filenames 
    halo_file = tdir+"/halos.hdf5"
    part_file = tdir+"/parts.hdf5"

    if RANK == 0:
        # necesary fields are id, pos, mmh & pos, mpeak, host mpeak, vmax frac
        halo_id = np.arange(1, n_halos+1)
        
        halo_x = np.random.uniform(0.0, boxsize, size=n_halos)
        halo_y = np.random.uniform(0.0, boxsize, size=n_halos)
        halo_z = np.random.uniform(0.0, boxsize, size=n_halos)

        mmhid = np.zeros_like(halo_x, dtype=np.double)
        mmh_x = np.zeros_like(halo_x, dtype=np.double)
        mmh_y = np.zeros_like(halo_x, dtype=np.double)
        mmh_z = np.zeros_like(halo_x, dtype=np.double)

        assert len(halo_id) == len(mmhid)
        print(RANK, 'LEN CHECK DONE', flush=True)

        # quantities that have logs calculated 
        mpeak = np.ones_like(halo_x, dtype=np.double)
        host_mpeak = np.ones_like(halo_x, dtype=np.double)
        vmax_frac = np.ones_like(halo_x, dtype=np.double)


        # sort of randomly assign mmh 
        for i, h in enumerate(halo_id):
            if mmhid[i] > 0: 
                continue 

            # get distances
            dist_to_h = np.sqrt( np.power(halo_x[i] - halo_x, 2) + 
                                 np.power(halo_y[i] - halo_y, 2) + 
                                 np.power(halo_z[i] - halo_z, 2) )

            # mask distances
            dist_mask = dist_to_h < mmh_dist

            # mask gals without an mmh already assigned within dist 
            mmhid_nearby_mask = mmhid[dist_mask] != 0

            # assign to this halo
            mmhid[dist_mask][mmhid_nearby_mask] = h
            mmhid[i] = -1

            mmh_x[dist_mask][mmhid_nearby_mask] = halo_x[i]
            mmh_y[dist_mask][mmhid_nearby_mask] = halo_y[i]
            mmh_z[dist_mask][mmhid_nearby_mask] = halo_z[i]

        # write to file 
        with h5py.File(halo_file, "w") as f:
            f.create_dataset("halo_id", data=halo_id, dtype="i8")

            f.create_dataset("x", data=halo_x, dtype="f4")
            f.create_dataset("y", data=halo_y, dtype="f4")
            f.create_dataset("z", data=halo_z, dtype="f4")

            f.create_dataset("mmhid", data=mmhid, dtype="i8")
            f.create_dataset("mmh_x", data=mmh_x, dtype="f4")
            f.create_dataset("mmh_y", data=mmh_y, dtype="f4")
            f.create_dataset("mmh_z", data=mmh_z, dtype="f4")

            f.create_dataset("mpeak", data=mpeak, dtype="f4")
            f.create_dataset("host_mpeak", data=host_mpeak, dtype="f4")
            f.create_dataset("vmax_frac", data=vmax_frac, dtype="f4")

            # these quantities won't be used but need to be present
            other_keys = ["vx", "vy", "vz", "upid", "host_x", "host_y", "host_z", "host_dist"]
            for key in other_keys:
                f.create_dataset(key, data=np.zeros_like(halo_x), dtype="f4")


        # temp 
        with h5py.File(halo_file, "r") as f:
            print(f.keys())


        # now the particle catalog; just needs positions and id 
        particle_id = np.arange(1, n_parts+1)

        particle_x = np.random.uniform(0.0, boxsize, size=n_parts)
        particle_y = np.random.uniform(0.0, boxsize, size=n_parts)
        particle_z = np.random.uniform(0.0, boxsize, size=n_parts)

        # write to file 
        with h5py.File(part_file, "w") as f:
            f.create_dataset("part_id", data=particle_id, dtype="i8")

            f.create_dataset("x", data=particle_x, dtype="f4")
            f.create_dataset("y", data=particle_y, dtype="f4")
            f.create_dataset("z", data=particle_z, dtype="f4")

    # return filepaths 
    return halo_file, part_file
        
