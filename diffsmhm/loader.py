import numpy as np
from numba import njit

import sys
from collections import OrderedDict
import h5py

from diffsmhm.galhalo_models.crossmatch import crossmatch_integers
# from diffsmhm.galhalo_models.merging import _calculate_indx_to_deposit

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1

import mpipartition


def _compute_host_index(halos):
    orig_indices = np.arange(len(halos["halo_id"])).astype("i8")
    host_index = np.zeros_like(orig_indices)
    host_ids = np.where(halos["upid"] == -1, halos["halo_id"], halos["upid"])
    idxA, idxB = crossmatch_integers(host_ids, halos["halo_id"])
    host_index[idxA] = orig_indices[idxB]
    return host_index


def _munge_halos(halos):
    #  Compute some logs once and for all
    halos["logmpeak"] = np.log10(halos["mpeak"])
    halos["loghost_mpeak"] = np.log10(halos["host_mpeak"])
    halos["logvmax_frac"] = np.log10(halos["vmax_frac"])

    # we need to chop by host halo pos, so replace those
    halos["halo_x"] = halos["x"].copy()
    halos["halo_y"] = halos["y"].copy()
    halos["halo_z"] = halos["z"].copy()
    del halos["x"]
    del halos["y"]
    del halos["z"]


@njit()
def wrap_to_local_volume_inplace(pos, cen, lbox):
    """Wrap a set of points to the local volume inplace.

    This function wraps all points so that they are in a range [-lbox/2, lbox/2]
    around cen. Points right at lbox/2 from cen have undeterministic behavior and
    can end up on either side.

    Parameters
    ----------
    pos : np.ndarray
        Array of positions in periodic space of size lbox to be wrapped.
    cen : float
        The center of the local volume.
    lbox : float
        The length of the periodic space.
    """
    lbox_2 = lbox / 2.0
    n_lbox_2 = -lbox_2
    n = pos.shape[0]

    for i in range(n):
        dpos = pos[i] - cen

        while dpos < n_lbox_2:
            dpos += lbox

        while dpos > lbox_2:
            dpos -= lbox

        pos[i] = dpos


def find_and_write_most_massive_hosts(halo_file, host_mpeak_cut=0, export=True):
    """
    find_and_write_most_massive_hosts(...)
        Find most massive host associated with each halo and write to catalog

    Parameters
    ----------
    halo_file : str
        The path to the HDF5 file with the halo data.
    host_mpeak_cut : float
        Minimum host mass to load, used for testing.
    export : bool
        If True, mmhid and mmh_x/y/z are added to halo_file

    Returns
    -------
    mmhid : array like
        most massive host of a each halo; may be an "indirect" host.
    mmh_x : array like
        x position of each halo's most massive host
    mmh_y : array_like
        y position of each halo's most massive host
    mmh_z : array_like
        z position of each halo's most massive host
    mmh_dist : array_like
        distance between each halo and it's mmh
    """
    # note these are different from load_and_chop
    important_keys = ["halo_id", "upid", "pid", "mpeak", "x", "y", "z",
                      "host_x", "host_y", "host_z", "host_dist", "rvir"]

    # load hdf5 data
    halos = OrderedDict()
    with h5py.File(halo_file, "r") as hdf:
        _host_mpeak_mask = np.log10(hdf["host_mpeak"][...]) >= host_mpeak_cut
        for key in hdf.keys():
            # only keep the columns that we want
            if key not in important_keys:
                continue

            # integer types
            if key in ("halo_id", "upid", "pid"):
                dt = "i8"
            else:
                dt = "f4"
            halos[key] = hdf[key][...][_host_mpeak_mask].astype(dt)

    # 1) Point independent hosts that share a sub with a larger host to that larger host
    # only consider subhalos with pid != upid
    pid_upid_diff = halos["pid"][halos["pid"] != halos["upid"]]

    # get list of indices of "pid hosts"
    host_indices = np.where(halos["upid"] == -1)[0]
    pid_indices = np.where(np.isin(halos["halo_id"], pid_upid_diff))[0]
    pid_hosts_indices = np.intersect1d(host_indices, pid_indices)

    # now each mpi rank determines it's chunk of the list of "pid hosts"
    avg, rem = divmod(len(pid_hosts_indices), N_RANKS)
    rank_count = [avg + 1 if p < rem else avg for p in range(N_RANKS)]
    displ = [sum(rank_count[:p]) for p in range(N_RANKS)]

    rank_count = rank_count[RANK]
    displ = displ[RANK]

    pid_hosts_indices_rank = pid_hosts_indices[displ:displ+rank_count]

    # get id's and upid's for this rank
    pid_hosts_rank = np.copy(halos["halo_id"][pid_hosts_indices_rank])
    upid_corr_rank = np.copy(halos["upid"][pid_hosts_indices_rank])

    # determine and assign new upids for "pid hosts"
    for i, host in enumerate(pid_hosts_rank):
        mask_this_host = halos["halo_id"] == host

        # get all subs that have this host as their pid
        subs_this_pid = halos["halo_id"][halos["pid"] == host]

        # get all upids of host's subs
        upids_poss = np.unique(halos["upid"][np.isin(halos["halo_id"], subs_this_pid)])
        # remove any subs from the upid list (otherwise can get looping)
        upids_poss = np.setdiff1d(upids_poss, subs_this_pid)
        # remove host from the upid list (otherwise can get looping)
        upids_poss = np.setdiff1d(upids_poss, host)
        if len(upids_poss) == 0:
            continue

        # get most massive upid in upids_poss
        upid_masses = halos["mpeak"][np.isin(halos["halo_id"], upids_poss)]
        if np.max(upid_masses) > halos["mpeak"][mask_this_host][0]:
            upid_corr_rank[i] = upids_poss[np.argmax(upid_masses)]

    # now gather
    upid_corr_all = np.copy(halos["upid"])
    upid_updated = np.empty(len(pid_hosts_indices), dtype="i8")
    COMM.Allgatherv(upid_corr_rank, upid_updated)

    upid_corr_all[pid_hosts_indices] = upid_updated

    # 2) Find most massive host by "walking up tree"
    # find indices of all subs
    sub_indices = np.where(upid_corr_all != -1)[0]

    # again each mpi rank determines it's chunk of the work
    avg, rem = divmod(len(sub_indices), N_RANKS)
    rank_count = [avg + 1 if p < rem else avg for p in range(N_RANKS)]
    displ = [sum(rank_count[:p]) for p in range(N_RANKS)]

    rank_count = rank_count[RANK]
    displ = displ[RANK]

    sub_indices_rank = sub_indices[displ:displ+rank_count]

    # copy fields bc many halos will not be changed
    mmhid_rank = np.copy(upid_corr_all[sub_indices_rank])
    mmh_x = np.copy(halos["host_x"][sub_indices_rank])
    mmh_y = np.copy(halos["host_y"][sub_indices_rank])
    mmh_z = np.copy(halos["host_z"][sub_indices_rank])

    subs_rank = halos["halo_id"][sub_indices_rank]

    # iterate
    for i, sub in enumerate(subs_rank):
        current_upid = mmhid_rank[i]
        next_upid = upid_corr_all[halos["halo_id"] == current_upid][0]

        visited_upids = [current_upid]

        while next_upid != -1:
            # update this sub's mmh
            mmhid_rank[i] = next_upid

            # check for loop
            if next_upid in visited_upids:
                visited_upids.append(next_upid)
                sys.exit("Circular structure found in halo catalog.")
            visited_upids.append(next_upid)

            # get next upid
            current_upid = next_upid
            next_upid = upid_corr_all[halos["halo_id"] == current_upid][0]

        # update mmh_x/y/z
        mmh_x[i] = halos["x"][halos["halo_id"] == current_upid][0]
        mmh_y[i] = halos["y"][halos["halo_id"] == current_upid][0]
        mmh_z[i] = halos["z"][halos["halo_id"] == current_upid][0]

    # gather info for all subs
    # mmhid
    mmhid_allsubs = np.empty(len(sub_indices), dtype="i8")
    mmhid_all = np.copy(upid_corr_all)
    COMM.Allgatherv(mmhid_rank, mmhid_allsubs)

    mmhid_all[sub_indices] = mmhid_allsubs

    # mmh_x/y/z
    mmh_x_allsubs = np.empty(len(sub_indices), dtype="f4")
    mmh_y_allsubs = np.empty(len(sub_indices), dtype="f4")
    mmh_z_allsubs = np.empty(len(sub_indices), dtype="f4")

    mmh_x_all = np.copy(halos["host_x"])
    mmh_y_all = np.copy(halos["host_y"])
    mmh_z_all = np.copy(halos["host_z"])

    COMM.Allgatherv(mmh_x, mmh_x_allsubs)
    COMM.Allgatherv(mmh_y, mmh_y_allsubs)
    COMM.Allgatherv(mmh_z, mmh_z_allsubs)

    mmh_x_all[sub_indices] = mmh_x_allsubs
    mmh_y_all[sub_indices] = mmh_y_allsubs
    mmh_z_all[sub_indices] = mmh_z_allsubs

    # calc distance to mmh
    mmh_dist_rank = np.sqrt(
        np.power(mmh_x-halos["x"][sub_indices_rank], 2)
        + np.power(mmh_y-halos["y"][sub_indices_rank], 2)
        + np.power(mmh_z-halos["z"][sub_indices_rank], 2)
    )

    # mmh_dist
    mmh_dist_allsubs = np.empty(len(sub_indices), dtype="f4")
    mmh_dist_all = np.copy(halos["host_dist"])
    COMM.Allgatherv(mmh_dist_rank, mmh_dist_allsubs)

    mmh_dist_all[sub_indices] = mmh_dist_allsubs

    # rank 0 write to file
    if RANK == 0 and export:
        # write to file
        with h5py.File(halo_file, "a") as f:
            if "mmhid" in f.keys():
                del f["mmhid"]
            f.create_dataset("mmhid", data=mmhid_all, dtype="i8")

            if "mmh_x" in f.keys():
                del f["mmh_x"]
            f.create_dataset("mmh_x", data=mmh_x_all, dtype="f4")

            if "mmh_y" in f.keys():
                del f["mmh_y"]
            f.create_dataset("mmh_y", data=mmh_y_all, dtype="f4")

            if "mmh_z" in f.keys():
                del f["mmh_z"]
            f.create_dataset("mmh_z", data=mmh_z_all, dtype="f4")

            if "mmh_dist" in f.keys():
                del f["mmh_dist"]
            f.create_dataset("mmh_dist", data=mmh_dist_all, dtype="f4")

    return mmhid_all, mmh_x_all, mmh_y_all, mmh_z_all, mmh_dist_all


def load_and_chop_data_bolshoi_planck(
        part_file, halo_file, box_length, buff_wprp, host_mpeak_cut=0
):
    """Load and chop the data.

    Parameters
    ----------
    part_file : str
        The path to the HDF5 file with the particle data.
    halo_file : str
        The path to the HDF5 file with the halo data.
    box_length : float
        The length of the periodic volume.
    buff_wprp : float
        The buffer length to use for wp(rp).
    host_mpeak_cut : float
        The cut in host Mpeak. Use this to load data for testing.

    Returns
    -------
    parts : dict
        The chopped particle data.
    halos : dict
        The chopped halo data.
    """

    # HALO FILE
    important_keys = [
        "x", "y", "z", "vx", "vy", "vz",
        "upid", "halo_id",
        "mpeak", "host_mpeak",
        "vmax_frac",
        "host_x", "host_y", "host_z", "host_dist",
        "mmhid", "mmh_x", "mmh_y", "mmh_z"
    ]

    # load in the halo file and make optional host mpeak cut
    halos = OrderedDict()
    with h5py.File(halo_file, "r") as hdf:
        _host_mpeak_mask = np.log10(hdf["host_mpeak"][...]) >= host_mpeak_cut
        for key in hdf.keys():
            # only keep columns we want
            if key not in important_keys:
                continue

            # integer dtypes
            if key in ("halo_id", "upid", "mmhid"):
                dt = "i8"
            else:
                dt = "f4"
            halos[key] = hdf[key][...][_host_mpeak_mask].astype(dt)

    # if mmhid not known, find it
    if "mmhid" not in halos.keys():

        mmh_info = find_and_write_most_massive_hosts(halo_file, export=False)

        halos["mmhid"] = mmh_info[0]

        halos["mmh_x"] = mmh_info[1]
        halos["mmh_y"] = mmh_info[2]
        halos["mmh_z"] = mmh_info[3]

    assert len((set(halos["halo_id"]))) == len(halos["halo_id"])

    # assign each rank a chunk to then distribute and overload
    # this is easier than only rank 0 loading for when we need to find mmh
    avg, rem = divmod(len(halos["halo_id"]), N_RANKS)
    rank_count = [avg + 1 if p < rem else avg for p in range(N_RANKS)]
    displ = [sum(rank_count[:p]) for p in range(N_RANKS)]

    rank_count = rank_count[RANK]
    displ = displ[RANK]

    halos_rank = {}
    for key in halos.keys():
        halos_rank[key] = halos[key][displ:displ+rank_count]

    # munge
    _munge_halos(halos_rank)

    # fix "out of bounds" halos using periodicty
    for pos in ["halo_x", "halo_y", "halo_z", "mmh_x", "mmh_y", "mmh_z"]:
        halos_rank[pos][halos_rank[pos] < 0] += box_length
        halos_rank[pos][halos_rank[pos] > box_length] -= box_length

    # use MPIPartition to distribute and overload
    partition = mpipartition.Partition()

    halos_rank = mpipartition.distribute(partition, box_length, data=halos_rank,
                                         coord_keys=("mmh_x", "mmh_y", "mmh_z"))
    halos_rank["rank"] = np.zeros_like(halos_rank["halo_x"], dtype=np.int32) + RANK

    # test: check that each partition has entire structure
    mmhid_mod = np.copy(halos_rank["mmhid"])
    mmhid_mod[mmhid_mod == -1] = halos_rank["halo_id"][halos_rank["mmhid"] == -1]

    allhosts = np.unique(mmhid_mod)
    neededsubs = halos["halo_id"][np.isin(halos["mmhid"], allhosts)]
    assert len(np.setdiff1d(neededsubs, halos_rank["halo_id"])) == 0

    halos_rank = mpipartition.overload(partition, box_length, halos_rank,
                                       buff_wprp,
                                       ["halo_x", "halo_y", "halo_z"],
                                       structure_key="mmhid"
                                       )

    halos_rank["_inside_subvol"] = halos_rank["rank"] == RANK

    # again, check that each rank has entire structure
    mmhid_mod = np.copy(halos_rank["mmhid"])
    mmhid_mod[mmhid_mod == -1] = halos_rank["halo_id"][halos_rank["mmhid"] == -1]

    allhosts = np.unique(mmhid_mod)
    neededsubs = halos["halo_id"][np.isin(halos["mmhid"], allhosts)]
    assert len(np.setdiff1d(neededsubs, halos_rank["halo_id"])) == 0

    # wrap to volume
    center = box_length * (
        np.array(partition.extent) / 2.0 +
        np.array(partition.origin)
    )

    wrap_to_local_volume_inplace(halos_rank["halo_x"], center[0], box_length)
    wrap_to_local_volume_inplace(halos_rank["halo_y"], center[1], box_length)
    wrap_to_local_volume_inplace(halos_rank["halo_z"], center[2], box_length)

    # PARTICLE FILE

    # load in particle file
    parts = OrderedDict()
    with h5py.File(part_file, "r") as hdf:
        parts["x"] = hdf["data"]["x"][...].astype("f4")
        parts["y"] = hdf["data"]["y"][...].astype("f4")
        parts["z"] = hdf["data"]["z"][...].astype("f4")
        parts["part_id"] = np.arange(len(parts["x"])).astype(np.int64)

    # chop the particle catalog
    parts_rank = mpipartition.distribute(partition, box_length, parts,
                                         ["x", "y", "z"])
    parts_rank = mpipartition.overload(partition, box_length, parts_rank,
                                       buff_wprp, ["x", "y", "z"])

    return halos_rank, parts_rank

# END load_and_chop_bolshoi_planck


# TODO move to mpipartition
# def load_and_chop_data_bolshoi_planck(
#     *, part_file, halo_file, box_length, buff_ds, buff_wprp, ndiv, host_mpeak_cut=0
# ):
#     """Load and chop the data.

#     Parameters
#     ----------
#     part_file : str
#         The path to the HDF5 file with the particle data.
#     halo_file : str
#         The path to the HDF5 file with the halo data.
#     box_length : float
#         The length of the periodic volumne.
#     buff_ds : float
#         The buffer length to use for DeltaSigma.
#     buff_wprp : float
#         The buffer length to use for wp(rp).
#     ndiv : int
#         The number of divisions on each dimension.
#     host_mpeak_cut : float
#         The cut in host Mpeak. Use this to load data for testing.

#     Returns
#     -------
#     halos : dict
#         The chopped halo data.
#     parts : dict
#         The chopped particle data.
#     """

#     assert (
#         box_length / ndiv + 2 * buff_wprp
#     ) < box_length, "The buffer region is too big or ndiv is too small for the halos!"
#     assert (
#         box_length / ndiv + 2 * buff_ds
#     ) < box_length, (
#         "The buffer region is too big or ndiv is too small for the particles!"
#     )

#     start = time.time()
#     if RANK == 0:
#         halos = OrderedDict()
#         with h5py.File(halo_file, "r") as hdf:
#             _host_mpeak_mask = hdf["host_mpeak"][...] >= host_mpeak_cut
#             for key in hdf.keys():
#                 # don't keep all of the columns
#                 if key not in [
#                     "x",
#                     "y",
#                     "z",
#                     "vx",
#                     "vy",
#                     "vz",
#                     "upid",
#                     "halo_id",
#                     "mpeak",
#                     "host_mpeak",
#                     "vmax_frac",
#                     "host_x",
#                     "host_y",
#                     "host_z",
#                 ]:
#                     continue

#                 # integer dtypes
#                 if key in ("halo_id", "upid"):
#                     dt = "i8"
#                 else:
#                     dt = "f4"

#                 halos[key] = hdf[key][...][_host_mpeak_mask].astype(dt)

#         assert len((set(halos["halo_id"]))) == len(halos["halo_id"])

#         print("number of halos = %d" % (halos["x"].size,))
#         _munge_halos(halos)

#         parts = OrderedDict()
#         with h5py.File(part_file, "r") as hdf:
#             parts["x"] = hdf["data"]["x"][...].astype("f4")
#             parts["y"] = hdf["data"]["y"][...].astype("f4")
#             parts["z"] = hdf["data"]["z"][...].astype("f4")
#         parts["part_id"] = np.arange(len(parts["x"])).astype(np.int64)

#         print("number of particles = %d" % (parts["x"].size,))

#     else:
#         halos = OrderedDict()
#         parts = OrderedDict()

#     COMM.Barrier()
#     if RANK == 0:
#         # we chop on host halo position so that we can do merging of disrupted
#         # sats
#         halos["x"] = halos["host_x"]
#         halos["y"] = halos["host_y"]
#         halos["z"] = halos["host_z"]

#     halocats_for_rank, cell_ids_for_rank = get_buffered_subvolumes(
#         COMM, halos, ndiv, ndiv, ndiv, box_length, buff_wprp,
#     )
#     assert (
#         len(halocats_for_rank) == 1
#     ), "You must have as many ranks as chopped regions!"
#     halos = halocats_for_rank[0]

#     assert len((set(halos["halo_id"]))) == len(
#         halos["halo_id"]
#     ), "I found duplicate halos. Something bad happened!"

#     # now we put back the proper halo positions
#     # and since we chopped on host position, we mark _inside_subvol using
#     # the host value of _inside_subvol
#     halos["x"] = halos["halo_x"]
#     halos["y"] = halos["halo_y"]
#     halos["z"] = halos["halo_z"]
#     halos["_inside_subvol"] = halos["_inside_subvol"][_compute_host_index(halos)]

#     particles_for_rank, __ = get_buffered_subvolumes(
#         COMM, parts, ndiv, ndiv, ndiv, box_length, buff_ds,
#     )
#     assert (
#         len(particles_for_rank) == 1
#     ), "You must have as many ranks as chopped regions!"
#     parts = particles_for_rank[0]

#     assert len((set(parts["part_id"]))) == len(
#         parts["part_id"]
#     ), "I found duplicate particles. Something bad happened!"

#     end = time.time()
#     if RANK == 0:
#         msg = "Runtime to chop all the data = {0:.1f} seconds"
#         print(msg.format(end - start))

#     # now we precompute the index of the host halo into which
#     # merging satellite mass would be deposited
#     halos["indx_to_deposit"] = _calculate_indx_to_deposit(
#         halos["upid"], halos["halo_id"],
#     )

#     return halos, parts
