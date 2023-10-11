import numpy as np
from numba import njit

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
