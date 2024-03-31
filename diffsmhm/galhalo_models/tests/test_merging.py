"""Unit-tests for merging."""
import numpy as np
import pytest
from collections import OrderedDict
from ..merging import deposit_stellar_mass, _calculate_indx_to_deposit


@pytest.mark.mpi_skip
def test1():
    """Create a dummy galaxy catalog and test merging."""
    rng = np.random.RandomState(43)
    ngals = int(1e5)
    nsats = int(0.5 * ngals)
    nhosts = ngals - nsats

    halo_ids_cens = np.arange(0, nhosts).astype("i8")
    halo_ids_sats = np.arange(nhosts, ngals).astype("i8")
    halo_ids = np.concatenate((halo_ids_cens, halo_ids_sats))

    host_ids_cens = np.arange(0, nhosts).astype("i8")
    halo_ids_sats = rng.randint(0, nhosts, nsats).astype("i8")
    host_ids = np.concatenate((host_ids_cens, halo_ids_sats))

    upid_cens = np.zeros(nhosts).astype("i8") - 1
    upid_sats = np.copy(halo_ids_sats)
    upids = np.concatenate((upid_cens, upid_sats))

    mstar_cens = rng.uniform(11, 12, nhosts)
    mstar_sats = rng.uniform(9, 10, nsats)
    mstar_gals = np.concatenate((mstar_cens, mstar_sats)).astype("f4")

    frac_to_deposit_cens = np.zeros(nhosts).astype("f4")
    frac_to_deposit_sats = rng.uniform(0, 1, nsats)
    frac_to_deposit = np.concatenate(
        (frac_to_deposit_cens, frac_to_deposit_sats)
    ).astype("f4")

    indx = np.lexsort((upids, host_ids))

    d = OrderedDict()
    d["indx"] = np.arange(ngals).astype("i8")
    d["upid"] = upids[indx]
    d["halo_id"] = halo_ids[indx]
    d["host_id"] = host_ids[indx]
    d["logsm"] = mstar_gals[indx]
    d["frac_to_deposit"] = frac_to_deposit[indx]

    host_id, indx, counts = np.unique(
        d["host_id"], return_index=True, return_counts=True
    )
    d["richness"] = np.repeat(counts, counts)

    #  Now scramble the data and deposit the mass from mergers
    d2 = OrderedDict()
    indx_ran = rng.choice(np.arange(ngals), ngals, replace=False).astype("i4")
    d2 = OrderedDict([(key, arr[indx_ran]) for key, arr in d.items()])
    #
    d = d2

    indx_to_deposit = _calculate_indx_to_deposit(d["upid"], d["halo_id"])
    d["total_mstar"] = deposit_stellar_mass(
        d["logsm"], indx_to_deposit, d["frac_to_deposit"]
    )

    #  Mass should be conserved
    assert np.allclose(np.sum(d["total_mstar"]), np.sum(10 ** d["logsm"]))

    #   Centrals should never lose mass due to merging
    cenmask = d["upid"] == -1
    assert np.all(np.log10(d["total_mstar"][cenmask]) >= d["logsm"][cenmask])

    #  Centrals without satellites should not gain mass from merging
    nosat_mask = d["richness"] < 2
    assert np.allclose(
        np.log10(d["total_mstar"][cenmask & nosat_mask]),
        d["logsm"][cenmask & nosat_mask]
    )

    #  We set up frac_merge to be positive, so satellites should strictly lose
    #  mass from merging
    assert np.all(np.log10(d["total_mstar"][~cenmask]) < d["logsm"][~cenmask])


@pytest.mark.mpi_skip
def test_calculate_indx_to_deposit_missing():
    # check that missing throws error
    upids = np.arange(10, dtype="i")
    halo_ids = np.arange(9, dtype="i")

    try:
        ok = False
        _calculate_indx_to_deposit(upids, halo_ids)
    except AssertionError:
        ok = True
    finally:
        assert ok, "calculate_indx_to_deposit failed to identify missing upid"

    # check that all present doesn't throw error
    upids = np.arange(9, dtype="i")

    try:
        ok = False
        _calculate_indx_to_deposit(upids, halo_ids)
    except AssertionError:
        ok = False
    else:
        ok = True
    finally:
        assert ok, "calculate_indx_to_deposit fails when all upids are present"
