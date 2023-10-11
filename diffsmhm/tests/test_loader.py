import numpy as np
from numpy.testing import assert_allclose
import pytest

from ..loader import _compute_host_index, _munge_halos, wrap_to_local_volume_inplace


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
