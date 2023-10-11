"""Testing functions of the sigmoid-based stellar-to-halo-mass relation."""
import numpy as np
import pytest

from ..sigmoid_smhm_sigma import logsm_sigma_from_logmhalo
from ..sigmoid_smhm_sigma import DEFAULT_PARAM_VALUES as smhm_sigma_params_dict

NPTS = 4
LOGMHALO = np.linspace(8, 17, NPTS)
LOGMSTAR_SIGMA = np.array([0.25986877, 0.25249794, 0.2450166 , 0.23775408])


@pytest.mark.mpi_skip
def test_sigmoid_smhm_sigma_smoke():
    logsm_sigma = logsm_sigma_from_logmhalo(LOGMHALO, **smhm_sigma_params_dict)
    assert np.all(logsm_sigma > 0), logsm_sigma


@pytest.mark.mpi_skip
def test_sigmoid_smhm_regression():
    logsm_sigma = logsm_sigma_from_logmhalo(LOGMHALO, **smhm_sigma_params_dict)
    assert np.allclose(logsm_sigma, LOGMSTAR_SIGMA)
