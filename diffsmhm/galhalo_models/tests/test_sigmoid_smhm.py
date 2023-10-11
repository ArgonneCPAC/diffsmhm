"""Testing functions of the sigmoid-based stellar-to-halo-mass relation."""
import numpy as np
import pytest

from ..sigmoid_smhm import logsm_from_logmhalo
from ..sigmoid_smhm import DEFAULT_PARAM_VALUES as smhm_params_dict

NPTS = 4
LOGMHALO = np.linspace(8, 17, NPTS)
LOGMSTAR = np.log10(
    np.array([2.28017414e01, 1.20320663e09, 1.26854219e11, 3.36092673e12])
)


@pytest.mark.mpi_skip
def test_sigmoid_smhm_smoke():
    logsm = logsm_from_logmhalo(LOGMHALO, **smhm_params_dict)
    assert np.all(np.diff(logsm) > 0), logsm

    smhm_logm_crit = smhm_params_dict["smhm_logm_crit"]
    smhm_ratio_logm_crit = smhm_params_dict["smhm_ratio_logm_crit"]
    correct_logsm_at_logm_crit = smhm_logm_crit + smhm_ratio_logm_crit

    implemented_logsm_at_logmcrit = logsm_from_logmhalo(
        smhm_logm_crit, **smhm_params_dict
    )

    assert np.allclose(implemented_logsm_at_logmcrit, correct_logsm_at_logm_crit)


@pytest.mark.mpi_skip
def test_sigmoid_smhm_regression():
    logsm = logsm_from_logmhalo(LOGMHALO, **smhm_params_dict)
    assert np.allclose(logsm, LOGMSTAR)
