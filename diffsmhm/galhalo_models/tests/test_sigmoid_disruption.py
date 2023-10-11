"""Testing functions of the sigmoid-based stellar-to-halo-mass relation."""
import numpy as np
import pytest

from ..sigmoid_disruption import satellite_disruption_probability
from ..sigmoid_disruption import disruption_probability
from ..sigmoid_disruption import DEFAULT_PARAM_VALUES as param_dict

# these values were extracted from the first implementation of the
# model by aphearin
# if the model is changed, then these values should be redone
# otherwise, leave them alone
# they are used for regression testing the code
NPTS = 4
LOGMHOST = np.linspace(8, 17, NPTS)
LOG_VMAX_BY_VMPEAK = np.linspace(-2, 0, NPTS)
SAT_DISRUPT_PROBS = np.array([0.95938978, 0.741627, 0.43357322, 0.09853502])


@pytest.mark.mpi_skip
def test_sigmoid_disruption_cen_regression():
    prob_disrupt = disruption_probability(
        -1, LOG_VMAX_BY_VMPEAK, LOGMHOST, **param_dict
    )
    assert np.allclose(prob_disrupt, 0)


@pytest.mark.mpi_skip
def test_sigmoid_disruption_sat_regression():
    prob_disrupt = satellite_disruption_probability(
        LOG_VMAX_BY_VMPEAK, LOGMHOST, **param_dict
    )
    assert np.all(prob_disrupt >= 0)
    assert np.all(prob_disrupt <= 1)

    assert np.any(prob_disrupt > 0)
    assert np.any(prob_disrupt < 1)

    assert np.allclose(prob_disrupt, SAT_DISRUPT_PROBS)


@pytest.mark.mpi_skip
def test_sigmoid_disruption_satcen_consistent():
    prob_disrupt = disruption_probability(1, LOG_VMAX_BY_VMPEAK, LOGMHOST, **param_dict)
    prob_disrupt2 = satellite_disruption_probability(
        LOG_VMAX_BY_VMPEAK, LOGMHOST, **param_dict
    )
    assert np.allclose(prob_disrupt, prob_disrupt2)

    prob_disrupt3 = disruption_probability(
        -1, LOG_VMAX_BY_VMPEAK, LOGMHOST, **param_dict
    )
    assert np.allclose(prob_disrupt3, 0)
