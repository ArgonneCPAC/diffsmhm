"""Imports for galhalo_models sub-package."""
# flake8: noqa

__all__ = (
    "logsm_from_logmhalo",
    "logsm_sigma_from_logmhalo",
    "quenching_prob",
    "disruption_probability",
    "deposit_stellar_mass",
)


from collections import OrderedDict
from .sigmoid_smhm import *
from .sigmoid_smhm_sigma import *
from .sigmoid_quenching import *
from .sigmoid_disruption import *
from .merging import *

from .sigmoid_smhm_sigma import DEFAULT_PARAM_VALUES as smhm_sigma_params
from .sigmoid_smhm import DEFAULT_PARAM_VALUES as smhm_params
from .sigmoid_disruption import DEFAULT_PARAM_VALUES as disruption_params
from .sigmoid_quenching import DEFAULT_PARAM_VALUES as quenching_params

default_model_params = OrderedDict()
default_model_params.update(smhm_params)
default_model_params.update(smhm_sigma_params)
default_model_params.update(disruption_params)
default_model_params.update(quenching_params)

from .sigmoid_smhm import PARAM_BOUNDS as smhm_param_bounds
from .sigmoid_smhm_sigma import PARAM_BOUNDS as smhm_sigma_param_bounds
from .sigmoid_disruption import PARAM_BOUNDS as disruption_param_bounds
from .sigmoid_quenching import PARAM_BOUNDS as quenching_param_bounds

param_bounds = OrderedDict()
param_bounds.update(smhm_param_bounds)
param_bounds.update(smhm_sigma_param_bounds)
param_bounds.update(disruption_param_bounds)
param_bounds.update(quenching_param_bounds)
