import numpy as np

from diffsmhm.galhalo_models.sigmoid_disruption import (
    DEFAULT_PARAM_VALUES as smhm_params,
    PARAM_VALUES as smhm_bounds
)
from diffsmhm.galhalo_models.sigmoid_smhm_sigma import (
    DEFAULT_PARAM_VALUES as smhm_sigma_params,
    PARAM_VALUES as smhm_sigma_bounds
)
from diffsmhm.galhalo_models.sigmoid_disruption import (
    DEFAULT_PARAM_VALUES as disruption_params,
    PARAM_BOUNDS as disruption_bounds
)


def get_default_params():
    theta = np.array(list(smhm_params.values()) +
                     list(smhm_sigma_params.values()) +
                     list(disruption_params.values()), dtype=np.float64)
    return theta


def get_param_bounds():
    lower_bounds = np.array([
        smhm_bounds["smhm_logm_crit"][0],
        smhm_bounds["smhm_ratio_logm_crit"][0],
        smhm_bounds["smhm_k_logm"][0],
        smhm_bounds["smhm_lowm_index"][0],
        smhm_bounds["smhm_highm_index"][0],
        smhm_sigma_bounds["smhm_sigma_low"][0],
        smhm_sigma_bounds["smhm_sigma_high"][0],
        smhm_sigma_bounds["smhm_sigma_logm_pivot"][0],
        smhm_sigma_bounds["smhm_sigma_logm_width"][0],
        disruption_bounds["satmerg_logmhost_crit"][0],
        disruption_bounds["satmerg_logmhost_k"][0],
        disruption_bounds["satmerg_logvr_crit_dwarfs"][0],
        disruption_bounds["satmerg_logvr_crit_clusters"][0],
        disruption_bounds["satmerg_logvr_k"][0],
    ], dtype=np.float64)
    upper_bounds = np.array([
        smhm_bounds["smhm_logm_crit"][1],
        smhm_bounds["smhm_ratio_logm_crit"][1],
        smhm_bounds["smhm_k_logm"][1],
        smhm_bounds["smhm_lowm_index"][1],
        smhm_bounds["smhm_highm_index"][1],
        smhm_sigma_bounds["smhm_sigma_low"][1],
        smhm_sigma_bounds["smhm_sigma_high"][1],
        smhm_sigma_bounds["smhm_sigma_logm_pivot"][1],
        smhm_sigma_bounds["smhm_sigma_logm_width"][1],
        disruption_bounds["satmerg_logmhost_crit"][1],
        disruption_bounds["satmerg_logmhost_k"][1],
        disruption_bounds["satmerg_logvr_crit_dwarfs"][1],
        disruption_bounds["satmerg_logvr_crit_clusters"][1],
        disruption_bounds["satmerg_logvr_k"][1],
    ], dtype=np.float64)

    return lower_bounds, upper_bounds
