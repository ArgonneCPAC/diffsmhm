from collections import namedtuple

WprpMPIData = namedtuple(
    "WprpMPIData",
    ["dd", "dd_jac", "w_tot", "w2_tot", "ww_jac_tot", "w_jac_tot"]
)
