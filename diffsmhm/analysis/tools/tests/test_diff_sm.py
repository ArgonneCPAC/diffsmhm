from jax import config
config.update("jax_enable_x64", True)

import pytest

import numpy as np
from numpy.testing import assert_allclose

from diffsmhm.galhalo_models.sigmoid_smhm import (
    DEFAULT_PARAM_VALUES as smhm_params
)
from diffsmhm.galhalo_models.sigmoid_smhm_sigma import (
    DEFAULT_PARAM_VALUES as smhm_sigma_params
)
from diffsmhm.galhalo_models.sigmoid_disruption import (
    DEFAULT_PARAM_VALUES as disruption_params
)
from diffsmhm.galhalo_models.sigmoid_quenching import (
    DEFAULT_PARAM_VALUES as quenching_params
)

from diffsmhm.analysis.tools.diff_sm import (
    compute_sm_and_jac,
    compute_weight_and_jac,
    compute_weight_and_jac_quench
)


def _get_data():
    np.random.seed(42)

    ngals = 1000
    hm = np.random.uniform(4.0, 16.0, size=ngals)
    vmax_frac = np.random.uniform(0.5, 1.5, size=ngals)
    upid = np.zeros(ngals, dtype="i")
    upid[0] = -1
    idx_to_deposit = np.zeros(ngals, dtype="i")
    #idx_to_deposit = np.random.randint(0, ngals, size=ngals)
    #hostm = hm[idx_to_deposit]
    hostm = np.ones(ngals, dtype=np.float64)*hm[0]
    tinfall = np.random.uniform(0.0, 2.0, size=ngals)

    theta = np.array(list(smhm_params.values()) +
                     list(smhm_sigma_params.values()) +
                     list(disruption_params.values()) +
                     list(quenching_params.values()), dtype=np.float64)

    return hm, hostm, vmax_frac, upid, idx_to_deposit, tinfall, theta

@pytest.mark.mpi_skip
def test_compute_sm_and_jac_smoke():
    hm, hostm, vmax_frac, upid, idx_to_deposit, _, theta = _get_data()

    sm, sm_jac = compute_sm_and_jac(
                    logmpeak=hm,
                    loghost_mpeak=hostm,
                    log_vmax_by_vmpeak=vmax_frac,
                    upid=upid,
                    idx_to_deposit=idx_to_deposit,
                    theta=theta
    )

    assert sm.shape == (len(hm),)
    assert sm_jac.shape == (len(theta), len(hm))
    assert np.all(np.isfinite(sm))
    assert np.all(np.isfinite(sm_jac))
    assert np.any(sm != 0)
    assert np.any(sm_jac != 0)


@pytest.mark.mpi_skip
def test_compute_sm_and_jac_derivs():
    hm, hostm, vmax_frac, upid, idx_to_deposit, _, theta = _get_data()
    npars = len(theta)

    # test gradient
    sm, sm_jac = compute_sm_and_jac(
                    logmpeak=hm,
                    loghost_mpeak=hostm,
                    log_vmax_by_vmpeak=vmax_frac,
                    upid=upid,
                    idx_to_deposit=idx_to_deposit,
                    theta=theta
    )
    
    eps = 1e-6
    for pind in range(npars):
        # skip the params that don't matter for this calculation
        if pind in [5,6,7,8, 14,15,16,17,18,19,20,21,22]:
            continue

        theta_p = np.copy(theta)
        theta_p[pind] += eps
        sm_p, _ = compute_sm_and_jac(
                    logmpeak=hm,
                    loghost_mpeak=hostm,
                    log_vmax_by_vmpeak=vmax_frac,
                    upid=upid,
                    idx_to_deposit=idx_to_deposit,
                    theta=theta_p
        )

        theta_m = np.copy(theta)
        theta_m[pind] -= eps
        sm_m, _ = compute_sm_and_jac(
                    logmpeak=hm,
                    loghost_mpeak=hostm,
                    log_vmax_by_vmpeak=vmax_frac,
                    upid=upid,
                    idx_to_deposit=idx_to_deposit,
                    theta=theta_m
        )

        print(pind)
        grad = (sm_p - sm_m)/2.0/eps
        assert_allclose(sm_jac[pind, :], grad)
        assert np.any(grad != 0)
        assert np.any(sm_jac[pind, :] != 0)


def test_compute_weight_and_jac_smoke():
    hm, hostm, vmax_frac, upid, idx_to_deposit, _, theta = _get_data()

    w, dw = compute_weight_and_jac(
                    logmpeak=hm,
                    loghost_mpeak=hostm,
                    log_vmax_by_vmpeak=vmax_frac,
                    upid=upid,
                    idx_to_deposit=idx_to_deposit,
                    mass_bin_low=10.0,
                    mass_bin_high=12.0,
                    theta=theta
    )

    assert w.shape == (len(hm),)
    assert dw.shape == (len(theta), len(hm))
    assert np.all(np.isfinite(w))
    assert np.all(np.isfinite(dw))


def test_compute_weight_and_jac_derivs():
    hm, hostm, vmax_frac, upid, idx_to_deposit, _, theta = _get_data()

    npars = len(theta)
    mb_low = 10.0
    mb_high = 12.0

    # test gradient
    w, dw = compute_weight_and_jac(
                    logmpeak=hm,
                    loghost_mpeak=hostm,
                    log_vmax_by_vmpeak=vmax_frac,
                    upid=upid,
                    idx_to_deposit=idx_to_deposit,
                    mass_bin_low=mb_low,
                    mass_bin_high=mb_high,
                    theta=theta
    )
    
    eps = 1e-6
    for pind in range(npars):
        # skip the params that don't matter for this calculation
        if pind in [14,15,16,17,18,19,20,21,22]:
            continue

        theta_p = np.copy(theta)
        theta_p[pind] += eps
        w_p, _ = compute_weight_and_jac(
                    logmpeak=hm,
                    loghost_mpeak=hostm,
                    log_vmax_by_vmpeak=vmax_frac,
                    upid=upid,
                    idx_to_deposit=idx_to_deposit,
                    mass_bin_low=mb_low,
                    mass_bin_high=mb_high,
                    theta=theta_p
        )

        theta_m = np.copy(theta)
        theta_m[pind] -= eps
        w_m, _ = compute_weight_and_jac(
                    logmpeak=hm,
                    loghost_mpeak=hostm,
                    log_vmax_by_vmpeak=vmax_frac,
                    upid=upid,
                    idx_to_deposit=idx_to_deposit,
                    mass_bin_low=mb_low,
                    mass_bin_high=mb_high,
                    theta=theta_m
        )

        print(pind)
        grad = (w_p - w_m)/2.0/eps
        assert_allclose(dw[pind, :], grad)
        assert np.any(grad != 0)
        assert np.any(sm_jac[pind, :] != 0)


def test_compute_weight_and_jac_quench_smoke():
    hm, hostm, vmax_frac, upid, idx_to_deposit, tinfall, theta = _get_data()

    wq, dwq, wnq, dwnq = compute_weight_and_jac_quench(
                            logmpeak=hm,
                            loghost_mpeak=hostm,
                            log_vmax_by_vmpeak=vmax_frac,
                            upid=upid,
                            time_since_infall=tinfall,
                            idx_to_deposit=idx_to_deposit,
                            mass_bin_low=10.0,
                            mass_bin_high=12.0,
                            theta=theta
    )

    assert wq.shape == (len(hm),)
    assert wnq.shape == (len(hm),)
    assert dwq.shape == (len(theta), len(hm))
    assert dwnq.shape == (len(theta), len(hm))
    assert np.all(np.isfinite(wq))
    assert np.all(np.isfinite(dwq))
    assert np.all(np.isfinite(wnq))
    assert np.all(np.isfinite(dwnq))


def test_compute_weight_and_jac_quench_derivs():
    hm, hostm, vmax_frac, upid, idx_to_deposit, tinfall, theta = _get_data()

    npars = len(theta)
    mb_low = 10.0
    mb_high = 12.0

    # test gradient
    wq, dwq, wnq, dwnq = compute_weight_and_jac_quench(
                            logmpeak=hm,
                            loghost_mpeak=hostm,
                            log_vmax_by_vmpeak=vmax_frac,
                            upid=upid,
                            time_since_infall=tinfall,
                            idx_to_deposit=idx_to_deposit,
                            mass_bin_low=mb_low,
                            mass_bin_high=mb_high,
                            theta=theta
    )
    
    eps = 1e-6
    for pind in range(npars):

        theta_p = np.copy(theta)
        theta_p[pind] += eps
        wq_p, _, wnq_p, _ = compute_weight_and_jac_quench(
                    logmpeak=hm,
                    loghost_mpeak=hostm,
                    log_vmax_by_vmpeak=vmax_frac,
                    upid=upid,
                    time_since_infall=tinfall,
                    idx_to_deposit=idx_to_deposit,
                    mass_bin_low=mb_low,
                    mass_bin_high=mb_high,
                    theta=theta_p
        )

        theta_m = np.copy(theta)
        theta_m[pind] -= eps
        wq_m, _, wnq_m, _ = compute_weight_and_jac_quench(
                    logmpeak=hm,
                    loghost_mpeak=hostm,
                    log_vmax_by_vmpeak=vmax_frac,
                    upid=upid,
                    time_since_infall=tinfall,
                    idx_to_deposit=idx_to_deposit,
                    mass_bin_low=mb_low,
                    mass_bin_high=mb_high,
                    theta=theta_m
        )

        print(pind)
        grad_q = (wq_p - wq_m)/2.0/eps
        grad_nq = (wnq_p - wnq_m)/2.0/eps
        assert_allclose(dwq[pind, :], grad_q)
        assert_allclose(dwnq[pind, :], grad_nq)
        assert np.any(grad_q != 0)
        assert np.any(grad_nq != 0)
        assert np.any(dwq[pind, :] != 0)
        assert np.any(dwnq[pind, :] != 0)
    