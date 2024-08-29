import pytest

import numpy as np
from numpy.testing import assert_allclose

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1

from diffsmhm.analysis.adam import adam


@pytest.mark.mpi_skip
def test_adam():
    # define model
    x = np.linspace(1, 100, 1000, dtype=np.float64)

    def model(theta, x=x):
        a = theta[0]
        b = theta[1]

        y = a*x*x + b*x
        y_grad = np.array([x*x, x], dtype=np.float64)

        return y, y_grad

    theta_goal = np.array([1.0, 3.0], dtype=np.float64)
    theta_init = np.array([2.0, 2.0], dtype=np.float64)

    y_goal = model(theta_goal)[0]

    def mse_test(theta):
        y, y_grad = model(theta)

        error = np.sum((y - y_goal)**2) / len(y)
        error_grad = np.sum(2 * y_grad * (y - y_goal), axis=1) / len(y)

        return error, error_grad

    # do optimization
    theta_opt, err_history = adam(
                                opt_params=theta_init,
                                err_func=mse_test,
                                a=0.1,
                                maxiter=10_000,
    )

    # lower error threshold here just to lower number of iters needed
    assert_allclose(theta_goal, theta_opt, rtol=0.001)
