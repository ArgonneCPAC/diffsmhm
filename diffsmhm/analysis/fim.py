import numpy as np


def make_FIM(logdensity_grad, theta, eps=1e-6):
    """
    Make fisher info matrix given a logdensity function's gradient and position

    Parameters
    ----------
    logdensity_grad : function handle
        Function for the gradient of logdensity. Takes one argument, theta.
    theta : array_like, shape (n_params,)
        Position to compute second derivative around.
    eps : float
        Finite difference parameter.

    Returns
    -------
    FIM : array_like, shape (n_params, n_params)
        Fisher information matrix around  `theta`.
    min_eig : float
        Square root of the minimum eigenvalue of FIM, can be used as an
        initial estimate of step size in HMC.
    """

    n_params = len(theta)

    FIM = np.zeros((n_params, n_params), dtype=np.float64)

    for i in range(n_params):
        # hessian of logdensity fn by finite difference
        theta_p = np.copy(theta)
        theta_p[i] += eps
        val_p = logdensity_grad(theta_p)

        theta_m = np.copy(theta)
        theta_m[i] -= eps
        val_m = logdensity_grad(theta_m)

        grad = (val_p - val_m) / 2.0 / eps

        FIM[i, :] = -1.0 * grad

    # check for SPD
    try:
        _ = np.linalg.cholesky(FIM)
    except RuntimeError:
        print("FIM is not SPD")
        exit()

    # smallest eig as a rough starting place for HMC step size
    eigs = np.linalg.eig(FIM)[0]

    return FIM, np.sqrt(np.min(eigs))
