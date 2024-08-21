import numpy as np


def make_FIM(logdensity_grad, theta, eps=1e-6):
    """
    make fisher info matrix given a logdensity fns grad and position
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

    # print smallest eig as starting place for step size
    eigs = np.linalg.eig(FIM)[0]
    print("min eig of FIM:", np.sqrt(np.min(eigs)), flush=True)

    return FIM
