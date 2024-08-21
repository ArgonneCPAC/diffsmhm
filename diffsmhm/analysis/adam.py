import numpy as np
import time


def adam(
    *,
    static_params,
    opt_params,
    err_func,
    maxiter,
    tmax=-1,
    err_threshold=1e-6,
    a=0.001,
    b1=0.9,
    b2=0.999,
    eps=10**-8
):
    """Adam optimizer for a given error function.

    Parameters
    ---------
    static_params : array-like
        Parameters required for an error measurement but not to be optimized.
    opt_params : array-like, shape(n_params,)
        Parameters to optimize
    err_func : function
        Function that takes in (static_params, opt_params) and returns
        (error, error_jacobian).
    maxiter : int
        Maximum number of optimization loops to perform
    tmax : float, optional
        Maximum time for which to run the optimizer in seconds
    err_threshold : float, optional
        Error threshold at which algorithm will stop. Default is 1e-6.
    a : float, optional
        Adam parameter controlling stepsize scaling. Default is 0.001, taken
        from Kingma & Ba (2015).
    b1, b2 : float, optional
        Adam parameters controlling decay rates of step size. Defaults are
        b1=0.9, b2=0.999 which are taken from Kingma & Ba (2015).

    Returns
    -------
    theta : array-like, shape(n_params)
        Optimized values for input opt_params.
    error_history : array-like, shape(n_iter,)
        Error per iteration.
    """

    n_params = len(opt_params)

    # initialize vectors
    m = np.zeros(n_params, dtype=np.float64)
    v = np.zeros(n_params, dtype=np.float64)
    t = 0

    err_history = []

    theta = np.copy(opt_params)

    # get start time
    tstart = time.time()

    # optimize
    while True:
        t += 1

        # get error and gradient
        err, err_grad = err_func(theta)

        err_history.append(err)

        # rank 0 check loop condition
        cont = True
        if t > maxiter:
            cont = False
        telapsed = time.time() - tstart
        if tmax > 0 and telapsed > tmax:
            cont = False
        if err < err_threshold:
            cont = False
        if not cont:
            break

        # update params

        # biased first moment
        m = b1*m + (1-b1)*err_grad
        # biased second moment
        v = b2*v + (1-b2)*(err_grad**2)
        # bias correct first moment
        mhat = m/(1-b1**t)
        # bias correct second moment
        vhat = v/(1-b2**t)
        # update_parameters
        theta -= a*mhat/(np.sqrt(vhat)+eps)

    # return updated parameters and error history
    return theta, err_history
