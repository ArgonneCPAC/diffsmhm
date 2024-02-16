import numpy as np
import time

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1

def adam(
    static_params,
    opt_params,
    err_func,
    maxiter,
    minerr,
    tmax=-1,
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
    minerr : float
        Error target at which to stop optimization
    tmax : float, optional
        Maximum time for which to run the optimizer in seconds
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
        err, err_grad = err_func(static_params, theta)

        err = COMM.bcast(err, root=0)
        err_grad = COMM.bcast(err_grad, root=0)

        err_history.append(err)

        # check loop condition        
        if err < minerr or t > maxiter:
            break
        if tmax > 0 and time.time() - tstart > tmax:
            break

        # update params

        # biased first moment
        m = b1*m + (1-b1)*err_grad
        # biased second moment
        v = b2*v + (1-b2)*err_grad**2
        # bias correct first moment
        mhat = m/(1-b1**t)
        # bias correct second moment
        vhat = v/(1-b2**t)
        # update_parameters
        theta -= a*mhat/(np.sqrt(vhat)+eps)

    # return updated parameters and error history
    return theta, err_history
