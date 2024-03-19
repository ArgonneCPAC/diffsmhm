import numpy as np
import matplotlib.pyplot as plt

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1

from adam import adam

# define function
def compute(x, a, b):
    if RANK == 0:
        return a*x + b
    else:
        return None

# define error
def error_adam_wrapped(static_params, opt_params):
    # static_params = [x, goal]
    # opt_params = [a, b, c]
    goal = static_params[1]

    guess = compute(static_params[0], opt_params[0], opt_params[1])

    if RANK == 0:
        err = np.sum((guess-goal)**2)
        err_grad = np.array([np.sum(2*x*(guess-goal)), np.sum(2*1*(guess-goal))], dtype=np.float64)

        return err, err_grad
    else:
        return None, None

# "goal" parameters
a_goal = 3.0
b_goal = 4.0

x = np.linspace(1,10, 1000, dtype=np.float64)

out_goal = compute(x, a_goal, b_goal)
out_init = compute(x, 1.0, 1.0)

# do optimization
opt_params = np.array([1.0, 1.0], dtype=np.float64)
static_params = [x, out_goal]

theta, error_history = adam(
                        static_params,
                        opt_params,
                        maxiter=10000000,
                        minerr=0.1,
                        tmax=10*60,
                        err_func=error_adam_wrapped
)

if RANK == 0:
    print("eh:", error_history[0], error_history[-1])
    print("theta:", theta)

# plot 
if RANK == 0:
    out_final = compute(x, theta[0], theta[1])

    fig = plt.figure(figsize=(8,6), facecolor="w")

    plt.plot(x, out_init, c="tab:blue", linewidth=3)
    plt.plot(x, out_final, c="tab:orange", linewidth=3)
    plt.plot(x, out_goal, c="k", linewidth=1)

    plt.legend(["initial", "final", "goal"])

    plt.savefig("adam_proof.png")
