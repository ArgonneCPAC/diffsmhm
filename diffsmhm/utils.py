import os
import sys
import time
import contextlib

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1


@contextlib.contextmanager
def time_step(step):
    try:
        t0 = time.time()
        yield
    finally:
        t0 = time.time() - t0
        if "DEBUG_TIMING" in os.environ:
            print(
                "TIME RANK %04d: %s took %f seconds" % (
                    RANK, step, t0,
                ),
                flush=True,
                file=sys.stderr,
            )
