import cupy as cp
import numpy as np


def get_array_backend():
    try:
        _ = cp.array([1])
        return cp
    except RuntimeError:
        return np
