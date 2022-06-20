import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True)
def interact(f1_x, f1_y, f2_x, f2_y, tlen, a, h):
    """
    f1_x: shape is 1 * t
    """
    # print("interaction function called wit tlen: ", tlen)
    val = np.zeros(tlen)
    # a = 0.001 # good
    # a = 0.1
    # h = 1.0
    for t in range(tlen):
        val[t] = h * np.exp(-0.5 * ((f1_x[t]-f2_x[t])**2 + (f1_y[t]-f2_y[t])**2) / a) / np.sqrt(2 * np.pi * a)
        # print("val[t]: ", val[t])
    return val.max()
    # return 0.0
