import numpy as np
from numba import jit, prange
from .distnav_interact import interact


@jit(nopython=True, cache=True, parallel=True)
def importance_scores(agents_mu_x, agents_mu_y, agents_sample_x, agents_sample_y,
              traj_len, num_peds, num_samples, p2w_x, p2w_y, a, h):
    influence_scores = np.ones((num_peds+1, num_peds+1), dtype=np.float32)
    i = num_peds
    for j in prange(num_peds+1):
        exp_score = 0.0
        for k in prange(num_samples):
            val = interact(agents_mu_x[i]*p2w_x, agents_mu_y[i]*p2w_y,
                           agents_sample_x[j*num_samples+k]*p2w_x, agents_sample_y[j*num_samples+k]*p2w_y,
                           traj_len, a, h)
            exp_score += val
        exp_score /= num_samples
        influence_scores[i][j] = exp_score

    return influence_scores
