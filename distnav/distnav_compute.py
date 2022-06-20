import numpy as np
from numba import jit, prange
from .distnav_interact import interact


@jit(nopython=True, cache=False)
def init_config(num_agents):
    sub_table = np.zeros((num_agents, num_agents - 1), dtype=np.int32)
    for i in range(num_agents):
        for j in range(num_agents):
            if i < j:
                sub_table[i][j - 1] = j
            elif i > j:
                sub_table[i][j] = j
            else:
                continue
    return sub_table


@jit(nopython=True, parallel=True, cache=False)
def generate_expect_table(samples_x, samples_y, num_pts, num_agents,
                          pred_len, a, h, p2w_x, p2w_y):
    table = np.zeros((num_agents * num_pts, num_agents * num_pts))
    for i in prange(num_agents * num_pts):
        for j in range(num_agents * num_pts):
            f1_x = samples_x[i] * p2w_x
            f1_y = samples_y[i] * p2w_y
            f2_x = samples_x[j] * p2w_x
            f2_y = samples_y[j] * p2w_y
            table[i][j] = interact(f1_x, f1_y, f2_x, f2_y, pred_len, a, h)

    return table


@jit(nopython=True, cache=False, parallel=True)
def one_iteration(weights, table, sub_table, host_id, num_pts, num_agents):
    new_weights = weights.copy()
    for i in prange(num_pts):
        ev = 0.
        for j in prange(num_pts):
            for k in prange(num_agents - 1):
                client_id = sub_table[host_id][k]
                ev += table[host_id * num_pts + i][client_id * num_pts + j] * new_weights[client_id][j]
        ev /= num_pts
        new_weights[host_id][i] *= np.exp(-ev)
    new_weights[host_id] /= np.sum(new_weights[host_id]) / num_pts
    return new_weights.copy()


@jit(nopython=True, cache=False, parallel=True)
def one_iteration2(weights, table, sub_table, host_id, num_pts, num_agents, coll_weight):
    new_weights = weights.copy()
    for i in prange(num_pts):
        ev = 0.
        for j in prange(num_pts):
            for k in prange(num_agents - 1):
                client_id = sub_table[host_id][k]
                ev += table[host_id * num_pts + i][client_id * num_pts + j] * new_weights[client_id][j]
        ev /= num_pts
        new_weights[host_id][i] = np.exp(-coll_weight * ev)
    new_weights[host_id] /= np.sum(new_weights[host_id]) / num_pts
    return new_weights.copy()


@jit(nopython=True, cache=False, parallel=True)
def objective(table, weights, num_agents, num_pts):
    val = 0.
    for i in prange(num_agents):
        for j in prange(i + 1, num_agents):
            for k in prange(num_pts):
                val += table[i * num_pts + k][j * num_pts + k] * weights[i][k] * weights[j][k]
    val /= num_pts
    return val


@jit(nopython=True, cache=False)
def vigp_compute(essential_samples_x, essential_samples_y, essential_pdf_x, essential_pdf_y,
                 essential_num, num_samples, traj_len, p2w_x, p2w_y, a, h, obj_thred, max_iter, coll_weight=None):
    if coll_weight is None:
        print("optimization scheme: ", 1)
    else:
        print("optimization scheme: ", 2)

    weights = np.ones((essential_num, num_samples), dtype=np.float32)

    sub_table = init_config(essential_num)
    table = generate_expect_table(essential_samples_x, essential_samples_y, num_samples,
                                  essential_num, traj_len, a, h, p2w_x, p2w_y)
    print("score table generated.")

    obj = 10
    it = 0
    while True:
        if it == 0:
            print("start optimization ...")
        obj = objective(table, weights, essential_num, num_samples)
        # print("iteration: ", it, obj)
        if obj < obj_thred or it > max_iter:
            print("terminate iteration at: ", it, obj)
            break
        # kl_div = 0.
        for pid in range(essential_num):
            if coll_weight is None:
                new_weights = one_iteration(weights, table, sub_table, pid, num_samples, essential_num)
            else:
                new_weights = one_iteration2(weights, table, sub_table, pid, num_samples, essential_num, coll_weight)
            # kl_div += np.log(weights[pid] / new_weights[pid]).sum() / num_samples
            weights = new_weights.copy()
        # kl_div /= essential_num
        # print("it: ", it, ", kl_div: ", kl_div)
        it += 1

    return weights.copy()
