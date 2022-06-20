import numpy as np


def rename(num_peds, conditioned,
           mu_linear_conditioned_x, mu_linear_conditioned_y,
           mu_linear_un_x, mu_linear_un_y,
           cov_linear_conditioned_x, cov_linear_conditioned_y,
           cov_un_x, cov_un_y):
    ped_mu_x = [0. for _ in range(num_peds)]
    ped_mu_y = [0. for _ in range(num_peds)]

    ped_cov_x = [0. for _ in range(num_peds)]
    ped_cov_y = [0. for _ in range(num_peds)]

    one_over_ped_cov_x = [0. for _ in range(num_peds)]
    one_over_ped_cov_y = [0. for _ in range(num_peds)]

    L_ped_cov_x = [0. for _ in range(num_peds)]
    L_ped_cov_y = [0. for _ in range(num_peds)]

    inv_cov_ped_x = [0. for _ in range(num_peds)]
    inv_cov_ped_y = [0. for _ in range(num_peds)]

    inv_var_ped_x = [0. for _ in range(num_peds)]
    inv_var_ped_y = [0. for _ in range(num_peds)]

    cov_sum_x = [0. for _ in range(num_peds)]
    cov_sum_y = [0. for _ in range(num_peds)]

    L_cov_sum_x = [0. for _ in range(num_peds)]
    L_cov_sum_y = [0. for _ in range(num_peds)]

    L_cov_sum_x = [0. for _ in range(num_peds)]
    L_cov_sum_y = [0. for _ in range(num_peds)]

    inv_cov_sum_x = [0. for _ in range(num_peds)]
    inv_cov_sum_y = [0. for _ in range(num_peds)]

    inv_var_sum_x = [0. for _ in range(num_peds)]
    inv_var_sum_y = [0. for _ in range(num_peds)]

    one_over_cov_sum_x = [0. for _ in range(num_peds)]
    one_over_cov_sum_y = [0. for _ in range(num_peds)]

    n = num_peds
    one_over_cov_sumij_x = [[0. for _ in range(n)] for _ in range(n)]
    one_over_cov_sumij_y = [[0. for _ in range(n)] for _ in range(n)]

    # one_over_std_sum_x = [0. for _ in range(num_peds)]
    # one_over_std_sum_y = [0. for _ in range(num_peds)]

    robot_mu_x = np.asarray(mu_linear_conditioned_x[num_peds])
    robot_mu_y = np.asarray(mu_linear_conditioned_y[num_peds])

    robot_cov_x = cov_linear_conditioned_x[num_peds]
    robot_cov_y = cov_linear_conditioned_y[num_peds]

    one_over_robot_cov_x = np.power(cov_linear_conditioned_x[num_peds], -1)
    one_over_robot_cov_y = np.power(cov_linear_conditioned_y[num_peds], -1)

    inv_var_robot_x = np.power(np.diag(robot_cov_x), -1)
    inv_var_robot_y = np.power(np.diag(robot_cov_y), -1)

    L_robot_cov_x = np.linalg.cholesky(robot_cov_x)
    L_robot_cov_y = np.linalg.cholesky(robot_cov_y)

    inv_cov_robot_x = np.dot(np.linalg.inv(L_robot_cov_x.T),
                             np.linalg.inv(L_robot_cov_x))
    inv_cov_robot_y = np.dot(np.linalg.inv(L_robot_cov_y.T),
                             np.linalg.inv(L_robot_cov_y))

    for ped in range(num_peds):
        if conditioned:
            ped_mu_x[ped] = mu_linear_conditioned_x[ped]
            ped_mu_y[ped] = mu_linear_conditioned_y[ped]

            ped_cov_x[ped] = cov_linear_conditioned_x[ped]
            ped_cov_y[ped] = cov_linear_conditioned_y[ped]
            # if linear:
            #     one_over_ped_cov_x[ped] = \
            #         np.diag(
            #             np.power(np.diag(cov_linear_conditioned_x[ped]), -1))
            #     one_over_ped_cov_y[ped] = \
            #         np.diag(
            #             np.power(np.diag(cov_linear_conditioned_y[ped]), -1))
            # else:
            #     one_over_ped_cov_x[ped] = np.power(
            #         cov_linear_conditioned_x[ped], -1)
            #     one_over_ped_cov_y[ped] = np.power(
            #         cov_linear_conditioned_y[ped], -1)
            one_over_ped_cov_x[ped] = np.power(
                cov_linear_conditioned_x[ped], -1)
            one_over_ped_cov_y[ped] = np.power(
                cov_linear_conditioned_y[ped], -1)
        else:
            ped_mu_x[ped] = mu_linear_un_x[ped]
            ped_mu_y[ped] = mu_linear_un_y[ped]

            ped_cov_x[ped] = cov_un_x[ped]
            ped_cov_y[ped] = cov_un_y[ped]

            one_over_ped_cov_x[ped] = np.power(cov_un_x[ped], -1)
            one_over_ped_cov_y[ped] = np.power(cov_un_y[ped], -1)

        L_ped_cov_x[ped] = np.linalg.cholesky(ped_cov_x[ped])
        L_ped_cov_y[ped] = np.linalg.cholesky(ped_cov_y[ped])

        inv_cov_ped_x[ped] = np.dot(np.linalg.inv(L_ped_cov_x[ped].T),
                                    np.linalg.inv(L_ped_cov_x[ped]))
        inv_cov_ped_y[ped] = np.dot(np.linalg.inv(L_ped_cov_y[ped].T),
                                    np.linalg.inv(L_ped_cov_y[ped]))
        cov_sum_x[ped] = np.add(robot_cov_x, ped_cov_x[ped])
        cov_sum_y[ped] = np.add(robot_cov_y, ped_cov_y[ped])

        L_cov_sum_x[ped] = np.linalg.cholesky(cov_sum_x[ped])
        L_cov_sum_y[ped] = np.linalg.cholesky(cov_sum_y[ped])

        inv_cov_sum_x[ped] = np.dot(np.linalg.inv(L_cov_sum_x[ped].T),
                                    np.linalg.inv(L_cov_sum_x[ped]))
        inv_cov_sum_y[ped] = np.dot(np.linalg.inv(L_cov_sum_y[ped].T),
                                    np.linalg.inv(L_cov_sum_y[ped]))
        one_over_cov_sum_x[ped] = np.power(cov_sum_x[ped], -1)
        one_over_cov_sum_y[ped] = np.power(cov_sum_y[ped], -1)

        # one_over_std_sum_x[ped] = np.power(cov_sum_x[ped], -0.5)
        # one_over_std_sum_y[ped] = np.power(cov_sum_y[ped], -0.5)

        inv_var_ped_x[ped] = np.power(np.diag(ped_cov_x[ped]), -1)
        inv_var_ped_y[ped] = np.power(np.diag(ped_cov_y[ped]), -1)

        inv_var_sum_x[ped] = np.power(np.diag(cov_sum_x[ped]), -1)
        inv_var_sum_y[ped] = np.power(np.diag(cov_sum_y[ped]), -1)
    eps = 1e-7
    for i in range(num_peds):
        for j in range(num_peds):
            sum_x = np.add(ped_cov_x[i], ped_cov_x[j])
            sum_y = np.add(ped_cov_y[i], ped_cov_y[j])
            sum_x = np.add(sum_x, eps)
            sum_y = np.add(sum_x, eps)
            one_over_cov_sumij_x[i][j] = np.power(sum_x, -1)
            one_over_cov_sumij_y[i][j] = np.power(sum_y, -1)

    return robot_mu_x, robot_mu_y, robot_cov_x, robot_cov_y, \
        inv_var_robot_x, inv_var_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
        ped_mu_x, ped_mu_y, ped_cov_x, ped_cov_y, \
        cov_sum_x, cov_sum_y, \
        inv_var_ped_x, inv_var_ped_y, \
        inv_cov_ped_x, inv_cov_ped_y, \
        inv_cov_sum_x, inv_cov_sum_y, \
        one_over_robot_cov_x, one_over_robot_cov_y, \
        one_over_ped_cov_x, one_over_ped_cov_y, \
        one_over_cov_sum_x, one_over_cov_sum_y, \
        one_over_cov_sumij_x, one_over_cov_sumij_y
    # one_over_std_sum_x, one_over_std_sum_y
