import numpy as np
from copy import deepcopy
import time


def distnav_gp_computation(frame, num_peds, x, y, x_obs, y_obs,
                          x_obs_un, y_obs_un, err_magnitude_ped, err_magnitude_robot,
                          end_point_err_ped, end_point_err_robot, buffer_robot, buffer_ped,
                          robot_start_x, robot_start_y, robot_goal_x, robot_goal_y,
                          cmd_x, cmd_y, obs_duration_robot, obs_duration_ped, Tdex, gp_x, gp_y):
    grid = np.linspace(0, 500, 501)

    x_obs_times = [0. for _ in range(num_peds+1)]
    y_obs_times = [0. for _ in range(num_peds+1)]

    x_obs_times_un = [0. for _ in range(num_peds+1)]
    y_obs_times_un = [0. for _ in range(num_peds+1)]

    err = [0. for _ in range(num_peds+1)]
    err_un = [0. for _ in range(num_peds+1)]
    # err corresponds to noise that the GP was trained on.
    # ERR IS THE NOISE THAT THE SENSORS ARE EXPERIENCING
    # 10*p2w = .28 meters.  That's the std of the sensors

    mu_linear_conditioned_x = np.array([[0. for _ in range(Tdex[0])]
                                        for _ in range(num_peds+1)], dtype=np.dtype(float))
    mu_linear_conditioned_y = np.array([[0. for _ in range(Tdex[0])]
                                        for _ in range(num_peds+1)], dtype=np.dtype(float))
    mu_linear_un_x = np.array([[0. for _ in range(Tdex[0])]
                               for _ in range(num_peds+1)], dtype=np.dtype(float))
    mu_linear_un_y = np.array([[0. for _ in range(Tdex[0])]
                               for _ in range(num_peds+1)], dtype=np.dtype(float))

    cov_linear_conditioned_x = [0. for _ in range(num_peds+1)]
    cov_linear_conditioned_y = [0. for _ in range(num_peds+1)]

    cov_un_x = [0. for _ in range(num_peds+1)]
    cov_un_y = [0. for _ in range(num_peds+1)]

    linear_goal_x = np.array([0. for _ in range(num_peds+1)],
                             dtype=np.dtype(float))
    linear_goal_y = np.array([0. for _ in range(num_peds+1)],
                             dtype=np.dtype(float))

    # joint_sample_x = np.zeros([num_peds+1, num_intents, Tdex[0]])
    # joint_sample_y = np.zeros([num_peds+1, num_intents, Tdex[0]])

    # var_samples_x = np.zeros([num_peds+1, 2*num_var_samples, Tdex[0]])
    # var_samples_y = np.zeros([num_peds+1, 2*num_var_samples, Tdex[0]])

    for ped in range(num_peds+1):
        if ped == num_peds:  # Robot GP
            time_gp = time.time()
            if buffer_robot == 1:
                if frame == 0:
                    x_obs[ped] = np.array([robot_start_x, robot_goal_x])
                    y_obs[ped] = np.array([robot_start_y, robot_goal_y])

                    x_obs_un[ped] = np.array([robot_start_x])
                    y_obs_un[ped] = np.array([robot_start_x])
                else:
                    x_obs[ped][0] = cmd_x
                    y_obs[ped][0] = cmd_y

                    x_obs_un[ped][0] = cmd_x
                    y_obs_un[ped][0] = cmd_y

                    x_obs[ped][-1] = robot_goal_x
                    y_obs[ped][-1] = robot_goal_y
            elif buffer_robot > 1:
                if frame == 0:
                    x_obs[ped] = np.array([robot_start_x, robot_goal_x])
                    y_obs[ped] = np.array([robot_start_y, robot_goal_y])

                    x_obs_un[ped] = robot_start_x
                    y_obs_un[ped] = robot_start_y
                elif frame > 0 and frame < buffer_robot:
                    x_obs[ped][-1] = cmd_x
                    y_obs[ped][-1] = cmd_y

                    x_obs_un[ped][-1] = cmd_x
                    y_obs_un[ped][-1] = cmd_y

                    x_obs[ped] = np.append(x_obs[ped], robot_goal_x)
                    y_obs[ped] = np.append(y_obs[ped], robot_goal_y)
                else:
                    x_obs[ped] = np.delete(x_obs[ped], 0)
                    y_obs[ped] = np.delete(y_obs[ped], 0)

                    x_obs_un[ped] = np.delete(x_obs_un[ped], 0)
                    y_obs_un[ped] = np.delete(y_obs_un[ped], 0)

                    x_obs[ped][-1] = cmd_x
                    y_obs[ped][-1] = cmd_y

                    x_obs_un[ped][-1] = cmd_x
                    y_obs_un[ped][-1] = cmd_y

                    x_obs[ped] = np.append(x_obs[ped], robot_goal_x)
                    y_obs[ped] = np.append(y_obs[ped], robot_goal_y)

            x_obs_times[ped] = np.linspace(0, obs_duration_robot,
                                           obs_duration_robot+1)
            y_obs_times[ped] = np.linspace(0, obs_duration_robot,
                                           obs_duration_robot+1)
            x_obs_times_un[ped] = np.linspace(0, obs_duration_robot-1,
                                              obs_duration_robot)
            y_obs_times_un[ped] = np.linspace(0, obs_duration_robot-1,
                                              obs_duration_robot)
            err[ped] = err_magnitude_robot*np.ones_like(x_obs_times[ped])
            err[ped][-1] = end_point_err_robot

            err_un[ped] = err_magnitude_robot*np.ones_like(x_obs_times_un[ped])

            x_obs_times[ped][-1] = Tdex[ped]
            y_obs_times[ped][-1] = Tdex[ped]

            gp_x[ped].compute(x_obs_times_un[ped], err_un[ped])
            gp_y[ped].compute(y_obs_times_un[ped], err_un[ped])

            mu_linear_un_x[ped], cov_un_x[ped] = gp_x[ped].predict(
                x_obs_un[ped],
                grid[(len(x_obs_un[ped])-1):Tdex[ped]+(len(x_obs_un[ped])-1)],
                return_cov=True)
            mu_linear_un_y[ped], cov_un_y[ped] = \
                gp_y[ped].predict(y_obs_un[ped],
                                  grid[(len(x_obs_un[ped])-1)
                                        :Tdex[ped]+(len(x_obs_un[ped])-1)],
                                  return_cov=True)
            gp_x[ped].compute(x_obs_times[ped], err[ped])
            gp_y[ped].compute(y_obs_times[ped], err[ped])

            mu_linear_conditioned_x[ped], cov_linear_conditioned_x[ped] = \
                gp_x[ped].predict(x_obs[ped],
                                  grid[(len(x_obs[ped])-1):Tdex[ped] +
                                       (len(x_obs[ped])-1)],
                                  return_cov=True)
            mu_linear_conditioned_y[ped], cov_linear_conditioned_y[ped] = \
                gp_y[ped].predict(y_obs[ped],
                                  grid[(len(x_obs[ped])-1):Tdex[ped] +
                                       (len(x_obs[ped])-1)],
                                  return_cov=True)
            time_gp = time.time() - time_gp

            # j = 0
            # for var in range(num_var_samples):
                # if var = 1/var_ratio:
                #   var_samples_x_ess[ped, j, :] = mu_linear_conditioned_x[ped] + \
                #                                    (var+1)*var_ratio*np.sqrt(np.diag(\
                #                                        cov_linear_conditioned_x[ped]))
                #   var_samples_y_ess[ped, j,:] = mu_linear_conditioned_y[ped] + \
                #                             (var+1)*var_ratio*np.sqrt(np.diag(\
                #                                  cov_linear_conditioned_y[ped]))
                # var_samples_x[ped, j, :] = mu_linear_conditioned_x[ped] + \
                #     (var+1)*var_ratio*np.sqrt(np.diag(
                #         cov_linear_conditioned_x[ped]))
                # var_samples_y[ped, j, :] = mu_linear_conditioned_y[ped] + \
                #     (var+1)*var_ratio*np.sqrt(np.diag(
                #         cov_linear_conditioned_y[ped]))
                # var_samples_x[ped, j+1, :] = mu_linear_conditioned_x[ped] - \
                #     (var+1)*var_ratio*np.sqrt(np.diag(
                #         cov_linear_conditioned_x[ped]))
                # var_samples_y[ped, j+1, :] = mu_linear_conditioned_y[ped] - \
                #     (var+1)*var_ratio*np.sqrt(np.diag(
                #         cov_linear_conditioned_y[ped]))
                # j = j + 2

            # joint_sample_x[ped, :, :] = gp_x[ped].sample_conditional(x_obs[ped],
            #                                                          grid[(
            #                                                              len(x_obs[ped])-1):Tdex[ped]+(len(x_obs[ped])-1)],
            #                                                          size=num_intents)
            # joint_sample_y[ped, :, :] = gp_y[ped].sample_conditional(y_obs[ped],
            #                                                          grid[(
            #                                                              len(x_obs[ped])-1):Tdex[ped]+(len(x_obs[ped])-1)],
            #                                                          size=num_intents)
        else:  # Pedestrian GPs
            if frame < buffer_ped:
                x_obs[ped] = deepcopy(x[ped][0:frame+1])
                y_obs[ped] = deepcopy(y[ped][0:frame+1])

                x_obs_un[ped] = deepcopy(x[ped][0:frame+1])
                y_obs_un[ped] = deepcopy(y[ped][0:frame+1])
            else:
                x_obs[ped] = deepcopy(x[ped][(frame-buffer_ped):(frame+1)])
                y_obs[ped] = deepcopy(y[ped][(frame-buffer_ped):(frame+1)])

                x_obs_un[ped] = deepcopy(x[ped][(frame-buffer_ped):(frame+1)])
                y_obs_un[ped] = deepcopy(y[ped][(frame-buffer_ped):(frame+1)])

            ave_vel_x = 0.
            ave_vel_y = 0.

            if len(x_obs[ped]) <= 1:
                linear_goal_x[ped] = np.random.normal(0, .1) + x_obs[ped][-1]
                linear_goal_y[ped] = np.random.normal(0, .1) + y_obs[ped][-1]

                x_obs[ped] = np.append(x_obs[ped], linear_goal_x[ped])
                y_obs[ped] = np.append(y_obs[ped], linear_goal_y[ped])
            else:
                for t in range(len(x_obs[ped])-1):
                    ave_vel_x += x_obs[ped][t+1] - x_obs[ped][t]
                    ave_vel_y += y_obs[ped][t+1] - y_obs[ped][t]

                ave_vel_x = ave_vel_x/(len(x_obs[ped])-1.)
                ave_vel_y = ave_vel_y/(len(x_obs[ped])-1.)

                ave_vel_x = ave_vel_x/4.
                ave_vel_y = ave_vel_y/4.

                if frame < buffer_ped:
                    linear_goal_x[ped] = ave_vel_x*Tdex[ped] + x_obs[ped][-1]
                    linear_goal_y[ped] = ave_vel_y*Tdex[ped] + y_obs[ped][-1]

                    x_obs[ped] = np.append(x_obs[ped], linear_goal_x[ped])
                    y_obs[ped] = np.append(y_obs[ped], linear_goal_y[ped])
                else:
                    linear_goal_x[ped] = ave_vel_x*Tdex[ped] + x_obs[ped][-1]
                    linear_goal_y[ped] = ave_vel_y*Tdex[ped] + y_obs[ped][-1]

                    x_obs[ped] = np.delete(x_obs[ped], 0)
                    y_obs[ped] = np.delete(y_obs[ped], 0)

                    x_obs_un[ped] = np.delete(x_obs_un[ped], 0)
                    y_obs_un[ped] = np.delete(y_obs_un[ped], 0)

                    x_obs[ped] = np.append(x_obs[ped], linear_goal_x[ped])
                    y_obs[ped] = np.append(y_obs[ped], linear_goal_y[ped])

            x_obs_times[ped] = np.linspace(
                0, obs_duration_ped, obs_duration_ped+1)
            y_obs_times[ped] = np.linspace(
                0, obs_duration_ped, obs_duration_ped+1)

            x_obs_times_un[ped] = np.linspace(
                0, obs_duration_ped-1, obs_duration_ped)
            y_obs_times_un[ped] = np.linspace(
                0, obs_duration_ped-1, obs_duration_ped)

            x_obs_times[ped][-1] = Tdex[ped]
            y_obs_times[ped][-1] = Tdex[ped]

            err[ped] = err_magnitude_ped*np.ones_like(x_obs_times[ped])
            err[ped][-1] = end_point_err_ped

            err_un[ped] = err_magnitude_ped*np.ones_like(x_obs_times_un[ped])
# GP NOT conditioned on goal
            gp_x[ped].compute(x_obs_times_un[ped], err_un[ped])
            gp_y[ped].compute(y_obs_times_un[ped], err_un[ped])

            mu_linear_un_x[ped], cov_un_x[ped] = gp_x[ped].predict(
                x_obs_un[ped],
                grid[(len(x_obs_un[ped])-1):Tdex[ped]+(len(x_obs_un[ped])-1)],
                return_cov=True)
            mu_linear_un_y[ped], cov_un_y[ped] = gp_y[ped].predict(
                y_obs_un[ped],
                grid[(len(x_obs_un[ped])-1):Tdex[ped]+(len(x_obs_un[ped])-1)],
                return_cov=True)
            # if dwa:
            #     cov_un_x[ped] = cov_eps*np.eye(np.shape(cov_un_x[ped])[0])
            #     cov_un_y[ped] = cov_eps*np.eye(np.shape(cov_un_y[ped])[0])
            # if linear_diag:
            #     cov_un_x[ped] = np.diag(np.diag(cov_un_x[ped]))
            #     cov_un_y[ped] = np.diag(np.diag(cov_un_y[ped]))
# GP conditioned on goal
            gp_x[ped].compute(x_obs_times[ped], err[ped])
            gp_y[ped].compute(y_obs_times[ped], err[ped])

            mu_linear_conditioned_x[ped], cov_linear_conditioned_x[ped] = \
                gp_x[ped].predict(x_obs[ped],
                                  grid[(len(x_obs[ped])-1):Tdex[ped] +
                                       (len(x_obs[ped])-1)],
                                  return_cov=True)
            mu_linear_conditioned_y[ped], cov_linear_conditioned_y[ped] = \
                gp_y[ped].predict(y_obs[ped],
                                  grid[(len(x_obs[ped])-1):Tdex[ped] +
                                       (len(x_obs[ped])-1)],
                                  return_cov=True)
            # if dwa:
            #     cov_linear_conditioned_x[ped] = \
            #         cov_eps*np.eye(np.shape(cov_linear_conditioned_x[ped])[0])
            #     cov_linear_conditioned_y[ped] = \
            #         cov_eps*np.eye(np.shape(cov_linear_conditioned_y[ped])[0])
            # if linear_diag:
            #     cov_linear_conditioned_x[ped] = \
            #         np.diag(np.diag(cov_linear_conditioned_x[ped]))
            #     cov_linear_conditioned_y[ped] = \
            #         np.diag(np.diag(cov_linear_conditioned_y[ped]))

            # j = 0
            # for var in range(num_var_samples):
            #     var_samples_x[ped, j, :] = mu_linear_conditioned_x[ped] + \
            #         (var+1)*var_ratio*var_ratio*np.sqrt(np.diag(
            #             cov_linear_conditioned_x[ped]))
            #     var_samples_y[ped, j, :] = mu_linear_conditioned_y[ped] + \
            #         (var+1)*var_ratio*var_ratio*np.sqrt(np.diag(
            #             cov_linear_conditioned_y[ped]))
            #     var_samples_x[ped, j+1, :] = mu_linear_conditioned_x[ped] - \
            #         (var+1)*var_ratio*var_ratio*np.sqrt(np.diag(
            #             cov_linear_conditioned_x[ped]))
            #     var_samples_y[ped, j+1, :] = mu_linear_conditioned_y[ped] - \
            #         (var+1)*var_ratio*var_ratio*np.sqrt(np.diag(
            #             cov_linear_conditioned_y[ped]))
            #     j = j + 2

            # joint_sample_x[ped, :, :] = gp_x[ped].sample_conditional(x_obs[ped],
            #                                                          grid[(
            #                                                              len(x_obs[ped])-1):Tdex[ped]+(len(x_obs[ped])-1)],
            #                                                          size=num_intents)
            # joint_sample_y[ped, :, :] = gp_y[ped].sample_conditional(y_obs[ped],
            #                                                          grid[(
            #                                                              len(x_obs[ped])-1):Tdex[ped]+(len(x_obs[ped])-1)],
            #                                                          size=num_intents)
    return gp_x, gp_y, mu_linear_conditioned_x, mu_linear_conditioned_y, \
        mu_linear_un_x, mu_linear_un_y, \
        cov_linear_conditioned_x, cov_linear_conditioned_y, \
        cov_un_x, cov_un_y, x_obs, y_obs,\
        x_obs_un, y_obs_un, time_gp
