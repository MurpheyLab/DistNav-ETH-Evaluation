# GENERIC LIBRARIES
from copy import deepcopy
import math
import time
# SPECIAL LIBRARIES
# import scipy as sp
# import george
# import autograd as ag
# from autograd import value_and_grad
# reverse mode is more efficiet for scalar valued functions
# import autograd.numpy as np
import numpy as np
# import numpy.random as npr
# from autograd.numpy.linalg import solve
# from numpy.linalg import solve
# import mpmath as mp

from scipy.stats import multivariate_normal as mvn

# import igraph
# UTILITY LIBRARIES

from .eth_Tdex import eth_Tdex
from .distnav_gp_computation import distnav_gp_computation
from .distnav_rename import rename

from .distnav_importance_scores import importance_scores
from .distnav_compute import vigp_compute
from .distnav_interact import interact
from .distnav_generate_sample import generate_sample


def nav(a, h, score_thred, num_samples, cov_scale, obj_thred, 
        max_iter, include_pdf_weight, coll_weight, Tdex_max, frame,
        num_peds, max_vel_robot, max_vel_ped, p2w_x, p2w_y,
        robot_start_x, robot_start_y, robot_goal_x, robot_goal_y,
        vel_x, vel_y, cmd_x, cmd_y,
        x, y, x_obs, y_obs, x_obs_un, y_obs_un,
        err_magnitude_ped, err_magnitude_robot, end_point_err_ped,
        end_point_err_robot, buffer_robot, buffer_ped, obs_duration_robot,
        obs_duration_ped, gp_x, gp_y,
        ess_time_array,
        ess_array, conditioned, data_set, support_boost,
        goal_dex, x_nonzero, y_nonzero, 
        normal_vel, full_traj):
    # x,y is x_follow, y_follow
    # TDEX
    if frame == 0:
        Tdex, robot_goal_x, robot_goal_y \
            = eth_Tdex(Tdex_max, frame, num_peds, max_vel_robot,
                          robot_start_x, robot_start_y,
                          robot_goal_x, robot_goal_y,
                          0, 0, 0, 0,
                          data_set, support_boost,
                          goal_dex, x_nonzero, y_nonzero,
                          normal_vel, full_traj)
    else:
        Tdex, robot_goal_x, robot_goal_y \
            = eth_Tdex(Tdex_max, frame, num_peds, max_vel_robot,
                          robot_start_x, robot_start_y,
                          robot_goal_x, robot_goal_y,
                          vel_x, vel_y, cmd_x, cmd_y,
                          data_set, support_boost,
                          goal_dex, x_nonzero, y_nonzero,
                          normal_vel, full_traj)
    # print('Tdex:\n', Tdex)

    # start_time = time.time()
    # GP COMPUTATION
    if frame == 0:
        gp_x, gp_y, mu_linear_conditioned_x, mu_linear_conditioned_y, \
        mu_linear_un_x, mu_linear_un_y, \
        cov_linear_conditioned_x, cov_linear_conditioned_y, cov_un_x, cov_un_y, \
        x_obs, y_obs, x_obs_un, y_obs_un, time_gp \
            = distnav_gp_computation(frame, num_peds, x, y, x_obs, y_obs,
                                    x_obs_un, y_obs_un, err_magnitude_ped, err_magnitude_robot,
                                    end_point_err_ped, end_point_err_robot, buffer_robot, buffer_ped,
                                    robot_start_x, robot_start_y, robot_goal_x, robot_goal_y,
                                    0, 0, obs_duration_robot, obs_duration_ped, Tdex, gp_x, gp_y)
    else:
        gp_x, gp_y, mu_linear_conditioned_x, mu_linear_conditioned_y, \
        mu_linear_un_x, mu_linear_un_y, \
        cov_linear_conditioned_x, cov_linear_conditioned_y, cov_un_x, cov_un_y, \
        x_obs, y_obs, x_obs_un, y_obs_un, time_gp \
            = distnav_gp_computation(frame, num_peds, x, y, x_obs, y_obs,
                                    x_obs_un, y_obs_un, err_magnitude_ped, err_magnitude_robot,
                                    end_point_err_ped, end_point_err_robot, buffer_robot, buffer_ped,
                                    robot_start_x, robot_start_y, robot_goal_x, robot_goal_y,
                                    cmd_x, cmd_y, obs_duration_robot, obs_duration_ped,
                                    Tdex, gp_x, gp_y)

    # RENAMING
    robot_mu_x, robot_mu_y, robot_cov_x, robot_cov_y, \
    inv_var_robot_x, inv_var_robot_y, inv_cov_robot_x, inv_cov_robot_y, \
    ped_mu_x, ped_mu_y, ped_cov_x, ped_cov_y, cov_sum_x, cov_sum_y, \
    inv_var_ped_x, inv_var_ped_y, inv_cov_ped_x, inv_cov_ped_y, \
    inv_cov_sum_x, inv_cov_sum_y, one_over_robot_cov_x, one_over_robot_cov_y, \
    one_over_ped_cov_x, one_over_ped_cov_y, \
    one_over_cov_sum_x, one_over_cov_sum_y, \
    one_over_cov_sumij_x, one_over_cov_sumij_y = rename(num_peds, conditioned,
                                                        mu_linear_conditioned_x, mu_linear_conditioned_y,
                                                        mu_linear_un_x, mu_linear_un_y,
                                                        cov_linear_conditioned_x, cov_linear_conditioned_y,
                                                        cov_un_x, cov_un_y)

    #######################################################################################################
    # IGP Computation
    start_time = time.time()
    
    # print('robot_mu_x type/shape:', type(robot_mu_x), robot_mu_x.shape)
    # print('robot_cov_x shape: ', robot_cov_x.shape)
    # print('ped_mu_x type/len:', type(ped_mu_x), len(ped_mu_x), ped_mu_x[0].shape)
    # print('ped_cov_x type/len:', type(ped_cov_x), len(ped_cov_x), ped_cov_x[0].shape)
    # print('mu_linear_conditioned_x len:', len(mu_linear_conditioned_x))

    # Filter out nonzero mean and cov
    nonzero_ped_idx = np.nonzero(np.sum(ped_mu_x, axis=1))[0]
    nonzero_ped_idx_robot = np.concatenate((nonzero_ped_idx, [num_peds]))
    num_peds_nonzero = len(nonzero_ped_idx)
    print('nonzero_ped_idx: ', len(nonzero_ped_idx))

    # Generate samples for each agent
    scale = cov_scale
    # num_samples = 500
    traj_len = len(robot_mu_x)
    agents_sample_x, agents_sample_y, agents_sample_x_nonzero, agents_sample_y_nonzero, \
    agents_pdf_x, agents_pdf_y, agents_pdf_x_nonzero, agents_pdf_y_nonzero = \
        generate_sample(robot_mu_x, robot_cov_x, robot_mu_y, robot_cov_y, ped_mu_x, ped_cov_x, ped_mu_y, ped_cov_y,
                        traj_len, num_peds, num_samples, nonzero_ped_idx, scale)
    robot_rv_x = agents_sample_x[num_samples * num_peds:num_samples * (num_peds + 1)]
    robot_rv_y = agents_sample_y[num_samples * num_peds:num_samples * (num_peds + 1)]
    # print('test pdf list: ')
    # print(agents_pdf_x[0])
    # print("max robot cov: ", np.max(np.diag(robot_cov_x)))
    # print("robot cov: ", robot_cov_x)

    agents_mu_x = deepcopy(ped_mu_x)
    agents_mu_x.append(robot_mu_x)
    agents_mu_x = np.array(agents_mu_x)
    agents_mu_x_nonzero = agents_mu_x[nonzero_ped_idx_robot.astype(int)]
    # print(agents_mu_x_nonzero.shape)

    agents_mu_y = deepcopy(ped_mu_y)
    agents_mu_y.append(robot_mu_y)
    agents_mu_y = np.array(agents_mu_y)
    agents_mu_y_nonzero = agents_mu_y[nonzero_ped_idx_robot.astype(int)]

    # start_time = time.time()
    print("select important agents.")
    influence_scores = importance_scores(agents_mu_x_nonzero, agents_mu_y_nonzero,
                                         agents_sample_x_nonzero, agents_sample_y_nonzero,
                                         traj_len, num_peds_nonzero, num_samples,
                                         p2w_x, p2w_y, a, h)
    print('importance score computation time: ', time.time() - start_time)
    influence_scores[np.diag_indices(num_peds_nonzero + 1)] = np.zeros(num_peds_nonzero + 1, dtype=np.float32)
    # print('influence_scores: ', influence_scores[num_peds_nonzero])
    # print(influence_scores)
    graph_adjacency_mat = (influence_scores > score_thred).astype(np.int32)

    #########
    # clustering just for the robot node (depth: 1)
    robot_influence_nodes = np.nonzero(graph_adjacency_mat[num_peds_nonzero])[0]
    robot_influence_idx = nonzero_ped_idx[robot_influence_nodes]
    essential_cluster = np.concatenate((robot_influence_idx, [num_peds]))
    print('essential cluster: ', essential_cluster)
    ess = len(essential_cluster) - 1
    ess_array[frame] = ess

    #######################################################################################################
    # Real IGP Compuation

    # extract the samples based on essential cluster
    essential_num = len(essential_cluster)
    essential_samples_x = np.zeros((num_samples * essential_num, traj_len))
    essential_samples_y = np.zeros((num_samples * essential_num, traj_len))
    for i in range(essential_num):
        idx = essential_cluster[i]
        essential_samples_x[i * num_samples:(i + 1) * num_samples] = agents_sample_x[
                                                                     idx * num_samples:(idx + 1) * num_samples]
        essential_samples_y[i * num_samples:(i + 1) * num_samples] = agents_sample_y[
                                                                     idx * num_samples:(idx + 1) * num_samples]
    # print('essential samples shape: ', essential_samples_x.shape)
    essential_pdf_x = agents_pdf_x[essential_cluster]
    essential_pdf_y = agents_pdf_y[essential_cluster]

    robot_eql_traj_x = np.zeros((num_samples, traj_len))
    robot_eql_traj_y = np.zeros((num_samples, traj_len))
    robot_eql_idx = np.zeros(num_samples)

    if essential_num != 1:
        # compute IGP weights
        print("compute IGP weights")
        weights = vigp_compute(essential_samples_x, essential_samples_y, essential_pdf_x, essential_pdf_y,
                               essential_num, num_samples, traj_len, p2w_x, p2w_y, a, h, obj_thred, max_iter,
                               coll_weight)
        if include_pdf_weight:  # times weights with original pdf for selection
            # extract optimal robot trajectory
            robot_pdf = agents_pdf_x[-1] * agents_pdf_y[-1]
            opt_idx = np.argmax(robot_pdf * weights[-1])
            opt_robot_traj_x = agents_sample_x[num_samples * num_peds:num_samples * (num_peds + 1)][opt_idx].copy()
            opt_robot_traj_y = agents_sample_y[num_samples * num_peds:num_samples * (num_peds + 1)][opt_idx].copy()
            robot_eql_traj_x = opt_robot_traj_x.copy()
            robot_eql_traj_y = opt_robot_traj_y.copy()
            # extract optimal pedestrian trajectories
            opt_joint_traj_x = np.zeros((essential_num, traj_len))
            opt_joint_traj_y = np.zeros((essential_num, traj_len))
            for i in range(essential_num):
                agent_idx = essential_cluster[i]
                single_agent_pdf = agents_pdf_x[agent_idx] * agents_pdf_y[agent_idx]
                opt_idx = np.argmax(single_agent_pdf * weights[i])
                opt_joint_traj_x[i] = agents_sample_x[num_samples * agent_idx:num_samples * (agent_idx + 1)][
                    opt_idx].copy()
                opt_joint_traj_y[i] = agents_sample_y[num_samples * agent_idx:num_samples * (agent_idx + 1)][
                    opt_idx].copy()
        else:
            # extract optimal robot trajectory
            # robot_pdf = agents_pdf_x[-1] * agents_pdf_x[-1]
            opt_idx = np.argmax(weights[-1])
            opt_robot_traj_x = agents_sample_x[num_samples * num_peds:num_samples * (num_peds + 1)][opt_idx].copy()
            opt_robot_traj_y = agents_sample_y[num_samples * num_peds:num_samples * (num_peds + 1)][opt_idx].copy()
            robot_eql_traj_x = opt_robot_traj_x.copy()
            robot_eql_traj_y = opt_robot_traj_y.copy()
            # extract optimal pedestrian trajectories
            opt_joint_traj_x = np.zeros((essential_num, traj_len))
            opt_joint_traj_y = np.zeros((essential_num, traj_len))
            for i in range(essential_num):
                agent_idx = essential_cluster[i]
                # single_agent_pdf = agents_pdf_x[agent_idx] * agents_pdf_y[agent_idx]
                opt_idx = np.argmax(weights[i])
                opt_joint_traj_x[i] = agents_sample_x[num_samples * agent_idx:num_samples * (agent_idx + 1)][
                    opt_idx].copy()
                opt_joint_traj_y[i] = agents_sample_y[num_samples * agent_idx:num_samples * (agent_idx + 1)][
                    opt_idx].copy()
    else:
        opt_robot_traj_x = np.array([robot_mu_x.copy()])
        opt_robot_traj_y = np.array([robot_mu_y.copy()])
        opt_joint_traj_x = np.array([robot_mu_x.copy()])
        opt_joint_traj_y = np.array([robot_mu_y.copy()])
        robot_eql_idx = np.zeros(num_samples)
        robot_eql_traj_x = np.array([robot_mu_x.copy() for _ in range(num_samples)])
        robot_eql_traj_y = np.array([robot_mu_y.copy() for _ in range(num_samples)])

    # process and store optimization time
    ess_time = time.time() - start_time
    ess_time_array[frame] = ess_time
    ess_ave_time = math.trunc(1e3 * np.mean(ess_time_array[:frame + 1])) / 1e3
    ess_std_time = math.trunc(1e3 * np.std(ess_time_array[:frame + 1])) / 1e3

    print('e_igp_compute elapsed time: ', time.time() - start_time, time_gp * ess)

    time_gp = time_gp * (num_peds_nonzero + 1)

    #######################################################################################################

    return robot_goal_x, robot_goal_y, gp_x, gp_y, x_obs, y_obs, \
           x_obs_un, y_obs_un, robot_mu_x, robot_mu_y, ped_mu_x, ped_mu_y, \
           robot_cov_x, robot_cov_y, ped_cov_x, ped_cov_y, \
           nonzero_ped_idx, nonzero_ped_idx_robot, influence_scores, essential_cluster, \
           robot_rv_x, robot_rv_y, opt_robot_traj_x, opt_robot_traj_y, \
           opt_joint_traj_x, opt_joint_traj_y, robot_eql_idx.copy(), robot_eql_traj_x.copy(), robot_eql_traj_y.copy(), \
           ess, ess_array, ess_time, ess_time_array, ess_ave_time, ess_std_time, time_gp

    # ess, top_Z_indices, ess_array, ess_time, ess_time_array, \
    # ess_ave_time, \
    # ess_std_time, optima, optimal_ll, optima_dex, num_optima, \
    # norm_likelihood, global_optima_dex, time_gp, \
    # agent_disrupt, robot_agent_disrupt
