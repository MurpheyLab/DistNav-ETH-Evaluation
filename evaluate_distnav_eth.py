import time
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import math
import datetime
from copy import deepcopy

from utils._eth_runs import run_data
from utils._import_eth_data import import_eth_data
from utils._eth_start_goal import start_goal
from utils._eth_compute_ped_density import compute_ped_density
from utils._eth_metrics import metrics
from utils._eth_save_figs import save_figs
from utils._eth_stopping_condition import stopping_condition
from utils._eth_plotter_single import plotter_single
from utils._eth_save_data import save_data
from utils._eth_print_data import print_data

from distnav.distnav_gp_init import gp_init
from distnav.distnav_run import nav
from distnav.distnav_actuate import actuate

####################CREATE FIGURES
fig = plt.figure(
    num=None, figsize=(4.2, 4.2), dpi=200, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)

################RUN TIME PARAMETERS
home_dir = os.path.dirname(os.path.abspath(__file__))
date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

################SAVING AND PRINTING
save_dir = 'distnav-'+date_time
boost_factor = 1.2
saving = False
saving_final = False
save_pdf = False
printing = True
plotting = True

################TESTING PARAMETERS
data_set = 'eth_train'
conditioned = True

actuate_distance = True
actuate_to_step = False
actuate_to_index = False

################AGENT DATA
run_index = -1
for k in run_data():
    run_index += 1
    # if(run_index < 94):
    #     continue
    # if(str(k).split('_')[1] != '57'):
    #     continue
    # if k != 'run_20_full':
    #     continue

    remove_ped, remove_ped_start, goal_dex, max_vel_robot, full_traj \
        = run_data()[k]
    max_vel_ped = 6.74

    goal_noise_multiplier_robot = 1. / 1.
    goal_noise_multiplier_ped = 10.

    normal_vel = 12.
    support_boost = math.trunc((max_vel_robot / normal_vel) * boost_factor * 1e3) / 1e3

    radius = 3.
    show_radius = False

    x, y, x_follow, y_follow, num_peds_real, num_peds_follow, num_frames, \
    p2w_x, p2w_y, p2w, scaling \
        = import_eth_data(data_set, remove_ped, home_dir, remove_ped_start)

    x_nonzero = deepcopy(x[remove_ped][np.nonzero(x[remove_ped][:])])
    y_nonzero = deepcopy(y[remove_ped][np.nonzero(y[remove_ped][:])])

    sld = np.linalg.norm([x_nonzero[0] - x_nonzero[-1], \
                          y_nonzero[0] - y_nonzero[-1]])
    sld_now = np.linalg.norm([p2w_x * (x_nonzero[0] - x_nonzero[goal_dex]), \
                              p2w_y * (y_nonzero[0] - y_nonzero[goal_dex])])

    ####################IMPORT START AND GOAL INITIAL DATA
    robot_start_x, robot_start_y, robot_goal_x, robot_goal_y, Tdex_max, \
    remove_ped_path_length = start_goal(x_nonzero, y_nonzero, remove_ped, \
                                        sld, p2w_x, p2w_y, p2w, \
                                        data_set, goal_dex, full_traj)
    robot_history_x = robot_start_x
    robot_history_y = robot_start_y
    
    ####################ARRAYS
    x_obs = {}
    y_obs = {}

    x_obs_un = {}
    y_obs_un = {}

    ess_array = [0. for _ in range(num_frames)]

    ess_time_array = [0. for _ in range(num_frames)]

    safety_robot = [0. for _ in range(num_frames)]
    safety_remove_ped = [0. for _ in range(num_frames)]
    robot_agent_path_diff = [0. for _ in range(num_frames)]

    local_density = [0. for _ in range(num_frames)]

    time_list = [0. for _ in range(num_frames)]

    agent_disrupt = [0. for _ in range(num_frames)]
    robot_agent_disrupt = [0. for _ in range(num_frames)]
    
    ####################GP PARAMETERS
    buffer_ped = 2
    buffer_robot = 1

    obs_duration_robot = 0
    obs_duration_ped = 0
    # ERR MUST MATCH THE HYPERPARAMETERS
    # magic: 2 and
    err_magnitude_ped = 2.
    err_magnitude_robot = 5.

    end_point_err_ped = goal_noise_multiplier_ped * err_magnitude_ped
    end_point_err_robot = goal_noise_multiplier_robot * err_magnitude_robot
    
    ####################INIT GPs
    os.chdir(str(home_dir) + '/utils/gp_hyperparams_pixels/k12/')
    gp_x, gp_y = gp_init(num_peds_follow, home_dir)
    os.chdir(str(home_dir))
    
    ####################BEGIN SIMULATION
    for frame in range(num_frames):
        print(k)
        ####################OBS_DURATION
        if obs_duration_robot < buffer_robot:
            obs_duration_robot = obs_duration_robot + 1
        else:
            obs_duration_robot = buffer_robot
        if obs_duration_ped < buffer_ped:
            obs_duration_ped = obs_duration_ped + 1
        else:
            obs_duration_ped = buffer_ped

        if frame == 0:
            vel_x = 0.
            vel_y = 0.
            cmd_x = 0.
            cmd_y = 0.
        #######################IGP
        alpha = 0.1 # safety region
        h = 10.0 # safety weight
        score_thred = 0.2 # threshold for selecting critical agents
        num_samples = 100 # number of samples per agent
        cov_scale = 0.2 # scale to restrict GP covariance
        obj_thred = 0.0001 # threshold for terminating optimization
        max_iter = 150 # maximal number of iterations allowed, increase it with more agents involved
        coll_weight = None # forget about it at now :) this is for setting the KL-divergence constraint always wrt the original preference, instead of previous iteration, impose more strict constraint on preference deviation
        include_pdf_weight = True # include original GP probability density when selecting final trajectory
        
        robot_goal_x, robot_goal_y, gp_x, gp_y, x_obs, y_obs, x_obs_un, \
        y_obs_un, robot_mu_x, robot_mu_y, ped_mu_x, ped_mu_y, \
        robot_cov_x, robot_cov_y, ped_cov_x, ped_cov_y, \
        nonzero_ped_idx, nonzero_ped_idx_robot, influence_scores, essential_cluster, \
        robot_rv_x, robot_rv_y, opt_robot_traj_x, opt_robot_traj_y, \
        opt_joint_traj_x, opt_joint_traj_y, robot_eql_idx, robot_eql_traj_x, robot_eql_traj_y, \
        ess, ess_array, ess_time, ess_time_array, ess_ave_time, ess_std_time, time_gp \
            = nav(alpha, h, score_thred, num_samples, cov_scale, obj_thred, 
                  max_iter, include_pdf_weight, coll_weight, Tdex_max, frame,
                  num_peds_follow, max_vel_robot, max_vel_ped, p2w_x, p2w_y,
                  robot_start_x, robot_start_y, robot_goal_x, robot_goal_y,
                  vel_x, vel_y, cmd_x, cmd_y,
                  x_follow, y_follow, x_obs, y_obs, x_obs_un, y_obs_un,
                  err_magnitude_ped, err_magnitude_robot, end_point_err_ped,
                  end_point_err_robot, buffer_robot, buffer_ped, obs_duration_robot,
                  obs_duration_ped, gp_x, gp_y,
                  ess_time_array,
                  ess_array, conditioned, data_set, support_boost,
                  goal_dex, x_nonzero, y_nonzero, 
                  normal_vel, full_traj)

        time_list[frame] = ess_time_array[frame] + time_gp

        ####################ACTUATE
        T = np.size(robot_mu_x)
        # if opt_iter_robot or opt_iter_all:
        #     a = optima[global_optima_dex]
        # else:
        #     a = optima[global_optima_dex].x
        a = np.concatenate((opt_joint_traj_x[-1], opt_joint_traj_y[-1]))

        robot_history_x, robot_history_y, cmd_x, cmd_y, vel_x, vel_y \
            = actuate(a, T, x_obs, y_obs, max_vel_robot, robot_history_x,
                      robot_history_y, frame, x_follow, y_follow, num_peds_follow,
                      p2w_x, p2w_y, p2w, actuate_distance, actuate_to_step, actuate_to_index)

        ####################DENSITY
        local_density[frame] = compute_ped_density(frame, cmd_x, cmd_y,
                                                   x_follow, y_follow, radius, num_peds_follow, p2w_x, p2w_y)

        ####################COMPUTE METRICS
        safety_robot[frame], safety_remove_ped[frame], robot_path_length, \
        robot_agent_path_diff[frame], remove_ped_path_length \
            = metrics(frame, num_peds_follow, remove_ped, cmd_x, cmd_y, x_follow,
                      y_follow, x_nonzero, y_nonzero, robot_history_x, robot_history_y,
                      remove_ped_path_length, p2w_x, p2w_y, p2w)

        ####################STOPPING CONDITION
        # remove non-existing arguments: top_Z_indices, num_optima, optima_dex, optimal_ll, optima
        stop = stopping_condition(frame, remove_ped, home_dir, 
                                  p2w_x, p2w_y, full_traj, robot_goal_x,
                                  robot_goal_y, cmd_x, cmd_y, saving_final, ess, ess_time, ess_ave_time,
                                  ess_std_time, time_list, local_density, safety_remove_ped[:frame],
                                  safety_robot[:frame], robot_agent_path_diff,
                                  remove_ped_path_length, robot_path_length, sld_now, save_dir)
        if stop:
            break
        ####################PLOTTING
        # removed non-existing arguments: optima, optima_dex, num_optima, top_Z_indices
        if plotting:
            plotter_single(fig, ax, x_follow, y_follow, x_nonzero, y_nonzero,
                        frame, num_peds_follow, 
                        robot_mu_x, robot_mu_y,
                        robot_history_x, robot_history_y, cmd_x, cmd_y, ess_time, ess_ave_time,
                        ess_std_time, ess, time_list, data_set,
                        p2w_x, p2w_y, p2w, radius, show_radius, scaling, 
                        opt_joint_traj_x, opt_joint_traj_y)
        ####################PRINT DATA
        if printing:
            if frame == 0:
                a = safety_remove_ped[frame]
                b = safety_robot[frame]
            else:
                a = safety_remove_ped[:frame]
                b = safety_robot[:frame]
            # removed non-existing arguments: top_Z_indices, num_optima, optimal_ll, optima_dex, norm_likelihood, optima
            print_data(ess, frame,
                       ess_time, ess_ave_time, ess_std_time, remove_ped, robot_path_length,
                       a, b, robot_agent_path_diff, time_list, remove_ped_path_length, local_density)
        ###################SAVE DATA
        if saving:
            if frame == 0:
                a = safety_remove_ped[frame]
                b = safety_robot[frame]
            else:
                a = safety_remove_ped[:frame]
                b = safety_robot[:frame]
            # removed non-existing arguments: top_Z_indices, num_optima, optimal_ll, optima_dex, norm_likelihood, optima,
            save_data(date_time, ess, frame, \
                      ess_time, ess_ave_time, ess_std_time, 
                      robot_path_length, a, 
                      b, robot_agent_path_diff, remove_ped, time_list, home_dir, \
                      remove_ped_path_length, data_set, \
                      goal_dex, full_traj, remove_ped_start, local_density, \
                      save_dir)
            ####################SAVE PLOTS
            save_figs(frame, remove_ped, home_dir, data_set, goal_dex, full_traj, remove_ped_start, \
                      save_dir, save_pdf)
            os.chdir(str(home_dir))

