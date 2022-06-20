import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import pylab
import math
import numpy as np
import os
from copy import deepcopy


def plotter_single(fig, ax, x_follow, y_follow, x_nonzero, y_nonzero,
                   frame, num_peds_follow, 
                   robot_mu_x, robot_mu_y,
                   robot_history_x, robot_history_y, cmd_x, cmd_y, ess_time, ess_ave_time,
                   ess_std_time, ess, time, data_set,
                   p2w_x, p2w_y, p2w, radius, show_radius, scaling,
                   opt_joint_traj_x, opt_joint_traj_y):
    cm = pylab.get_cmap('plasma')

    ax.clear()
    # PLOT DENSITY CIRCLE
    if show_radius:
        density_circle = plt.Circle((cmd_x * p2w_x, cmd_y * p2w_y), radius, \
                                    color='b', fill=False)
        ax.add_artist(density_circle)
    # PLOT AGENT TRAJECTORIES[FRAME:}, REMOVE ZEROS
    lookahead = 10
    for ped in range(num_peds_follow):
        if data_set == 'eth_train':
            x_temp = x_follow[ped][frame:frame + lookahead]
            y_temp = y_follow[ped][frame:frame + lookahead]
            x_temp = x_temp[np.nonzero(x_temp)]
            y_temp = y_temp[np.nonzero(y_temp)]
        if data_set == 'eth_test':
            x_temp = x_follow[ped][frame:]
            y_temp = y_follow[ped][frame:]
            x_temp = x_temp[np.nonzero(x_temp)]
            y_temp = y_temp[np.nonzero(y_temp)]
        ax.plot(x_temp * p2w_x, y_temp * p2w_y, ".-k", markersize=2)

    # PLOT ROBOT_MU
    ax.plot(robot_mu_x * p2w_x, robot_mu_y * p2w_y, "+g", markersize=3)

    # PLOT REMOVE PED
    ax.plot(x_nonzero * p2w_x, y_nonzero * p2w_y, "--^g", markersize=1)

    # PLOT JOINT OPTIMA WITH LARGEST LL
    n = 0
    intent = 0
    T = np.size(robot_mu_x)

    # PLOT CURRENT POSE OF ALL AGENTS EXCEPT REMOVE_PED
    for ped in range(num_peds_follow):
        ax.plot(x_follow[ped][frame] * p2w_x, y_follow[ped][frame] * p2w_y, \
                "ob", markersize=3)

    # PLOT CURRENT POSE OF REMOVE_PED
    if frame < np.size(x_nonzero):
        ax.plot(x_nonzero[frame] * p2w_x, y_nonzero[frame] * p2w_y, "og", markersize=4)

    # PLOT CURRENT POSE OF ROBOT
    ax.plot(cmd_x * p2w_x, cmd_y * p2w_y, 'py', markersize=4)

    # PLOT HISTORY OF ROBOT
    ax.plot(robot_history_x * p2w_x, robot_history_y * p2w_y, 'x--r', markersize=2)

    # (ADDED BY MUCHEN) PLOT OPTIMAL JOINT TRAJ
    for i in range(opt_joint_traj_x.shape[0]):
      ax.plot(opt_joint_traj_x[i] * p2w_x, opt_joint_traj_y[i] * p2w_y, 'xc', markersize=1)
    ax.plot(opt_joint_traj_x[-1] * p2w_x, opt_joint_traj_y[-1] * p2w_y, 'xr', markersize=1)

    plt.xlabel('X Position', fontsize=10)
    plt.ylabel('Y position', fontsize=10)

    ess_time = math.trunc(ess_time * 1e3) / 1e3
    ess_ave_time = math.trunc(ess_ave_time * 1e3) / 1e3
    ess_std_time = math.trunc(ess_std_time * 1e3) / 1e3

    if frame == 0:
        ave_time = time[frame]
        std_time = time[frame]
        max_time = math.trunc(time[frame] * 1e3) / 1e3
    else:
        ave_time = np.mean(time[:frame])
        std_time = np.std(time[:frame])
        max_time = math.trunc(np.max(time[:frame]) * 1e3) / 1e3

    time_now = math.trunc(time[frame] * 1e3) / 1e3
    ave_time = math.trunc(ave_time * 1e3) / 1e3
    std_time = math.trunc(std_time * 1e3) / 1e3

    plt.title('Frame {0}, t_now={2}, t_max = {5}, t_ave={3}+/-{4}, ESS={1}'. \
              format(frame, ess, time_now, ave_time, std_time, max_time), fontsize=7)

    if data_set == 'eth_test':
        ax.set_xlim(-scaling * 9, scaling * 15)
        ax.set_ylim(scaling * 2, scaling * 9)
    if data_set == 'eth_train':
        if cmd_x * p2w_x > scaling * 3. and cmd_x * p2w_x < scaling * 10.:
            if cmd_y * p2w_y > scaling * 7. and cmd_y * p2w_y < scaling * 19.:
                ax.set_xlim(scaling * 1, scaling * 11)
                ax.set_ylim(scaling * 5, scaling * 15)
        if cmd_x * p2w_x >= scaling * 0. and cmd_x * p2w_x < scaling * 8.:
            if cmd_y * p2w_y >= scaling * 0. and cmd_y * p2w_y < scaling * 7:
                ax.set_xlim(-1., scaling * 9)
                ax.set_ylim(-1., scaling * 9)
        if cmd_x * p2w_x > scaling * 8. and cmd_x * p2w_x < scaling * 19.:
            if cmd_y * p2w_y > scaling * 0. and cmd_y * p2w_y < scaling * 10:
                ax.set_xlim(scaling * 3, scaling * 16)
                ax.set_ylim(scaling * 0, scaling * 13)

    plt.pause(0.001)
