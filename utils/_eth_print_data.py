import os
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
import math


def print_data(ess, frame,
               ess_time, ess_ave_time,
               ess_std_time, remove_ped,
               robot_path_length, safety_remove_ped, safety_robot,
               robot_agent_path_diff, time, remove_ped_path_length,
               local_density,
               ):
    ess_time = math.trunc(ess_time * 1e4) / 1e4
    ess_ave_time = math.trunc(ess_ave_time * 1e4) / 1e4
    ess_std_time = math.trunc(ess_std_time * 1e4) / 1e4

    if frame == 0:
        ave_time = time[frame]
        std_time = time[frame]
        # max_time = math.trunc((1 / (2 * (num_var_samples + 1))) * time[frame] * 1e4) / 1e4

        ave_density = local_density[frame]
        std_density = local_density[frame]
        max_density = math.trunc(local_density[frame] * 1e4) / 1e4
    else:
        ave_time = np.mean(time[:frame])
        std_time = np.std(time[:frame])
        # max_time = math.trunc((1 / (2 * (num_var_samples + 1))) * np.max(time[:frame]) * 1e4) / 1e4

        ave_density = np.mean(local_density[:frame])
        std_density = np.std(local_density[:frame])
        max_density = math.trunc(np.max(local_density[:frame]) * 1e4) / 1e4

    time_now = math.trunc(time[frame] * 1e4) / 1e4
    ave_time = math.trunc(ave_time * 1e4) / 1e4
    std_time = math.trunc(std_time * 1e4) / 1e4

    density_now = math.trunc(local_density[frame] * 1e4) / 1e4
    ave_density = math.trunc(ave_density * 1e4) / 1e4
    std_density = math.trunc(std_density * 1e4) / 1e4

    # print(f"DIAG IS : {diag_or_full}")
    print(f"REMOVE PED: {remove_ped}")
    print(f"FRAME NUMBER: {frame}")
    print(f"ESS: {ess}")

    print(f"OPTIMIZATION TIME NOW: {ess_time}")
    print(f"OPTIMIZATION TIME MEAN: {ess_ave_time}+/-{ess_std_time}")
    print(f"TIME NOW: {time_now}")
    print(f"TIME MEAN: {ave_time}+/-{std_time}")
    # print(f"TIME MAX: {max_time}")
    print(f"SAFETY AGENT MIN: \
{math.trunc(np.min(safety_remove_ped) * 1e4) / 1e4}")
    print(f"SAFETY ROBOT MIN: {math.trunc(np.min(safety_robot) * 1e4) / 1e4}")
    print(f"SAFETY AGENT MEAN: \
{math.trunc(np.mean(safety_remove_ped) * 1e4) / 1e4}+/-\
{math.trunc(np.std(safety_remove_ped) * 1e4) / 1e4}")
    print(f"SAFETY ROBOT MEAN: \
{math.trunc(np.mean(safety_robot) * 1e4) / 1e4}+/-\
{math.trunc(np.std(safety_robot) * 1e4) / 1e4}")
    if frame > 0:
        print(f"ROBOT-AGENT PATH DIFF MEAN \
{math.trunc(1e4 * np.mean(robot_agent_path_diff[:frame])) / 1e4}+/-\
{math.trunc(1e4 * np.std(robot_agent_path_diff[:frame])) / 1e4}")
        print(f"ROBOT-AGENT PATH DIFF MAX \
{math.trunc(np.max(robot_agent_path_diff) * 1e4) / 1e4}")
        print(f"ROBOT-AGENT PATH DIFF NOW: \
{math.trunc(robot_agent_path_diff[frame] * 1e4) / 1e4}")
    print(f"AGENT PATH LENGTH: \
{math.trunc(remove_ped_path_length * 1e4) / 1e4}")
    print(f"ROBOT PATH LENGTH: {math.trunc(robot_path_length * 1e4) / 1e4}")
    print(f"DENSITY NOW: {density_now}")
    print(f"DENSITY MEAN: {ave_density}+/-{std_density}")
    print(f"DENSITY MAX: {max_density}")
    print('')
