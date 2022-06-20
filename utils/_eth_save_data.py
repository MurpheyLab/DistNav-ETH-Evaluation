import os
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
import math

from ._dir_maker import dir_maker


def save_data(date_time, ess, frame,
              ess_time, ess_ave_time, ess_std_time,
              robot_path_length, safety_remove_ped,
              safety_robot, robot_agent_path_diff, remove_ped,
              time, home_dir, remove_ped_path_length,
              data_set,
              goal_dex, full_traj, remove_ped_start, local_density,
              save_dir):
    ess_time = math.trunc(ess_time * 1e3) / 1e3
    ess_ave_time = math.trunc(ess_ave_time * 1e3) / 1e3
    ess_std_time = math.trunc(ess_std_time * 1e3) / 1e3

    if frame == 0:
        ave_time = time[frame]
        std_time = time[frame]
        max_time = math.trunc(time[frame] * 1e4) / 1e4

        ave_density = local_density[frame]
        std_density = local_density[frame]
        max_density = math.trunc(local_density[frame] * 1e4) / 1e4
    else:
        ave_time = np.mean(time[:frame])
        std_time = np.std(time[:frame])
        max_time = math.trunc(np.max(time[:frame]) * 1e4) / 1e4

        ave_density = np.mean(local_density[:frame])
        std_density = np.std(local_density[:frame])
        max_density = math.trunc(np.max(local_density[:frame]) * 1e4) / 1e4

    time_now = math.trunc(time[frame] * 1e4) / 1e4
    ave_time = math.trunc(ave_time * 1e4) / 1e4
    std_time = math.trunc(std_time * 1e4) / 1e4

    density_now = math.trunc(local_density[frame] * 1e4) / 1e4
    ave_density = math.trunc(ave_density * 1e4) / 1e4
    std_density = math.trunc(std_density * 1e4) / 1e4

    plots_or_metrics = 'metrics'
    dir_maker(remove_ped, home_dir, data_set, goal_dex, full_traj, remove_ped_start, \
              plots_or_metrics, save_dir)

    filename = 'agent_' + str(remove_ped) + \
               '_start_' + str(remove_ped_start) + '_steps_' + str(goal_dex) + '.txt'
    print('filename: ', filename)

    """
    Format:
    Frame ESS OptimizationTime TotalTime MinSafetyAgent MinSafetyRobot PathLengthAgent PathLengthRobot Density
    """
    with open(filename, 'a') as text_file:
        print("{} {} {} {} {} {} {} {} {}".format(frame, ess, ess_time, time_now, np.min(safety_remove_ped), np.min(safety_robot), remove_ped_path_length, robot_path_length, density_now), file=text_file)

#         print(f"REMOVE PED: {remove_ped}", file=text_file)
#         print(f"FRAME NUMBER: {frame}", file=text_file)
#         print(f"ESS: {ess}", file=text_file)

#         print(f"OPTIMIZATION TIME NOW: {ess_time}", file=text_file)
#         print(f"OPTIMIZATION TIME MEAN: {ess_ave_time}+/-{ess_std_time}", \
#               file=text_file)
#         print(f"TIME NOW: {time_now}", file=text_file)
#         print(f"TIME MEAN: {ave_time}+/-{std_time}", file=text_file)
#         print(f"TIME MAX: {max_time}", file=text_file)
#         print(f"SAFETY AGENT MIN: \
# {math.trunc(np.min(safety_remove_ped) * 1e4) / 1e4}", file=text_file)
#         print(f"SAFETY ROBOT MIN: {math.trunc(np.min(safety_robot) * 1e4) / 1e4}", \
#               file=text_file)
#         print(f"SAFETY AGENT MEAN: \
# {math.trunc(np.mean(safety_remove_ped) * 1e4) / 1e4}+/-\
# {math.trunc(np.std(safety_remove_ped) * 1e4) / 1e4}", file=text_file)
#         print(f"SAFETY ROBOT MEAN: \
# {math.trunc(np.mean(safety_robot) * 1e4) / 1e4}+/-\
# {math.trunc(np.std(safety_robot) * 1e4) / 1e4}", file=text_file)
#         if frame > 0:
#             print(f"ROBOT-AGENT PATH DIFF MEAN \
# {math.trunc(1e4 * np.mean(robot_agent_path_diff[:frame])) / 1e4}+/-\
# {math.trunc(1e4 * np.std(robot_agent_path_diff[:frame])) / 1e4}", file=text_file)
#             print(f"ROBOT-AGENT PATH DIFF MAX \
# {math.trunc(np.max(robot_agent_path_diff) * 1e4) / 1e4}", file=text_file)
#             print(f"ROBOT-AGENT PATH DIFF NOW: \
# {math.trunc(robot_agent_path_diff[frame] * 1e4) / 1e4}", file=text_file)
#         print(f"AGENT PATH LENGTH: \
# {math.trunc(remove_ped_path_length * 1e4) / 1e4}", file=text_file)
#         print(f"ROBOT PATH LENGTH: {math.trunc(robot_path_length * 1e4) / 1e4}", \
#               file=text_file)
#         print(f"DENSITY NOW: {density_now}", file=text_file)
#         print(f"DENSITY MEAN: {ave_density}+/-{std_density}", file=text_file)
#         print(f"DENSITY MAX: {max_density}", file=text_file)
#         print(f" ", file=text_file)

#         print(f"REMOVE PED: {remove_ped}", file=text_file)
#         print(f"FRAME NUMBER: {frame}", file=text_file)
#         print(f"ESS: {ess}", file=text_file)

#         print(f"OPTIMIZATION TIME NOW: {ess_time}", file=text_file)
#         print(f"OPTIMIZATION TIME MEAN: {ess_ave_time}+/-{ess_std_time}", \
#               file=text_file)
#         print(f"TIME NOW: {time_now}", file=text_file)
#         print(f"TIME MEAN: {ave_time}+/-{std_time}", file=text_file)
#         print(f"TIME MAX: {max_time}", file=text_file)
#         print(f"SAFETY AGENT MIN: \
# {math.trunc(np.min(safety_remove_ped) * 1e4) / 1e4}", file=text_file)
#         print(f"SAFETY ROBOT MIN: {math.trunc(np.min(safety_robot) * 1e4) / 1e4}", \
#               file=text_file)
#         print(f"SAFETY AGENT MEAN: \
# {math.trunc(np.mean(safety_remove_ped) * 1e4) / 1e4}+/-\
# {math.trunc(np.std(safety_remove_ped) * 1e4) / 1e4}", file=text_file)
#         print(f"SAFETY ROBOT MEAN: \
# {math.trunc(np.mean(safety_robot) * 1e4) / 1e4}+/-\
# {math.trunc(np.std(safety_robot) * 1e4) / 1e4}", file=text_file)
#         if frame > 0:
#             print(f"ROBOT-AGENT PATH DIFF MEAN \
# {math.trunc(1e4 * np.mean(robot_agent_path_diff[:frame])) / 1e4}+/-\
# {math.trunc(1e4 * np.std(robot_agent_path_diff[:frame])) / 1e4}", file=text_file)
#             print(f"ROBOT-AGENT PATH DIFF MAX \
# {math.trunc(np.max(robot_agent_path_diff) * 1e4) / 1e4}", file=text_file)
#             print(f"ROBOT-AGENT PATH DIFF NOW: \
# {math.trunc(robot_agent_path_diff[frame] * 1e4) / 1e4}", file=text_file)
#         print(f"AGENT PATH LENGTH: \
# {math.trunc(remove_ped_path_length * 1e4) / 1e4}", file=text_file)
#         print(f"ROBOT PATH LENGTH: {math.trunc(robot_path_length * 1e4) / 1e4}", \
#               file=text_file)
#         print(f"DENSITY NOW: {density_now}", file=text_file)
#         print(f"DENSITY MEAN: {ave_density}+/-{std_density}", file=text_file)
#         print(f"DENSITY MAX: {max_density}", file=text_file)
#         print(f" ", file=text_file)
