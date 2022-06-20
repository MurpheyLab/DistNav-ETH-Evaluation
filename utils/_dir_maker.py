import os
import math
import numpy as np


def dir_maker(remove_ped, home_dir, data_set, \
              goal_dex, full_traj, remove_ped_start, plots_or_metrics, \
              save_dir):
    if full_traj:
        traj_dir = 'full_traj'
    else:
        traj_dir = 'partial_traj'
    check_dir = str(home_dir) + '/results/' + str(save_dir) + '/' + \
                str(plots_or_metrics) + '/' + str(data_set) + '/' + str(traj_dir) + '/'

    save_folder = str(home_dir) + '/results/' + str(save_dir) + '/' + \
                  str(plots_or_metrics) + '/' + str(data_set) + '/' + str(traj_dir) + '/' + \
                  'agent_' + str(remove_ped) + \
                  '_start_' + str(remove_ped_start) + \
                  '_steps_' + str(goal_dex)

    if not os.path.exists(check_dir):
        # os.mkdir(check_dir)
        os.makedirs(check_dir)

    if not os.path.exists(save_folder):
        # os.mkdir(save_folder)
        os.makedirs(save_folder)

    os.chdir(save_folder)