import matplotlib.pyplot as plt
import os
import math
import numpy as np

from ._dir_maker import dir_maker


def save_figs(frame, remove_ped, home_dir, data_set, goal_dex, full_traj,
              remove_ped_start, save_dir, save_pdf):
    plots_or_metrics = 'plots'
    dir_maker(remove_ped, home_dir, data_set, goal_dex, full_traj, remove_ped_start, \
              plots_or_metrics, save_dir)

    if save_pdf:
        file_type = '.pdf'
    else:
        file_type = '.png'
    plt.savefig('agent_' + str(remove_ped) +
                '_start_' + str(remove_ped_start) +
                '_steps_' + str(goal_dex) +
                '_frame_' + str(frame) + file_type)
