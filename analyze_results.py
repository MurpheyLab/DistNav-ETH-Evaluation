import time
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import math
import datetime
from copy import deepcopy

from utils._eth_runs import run_data

def analyze_file(file_name):
    file_data = np.loadtxt(file_name)

    steps = file_data.shape[0]
    opt_time = np.mean(file_data[1:,3])
    safety_human = np.min(file_data[:,4])
    safety_robot = np.min(file_data[:,5])
    human_path_length = np.max(file_data[:,6])
    robot_path_length = np.max(file_data[:,7])

    return steps-1, opt_time, safety_human, safety_robot, human_path_length, robot_path_length


time_prefix = "20220614-132457"  # you will need to replace the time_prefix with the current test, you can find it under the "results" directory

discomfort_list = []
freezing_list = []
collision_list = []

opt_time_list = []
safety_human_list = []
safety_robot_list = []
human_path_length_list = []
robot_path_length_list = []

steps_total = 0
for k in run_data():
    words = k.split("_")
    
    if "full" in words:
        continue
    traj_prefix = "partial_traj"

    remove_ped, remove_ped_start, goal_dex, max_vel_robot, full_traj \
        = run_data()[k]
    
    trial_name = "agent_{}_start_{}_steps_{}".format(remove_ped, remove_ped_start, goal_dex) 
    file_name = "results/distnav-" + time_prefix + "/metrics/eth_train/" + traj_prefix + "/" + trial_name + "/" + trial_name + ".txt"

    try:
        steps, opt_time, safety_human, safety_robot, human_path_length, robot_path_length = analyze_file(file_name)
        
        steps_total += steps
        if(safety_robot < 0.21):
            collision_list.append(1.0)
        else:
            collision_list.append(0.0)
        if(safety_robot < 0.3):
            discomfort_list.append(1.0)
        else:
            discomfort_list.append(0.0)
        if(robot_path_length/human_path_length > 1.25):
            freezing_list.append(1.0)
        else:
            freezing_list.append(0.0)

        opt_time_list.append(opt_time)
        safety_human_list.append(safety_human)
        safety_robot_list.append(safety_robot)
        robot_path_length_list.append(robot_path_length)
        human_path_length_list.append(human_path_length)
    except:
        print(k)
        print("file does not exist.")

print("ave opt time: ", np.mean(opt_time_list))
print("ave robot safety: ", np.mean(safety_robot_list))
print("discomfort: ", np.mean(discomfort_list))
print("collision: ", np.mean(collision_list))
print("freezing: ", np.mean(freezing_list))
print("max length ratio: ", np.max(np.array(robot_path_length_list)/np.array(human_path_length_list)))
print("ave robot path length: ", np.mean(robot_path_length_list))
print("ave human path length: ", np.mean(human_path_length_list))
