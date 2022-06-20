import numpy as np


def eth_Tdex(Tdex_max, frame, num_peds, max_vel_robot,
                robot_start_x, robot_start_y, robot_goal_x, robot_goal_y,
                vel_x, vel_y, cmd_x, cmd_y, data_set, support_boost,
                goal_dex, x_follow, y_follow, normal_vel, follow_traj):
    if data_set == 'eth_test':
        if frame == 0:
            dist = np.power(np.power(robot_start_x-robot_goal_x, 2) +
                            np.power(robot_start_y-robot_goal_y, 2), 0.5)
        elif frame > 0:
            if follow_traj:
                robot_goal_x = robot_goal_x + vel_x
                robot_goal_y = robot_goal_y + vel_y
            dist = np.power(np.power(cmd_x-robot_goal_x, 2) +
                            np.power(cmd_y-robot_goal_y, 2), 0.5)
        f_Tdex = int(dist/(max_vel_robot))

        if f_Tdex > Tdex_max:
            Tdex = [Tdex_max for _ in range(num_peds+1)]
        else:
            Tdex = [f_Tdex for _ in range(num_peds+1)]

    elif data_set == 'eth_train':
        if frame == 0:
            dist = np.power(np.power(robot_start_x-robot_goal_x, 2) +
                            np.power(robot_start_y-robot_goal_y, 2), 0.5)
        elif frame > 0:
            if follow_traj:
                # robot_goal_x = robot_goal_x + vel_x
                # robot_goal_y = robot_goal_y + vel_y
                if (goal_dex+frame) < len(x_follow):
                    robot_goal_x = x_follow[goal_dex+frame]
                    robot_goal_y = y_follow[goal_dex+frame]
                else:
                    robot_goal_x = x_follow[-1]
                    robot_goal_y = y_follow[-1]
            dist = np.power(np.power(cmd_x-robot_goal_x, 2) +
                            np.power(cmd_y-robot_goal_y, 2), 0.5)
        if int(support_boost*int(dist/(max_vel_robot))) > 65:
            Tdex = [65 for _ in range(num_peds+1)]
        else:
            Tdex = [int(support_boost*int(dist/(max_vel_robot)))
                    for _ in range(num_peds+1)]

    return Tdex, robot_goal_x, robot_goal_y
