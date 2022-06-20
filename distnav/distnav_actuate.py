import numpy as np

def actuate(f, T, x_obs, y_obs, max_vel_robot, \
				    robot_history_x, robot_history_y, frame, \
				    x_follow, y_follow, num_peds_follow, p2w_x, p2w_y, p2w, \
				    actuate_distance, actuate_to_step, actuate_to_index):
	if actuate_distance:
		if frame == 0:
			vel_x = f[0] - robot_history_x
			vel_y = f[T] - robot_history_y
		else:
			vel_x = f[0] - robot_history_x[-1]
			vel_y = f[T] - robot_history_y[-1]

		vel = [vel_x, vel_y]
		h = np.power(np.linalg.norm(vel), -1.)
		vel_x = max_vel_robot*h*vel_x
		vel_y = max_vel_robot*h*vel_y

		if frame == 0:
			cmd_x = vel_x + robot_history_x
			cmd_y = vel_y + robot_history_y
		else:
			cmd_x = vel_x + robot_history_x[-1]
			cmd_y = vel_y + robot_history_y[-1]
#############ACTUATE TO STEP
	if actuate_to_step:
		if frame == 0:
			vel_x = f[0] - robot_history_x
			vel_y = f[T] - robot_history_y

			vel = [vel_x, vel_y]
			h = np.linalg.norm(vel)

			i = 0
			while h < max_vel_robot:
				i = i + 1
				vel_x = f[i] - f[i-1] + vel_x
				vel_y = f[T+i] - f[T+i-1] + vel_y
				vel = [vel_x, vel_y]
				h = np.linalg.norm(vel)
		else:
			vel_x = f[0] - robot_history_x[-1]
			vel_y = f[T] - robot_history_y[-1]

			vel = [vel_x, vel_y]
			h = np.linalg.norm(vel)

			i = 0
			while h < max_vel_robot:
				i = i + 1
				vel_x = f[i] - f[i-1] + vel_x
				vel_y = f[T+i] - f[T+i-1] + vel_y
				vel = [vel_x, vel_y]
				h = np.linalg.norm(vel)
		if frame == 0:
			cmd_x = vel_x + robot_history_x
			cmd_y = vel_y + robot_history_y
		else:
			cmd_x = vel_x + robot_history_x[-1]
			cmd_y = vel_y + robot_history_y[-1]
#############ACTUATE TO INDEX
	if actuate_to_index:
		if frame == 0:
			vel_x = f[0] - robot_history_x
			vel_y = f[T] - robot_history_y

			vel = [vel_x, vel_y]
			h = np.linalg.norm(vel)

			i = 0
			while h < max_vel_robot:
				i = i + 1
				vel_x = f[i] - f[i-1] + vel_x
				vel_y = f[T+i] - f[T+i-1] + vel_y
				vel = [vel_x, vel_y]
				h = np.linalg.norm(vel)
				# if i < 5:
				# 	break
		else:
			vel_x = f[0] - robot_history_x[-1]
			vel_y = f[T] - robot_history_y[-1]

			vel = [vel_x, vel_y]
			h = np.linalg.norm(vel)

			i = 0
			while h < max_vel_robot:
				i = i + 1
				vel_x = f[i] - f[i-1] + vel_x
				vel_y = f[T+i] - f[T+i-1] + vel_y
				vel = [vel_x, vel_y]
				h = np.linalg.norm(vel)
				# if i < 5:
				# 	break
		cmd_x = f[i]
		cmd_y = f[T+i]

	dist_robot = [0. for _ in range(num_peds_follow)]
	for ped in range(num_peds_follow):
		dist_robot[ped] = np.power(\
	                         np.power(p2w_x*(cmd_x-x_follow[ped][frame]), 2) + \
	                         np.power(p2w_y*(cmd_y-y_follow[ped][frame]), 2), 1/2)
	safety_robot = np.min(dist_robot)

	robot_history_x = np.append(robot_history_x, cmd_x)
	robot_history_y = np.append(robot_history_y, cmd_y)

	return robot_history_x, robot_history_y, cmd_x, cmd_y, vel_x, vel_y
















