import numpy as np

def start_goal(x, y, remove_ped, dist_agent_10, p2w_x, p2w_y, p2w, \
							 data_set, goal_dex, follow_traj):
	robot_start_x = x[0]
	robot_start_y = y[0]
	robot_goal_x = x[goal_dex]
	robot_goal_y = y[goal_dex]

	dist_agent = np.power(np.power(robot_start_x-robot_goal_x, 2) + \
											  np.power(robot_start_y-robot_goal_y, 2), 1/2)

	Tdex_10 = 25 #20 for full, 55 for diag.
	Tdex_max = np.int((Tdex_10/dist_agent_10)*dist_agent)

	path_length = 0.
	if goal_dex == -1 or follow_traj:
		for t in range(np.size(x)-1):
			path_length = path_length + np.power(np.power(p2w_x*(x[t]-x[t+1]), 2) + \
																				 np.power(p2w_y*(y[t]-y[t+1]), 2), 1/2)
	else:
		for t in range(goal_dex-1):
			path_length = path_length + np.power(np.power(p2w_x*(x[t]-x[t+1]), 2) + \
																				 np.power(p2w_y*(y[t]-y[t+1]), 2), 1/2)
	return robot_start_x, robot_start_y, robot_goal_x, robot_goal_y, \
	   		 Tdex_max, path_length
