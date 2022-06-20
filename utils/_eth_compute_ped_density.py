import numpy as np

def compute_ped_density(frame, cmd_x, cmd_y, x_follow, y_follow, r, num_peds_follow, \
            p2w_x, p2w_y):

  num_inside = 0
  for ped in range(num_peds_follow):
    dist = np.power(np.power(p2w_x*(cmd_x-x_follow[ped][frame]), 2) + \
                    np.power(p2w_y*(cmd_y-y_follow[ped][frame]), 2), 1/2)
    if dist < r:
      num_inside = num_inside + 1

  local_density = num_inside/(np.pi*np.power(r, 2))

  return local_density