import os
import scipy.io as sio
import numpy as np

import pickle

# try igp university test


def import_eth_data(data_set, remove_ped, file_dir, remove_ped_start):
    x = {}
    y = {}

    x_follow = {}
    y_follow = {}

    scaling = 2.5
    p2w_x = scaling*1./100.
    p2w_y = scaling*1./100.
    p2w = np.sqrt(p2w_x**2 + p2w_y**2)

    if data_set == 'eth_test':
        with open(str(file_dir) + '/eth_data/test_data_dict.pkl', 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')
            data_id = 11
            num_peds_real = 30
            num_peds_follow = num_peds_real - 1
            num_frames = 100
    if data_set == 'eth_train':
        with open(str(file_dir) + '/eth_data/igp_eth_train.pkl', 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')
            data_id = 11  # Total - train_data_dict:189 & test_data_dict:48
            num_peds_real = 150
            num_peds_follow = num_peds_real - 1
            num_frames = 100

    for ped in range(num_peds_real):
        x[ped] = data_dict[str(ped)][data_id][:, 0]
        # converting into cm
        x[ped] = x[ped]*100
        x[ped] = x[ped][remove_ped_start:num_frames]

        y[ped] = data_dict[str(ped)][data_id][:, 1]
        # converting into cm
        y[ped] = y[ped]*100
        y[ped] = y[ped][remove_ped_start:num_frames]

    n = 0
    for ped in range(num_peds_real):
        if(ped < remove_ped):
            x_follow[n] = data_dict[str(n)][data_id][:, 0]
            # converting into cm
            x_follow[n] = x_follow[n]*100
            x_follow[n] = x_follow[n][remove_ped_start:num_frames]

            y_follow[n] = data_dict[str(n)][data_id][:, 1]
            # converting into cm
            y_follow[n] = y_follow[n]*100
            y_follow[n] = y_follow[n][remove_ped_start:num_frames]
            n = n + 1
        elif(ped > remove_ped):
            x_follow[n] = data_dict[str(ped)][data_id][:, 0]
            # converting into cm
            x_follow[n] = x_follow[n]*100
            x_follow[n] = x_follow[n][remove_ped_start:num_frames]

            y_follow[n] = data_dict[str(ped)][data_id][:, 1]
            # converting into cm
            y_follow[n] = y_follow[n]*100
            y_follow[n] = y_follow[n][remove_ped_start:num_frames]
            n = n + 1

    return x, y, x_follow, y_follow, num_peds_real, num_peds_follow, num_frames, \
        p2w_x, p2w_y, p2w, scaling
