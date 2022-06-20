import george
import os
import numpy as np


def gp_init(num_peds, home_dir):
    gp_x = [0. for _ in range(num_peds+1)]
    gp_y = [0. for _ in range(num_peds+1)]
    # 2*p2w = .05 meters = 2 inches.  REasonable laser noise.

    # hyper_x_ped = np.load('gp_x_hyperparameters_123_err_1.4.npy')
    # hyper_y_ped = np.load('gp_y_hyperparameters_123_err_1.4.npy')
    # hyper_x_robot = np.load('gp_x_hyperparameters_123_err_1.4.npy')
    # hyper_y_robot = np.load('gp_y_hyperparameters_123_err_1.4.npy')
    os.chdir(str(home_dir)+'/utils/gp_hyperparams_pixels/k12/')
    hyper_x_ped = np.load('gp_x_hyperparameters_12_err_2.npy')
    hyper_y_ped = np.load('gp_y_hyperparameters_12_err_2.npy')

    hyper_x_robot = np.load('gp_x_hyperparameters_12_err_9.npy')
    hyper_y_robot = np.load('gp_y_hyperparameters_12_err_9.npy')
    # os.chdir(str(home_dir)+'/utils/gp_hyperparams_pixels/k123/')
    # hyper_x_robot = np.load('gp_x_hyperparameters_123_err_1.4.npy')
    # hyper_y_robot = np.load('gp_y_hyperparameters_123_err_1.4.npy')

    k1 = george.kernels.LinearKernel(np.exp(2*(2.8770)), order=1)
    k2 = 2.0 * george.kernels.Matern52Kernel(5.0)
    k3 = 1.0 * george.kernels.ExpSquaredKernel(0.5)

    kernel_ped_x = k1+k2
    kernel_ped_y = k1+k2

    kernel_robot_x = k1+k2
    kernel_robot_y = k1+k2
# WORLD: 2 pixels = .06 meters, 9 pixels = .25 meters
    # hyper_x_ped = np.load('gp_x_hyperparameters_12_err_0.06.npy')
    # hyper_y_ped = np.load('gp_y_hyperparameters_12_err_0.06.npy')
    # hyper_x_robot = np.load('gp_x_hyperparameters_12_err_0.25.npy')
    # hyper_y_robot = np.load('gp_y_hyperparameters_12_err_0.25.npy')

    # hyper_x_ped = np.load('gp_x_hyperparameters_12_err_0.06.npy')
    # hyper_y_ped = np.load('gp_y_hyperparameters_12_err_0.06.npy')
    # hyper_x_robot = np.load('gp_x_hyperparameters_12_err_0.25.npy')
    # hyper_y_robot = np.load('gp_y_hyperparameters_12_err_0.25.npy')

    # 3.5*p2w = .1 meters.  2 stds = .2 meters, which is the amount that we
    # don't
    # want the planner to extend beyond.  So with probability 95% the planner
    # will
    #not overplan
    # 5*p2w = .14 meters acts as a one standard deviation constraint.  2 std's
    # would
    # be .28 meters.  So we ask:
    # err_robot IS A FORM OF CONSTRAINT ON THE OPTIMIZATION!!!  WITH LARGE ERR
    # WE
    # SEE LARGE OVERSHOOTS, INDICATING THAT THE PLANNER BELIEVES THAT THE ROBOT
    # CAN
    # MOVE FASTER THAN IT ACTUALLY CAN.
    # CONVERSELY, WE CAN GET CLOSE TO IMPOSING KINEMATIC CONSTRAINTS VIA THE ERR
    # ON THE ROBOT, SO WE SHOULD USE THIS!  QUAD_ROBOT_MU IS NATURALLY A WAY TO
    # FORMULATE NEWTON INEQUALITY CONSTRAINED OPTIMIZATION.
    # 1/5 err_2 = .4 inches

    #err = [0. for _ in range(num_peds+2)]
    # err corresponds to noise that the GP was trained on.
    # ERR IS THE NOISE THAT THE SENSORS ARE EXPERIENCING
    # 10*p2w = .28 meters.  That's the std of the sensors

    for ped in range(num_peds+1):
        if ped >= num_peds:
            gp_x[ped] = george.GP(kernel_robot_x, fit_white_noise=True)
            gp_y[ped] = george.GP(kernel_robot_y, fit_white_noise=True)

            gp_x[ped].set_parameter_vector(hyper_x_robot)
            gp_y[ped].set_parameter_vector(hyper_y_robot)
        else:
            gp_x[ped] = george.GP(kernel_ped_x, fit_white_noise=True)
            gp_y[ped] = george.GP(kernel_ped_y, fit_white_noise=True)

            gp_x[ped].set_parameter_vector(hyper_x_ped)
            gp_y[ped].set_parameter_vector(hyper_y_ped)

    return gp_x, gp_y
