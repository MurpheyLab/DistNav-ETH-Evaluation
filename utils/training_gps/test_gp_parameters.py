import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import Ellipse
import pylab
#import imageio
#import cv2
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
import george
from george import kernels
from george.modeling import Model
from george import kernels
import time

pix2world_x = 12/510
pix2world_y = 5/322

num_peds = 14

####################BUILD GP
#os.chdir("/home/ptrautman/Dropbox/code/py_scripts/utils/gp_hyperparameters/")
os.chdir(
"/Users/peter.trautman/Dropbox/code/python/IGP_python/utils/gp_hyperparameters")
hyper_x = np.load('gp_x_hyperparameters_12.npy')
hyper_y = np.load('gp_y_hyperparameters_12.npy')

#os.chdir("/home/ptrautman/Dropbox/code/py_scripts/utils/ped_truth/")
os.chdir("/Users/peter.trautman/Dropbox/code/python/IGP_python/utils/ped_truth/")

x = {}
for i in range(num_peds):
    datax = sio.loadmat('x'+str(i+1)+'.mat',squeeze_me=True)
    x[i] = datax['x'+str(i+1)]
    x[i] = x[i].astype(float)
    x[i] = x[i][0:101]

y = {}
for i in range(num_peds):
    datay = sio.loadmat('y'+str(i+1)+'.mat',squeeze_me=True)
    y[i] = datay['y'+str(i+1)]
    y[i] = y[i].astype(float)
    y[i] = y[i][0:101]

grid = np.linspace(0, 100, 101)

k0 = george.kernels.ConstantKernel(np.exp(-2*2.8770))
k1 = george.kernels.LinearKernel(np.exp(2*(2.8770)),order=1) 
k2 = 2.0 * george.kernels.Matern52Kernel(5.0)
k3 = 1.0 * george.kernels.ExpSquaredKernel(0.5)

# k0+k1+k2+k3 IS THE BEST KERNEL
kernel_x = k1+k2
kernel_y = k1+k2

gp_x = george.GP(kernel_x,fit_white_noise=True)
gp_y = george.GP(kernel_y,fit_white_noise=True)

gp_x.set_parameter_vector(hyper_x)
gp_y.set_parameter_vector(hyper_y)

x_obs_times = np.linspace(0, 20, 21)
y_obs_times = np.linspace(0, 20, 21)

x_obs_times[20] = 50
y_obs_times[20] = 50

err_magnitude = 3.
err = err_magnitude * np.ones_like(x_obs_times)

gp_x.compute(x_obs_times, err)
gp_y.compute(y_obs_times, err)
####################GP BUILT

plt.clf()

# t1 = time.time()
for ped in range(num_peds):
    noise = 0.0
    noise_x = noise * np.random.rand(101)
    noise_y = noise * np.random.rand(101)
    
    x[ped] += noise_x
    y[ped] += noise_y
    
    # # REVERSE
    # x[i] = x[i][::-1]
    # y[i] = y[i][::-1]
    
    x_obs = x[ped][0:21]
    y_obs = y[ped][0:21]
    
    ave_vel_x = 0.
    ave_vel_y = 0.
    Tdex = 49.-21.
    for t in range(21):
        ave_vel_x = ave_vel_x + (x[ped][t+1] - x[ped][t])
        ave_vel_y = ave_vel_y + (y[ped][t+1] - y[ped][t])
    
    ave_vel_x = ave_vel_x/21.
    ave_vel_y = ave_vel_y/21.
    
    linear_goal_x = ave_vel_x*Tdex + x[ped][21]
    linear_goal_y = ave_vel_y*Tdex + y[ped][21]
    
#    x_obs[-1] = x[ped][49]
#    y_obs[-1] = y[ped][49]
    x_obs[-1] = linear_goal_x
    y_obs[-1] = linear_goal_y
    
    num_samples = 1000
    # samples_prior_x = gp_x.sample(grid[0:50], size=num_samples)
    samples_conditional_x = gp_x.sample_conditional(
        x_obs, grid[0:50],size=num_samples)
    samples_conditional_y = gp_y.sample_conditional(
        y_obs, grid[0:50],size=num_samples)
    
    # print("Done sampling")
    #########SAMPLING

    mu_x_test, sigma_x_test = gp_x.predict(x_obs, grid[0:50], return_var=True)
    mu_y_test, sigma_y_test = gp_y.predict(y_obs, grid[0:50], return_var=True)
    
# t2 = time.time()
# print(t2-t1)
    
    for j in range(1,num_samples):
        plt.plot(samples_conditional_x[j,:],
            samples_conditional_y[j,:])
    observed_x, =plt.plot(x[ped][0:20],y[ped][0:20]
                    , "o--g", label='Observed X',markersize=14)
    unobserved_x, =plt.plot(x[ped][21:50],y[ped][21:50],
                    "--^b", label='Not Observed',markersize=8)
    gp_mean, =plt.plot(mu_x_test[:],mu_y_test[:],
                    "^r",label='GP mean',markersize=6)
    
    plt.legend(
        handler_map={gp_mean: HandlerLine2D(numpoints=2)},prop={'size': 15})
    plt.xlabel('X Position',fontsize=20)
    plt.ylabel('Y position',fontsize=20)
    plt.title('Kernel=k0+k1+k2+k3, Noise={0} Error={1} Ped={2}'.format(
            noise,err_magnitude,ped),
        fontsize=12)

    plt.show(block=True)
#    plt.show()

# # ########GET COVARIANCE MATRIX
# # cov_x = gp_x.get_matrix(x,grid)
# # print(cov_x.shape)
# # mu_x, cov_x = gp_x.predict(x, grid, return_cov=True)
# ########GET COVARIANCE MATRIX
