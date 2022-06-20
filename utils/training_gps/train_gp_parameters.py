import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import Ellipse
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
import george
from george import kernels
from george.modeling import Model
from george import kernels

#######################DATA SETUP
##OSX HERE
#datax1 = sio.loadmat(
#'/Users/peter.trautman/Dropbox/code/py_scripts/utils/ped_truth/x1.mat',
#squeeze_me=True)
#datay1 = sio.loadmat(
#'/Users/peter.trautman/Dropbox/code/py_scripts/utils/ped_truth/y1.mat',
#squeeze_me=True)

file_dir = os.path.dirname(os.path.abspath(__file__))

plt.clf()

# os.chdir("/home/ptrautman/Dropbox/code/py_scripts/utils/ped_truth")
#os.chdir("/Users/peter.trautman/Dropbox/code/python/IGP_python\
# /utils/ped_truth/")

pix2world_x = 12./510.
pix2world_y = 5./322.
pix2world = np.sqrt(pix2world_x**2 + pix2world_y**2)

datax1 = sio.loadmat(str(file_dir)+'/../../chandler_data/ped_truth/x1.mat', \
                     squeeze_me=True)
datay1 = sio.loadmat(str(file_dir)+'/../../chandler_data/ped_truth/y1.mat', \
                     squeeze_me=True)

x = datax1['x1']
y = datay1['y1']

x = x*pix2world_x
y = y*pix2world_y

noise = 0.05
#magic: k1:k3 is noise 1.3; inv diag = 6700
#magic: k1:k2 is noise 1.8, inv diag = 8000
noise_x = noise * np.random.rand(101)
noise_y = noise * np.random.rand(101)

x += noise_x
y += noise_y

err_magnitude = noise
err = err_magnitude * np.ones_like(x)

k0 = george.kernels.ConstantKernel(np.exp(-2.))
k1 = george.kernels.LinearKernel(np.exp(2.),order=1)
k2 = 2.0 * george.kernels.Matern52Kernel(5.0)
k3 = 1.0 * george.kernels.ExpSquaredKernel(0.5)

# k1+k2 IS THE BEST KERNEL
#k0:k3 DOESN'T TRAIN
kernel_x = k1+k2
kernel_y = k1+k2

gp_x = george.GP(kernel_x,fit_white_noise=True)
gp_y = george.GP(kernel_y,fit_white_noise=True)

grid = np.linspace(0, 100, 101)

x_obs_times = np.linspace(0, 100, 101)
y_obs_times = np.linspace(0, 100, 101)

gp_x.compute(x_obs_times, err)
gp_y.compute(y_obs_times, err)

print("Initial ln-likelihood: {0:.2f}".format(gp_x.log_likelihood(x)))
print("Initial ln-likelihood: {0:.2f}".format(gp_y.log_likelihood(y)))

def neg_ln_like_x(p):
    gp_x.set_parameter_vector(p)
    return -gp_x.log_likelihood(x)

def grad_neg_ln_like_x(p):
    gp_x.set_parameter_vector(p)
    return -gp_x.grad_log_likelihood(x)

def neg_ln_like_y(p):
    gp_y.set_parameter_vector(p)
    return -gp_y.log_likelihood(y)

def grad_neg_ln_like_y(p):
    gp_y.set_parameter_vector(p)
    return -gp_y.grad_log_likelihood(y)

result_x = minimize(neg_ln_like_x, gp_x.get_parameter_vector(), \
                    jac=grad_neg_ln_like_x)

result_y = minimize(neg_ln_like_y, gp_y.get_parameter_vector(), \
                    jac=grad_neg_ln_like_y)

print(result_x)
print(result_y)

gp_x.set_parameter_vector(result_x.x)
gp_y.set_parameter_vector(result_y.x)

# os.chdir(
# "/Users/peter.trautman/Dropbox/code/python/IGP_python/\
# utils/gp_hyperparams_world/k12")

# np.save('gp_x_hyperparameters_12_err_0.1', result_x.x)
# np.save('gp_y_hyperparameters_12_err_0.1', result_y.x)

print("\nFinal x ln-likelihood: {0:.2f}".format(gp_x.log_likelihood(x)))
print("\nFinal y ln-likelihood: {0:.2f}".format(gp_y.log_likelihood(y)))

dur = 5
T = 30

x_obs_times = np.append(x_obs_times[:dur], grid[T])
y_obs_times = np.append(y_obs_times[:dur], grid[T])

err_end = 1.

error = np.append(err[:dur], err_end)

gp_x.compute(x_obs_times, error)
gp_y.compute(y_obs_times, error)

x_obs = x[:dur]
y_obs = y[:dur]

x_obs = np.append(x_obs, x[T])
y_obs = np.append(y_obs, y[T])

mu_x, cov_x = gp_x.predict(x_obs, grid, return_cov=True)
mu_y, cov_y = gp_x.predict(y_obs, grid, return_cov=True)

print('var x', np.diag(cov_x))
print('var y', np.diag(cov_x))

print('inv x', np.power(np.diag(cov_x[:dur+T,:dur+T]), -1))
print('inv y', np.power(np.diag(cov_y[:dur+T,:dur+T]), -1))

# ########GET COVARIANCE MATRIX
# cov_x = gp_x.get_matrix(x,grid)
# print(cov_x.shape)
# mu_x, cov_x = gp_x.predict(x, grid, return_cov=True)
########GET COVARIANCE MATRIX

# plt.fill_between(mu_x, mu_x - np.sqrt(sigma_x), mu_x + np.sqrt(sigma_x),
#                 color="k", alpha=0.2)
#plt.plot(mu_x, np.linspace(0,100,101), "^r",x,np.linspace(0,100,101), "--g")
plt.plot(mu_x, mu_y, '^r')
plt.plot(x, y, '--g')
#plt.plot(y,x, "^r", mu_y, mu_x, '--g')
plt.show(block=True)#to display from command line


##LINUX VERSION
#datax1 = sio.loadmat(
#'/home/ptrautman/Dropbox/code/py_scripts/utils/ped_truth/x1.mat',
#squeeze_me=True)
#datay1 = sio.loadmat(
#'/home/ptrautman/Dropbox/code/py_scripts/utils/ped_truth/y1.mat',
#squeeze_me=True)
#
#x = datax1['x1']
#y = datay1['y1']
#
#noise = 0.2
#noise_x = noise * np.random.rand(101)
#noise_y = noise * np.random.rand(101)
#
#x += noise_x
#y += noise_y
#
#grid = np.linspace(0, 100, 101)
#####################DATA IS SETUP
#
#os.chdir("/home/ptrautman/Dropbox/code/py_scripts/utils/gp_hyperparameters")
##os.chdir("/Users/peter.trautman/Dropbox/code/py_scripts/utils/\
# gp_hyperparameters")
#
#####################BUILD GP
#hyper_x = np.load('gp_x_hyperparameters_1234.npy')
#hyper_y = np.load('gp_y_hyperparameters_1234.npy')
#
#k0 = george.kernels.ConstantKernel(np.exp(-2*2.8770))
#k1 = george.kernels.LinearKernel(np.exp(2*(2.8770)),order=1)
#k2 = 2.0 * george.kernels.Matern52Kernel(5.0)
#k3 = 1.0 * george.kernels.ExpSquaredKernel(0.5)
#
## k0+k1+k2+k3 IS THE BEST KERNEL
#kernel_x = k0+k1+k2+k3
#kernel_y = k0+k1+k2+k3
#
#gp_x = george.GP(kernel_x,fit_white_noise=True)
#gp_y = george.GP(kernel_y,fit_white_noise=True)
#
#gp_x.set_parameter_vector(hyper_x)
#gp_y.set_parameter_vector(hyper_y)
#####################GP BUILT
#
#x_obs_times = np.linspace(0, 20, 21)
#y_obs_times = np.linspace(0, 20, 21)
#
#x_obs_times[20] = 50
#y_obs_times[20] = 50
#
#err_magnitude = 3.
#err = err_magnitude * np.ones_like(x_obs_times)
#
#gp_x.compute(x_obs_times, err)
#gp_y.compute(y_obs_times, err)
#
#x_obs = x[0:21]
#y_obs = y[0:21]
#
#x_obs[-1] = x[49]
#y_obs[-1] = y[49]
#
#########SAMPLING
#num_samples = 10
## samples_prior_x = gp_x.sample(grid[0:50], size=num_samples)
#samples_conditional_x = gp_x.sample_conditional(
#                        x_obs, grid[0:50],size=num_samples)
#samples_conditional_y = gp_y.sample_conditional(
#                        y_obs, grid[0:50],size=num_samples)
#print("Done sampling")
## ########SAMPLING
#
#mu_x_test, sigma_x_test = gp_x.predict(x_obs, grid[0:50], return_var=True)
#mu_y_test, sigma_y_test = gp_x.predict(y_obs, grid[0:50], return_var=True)
#
#plt.clf()
#for i in range(1,num_samples):
#    plt.plot(samples_conditional_x[i,:],samples_conditional_y[i,:])
#
#gp_mean, =plt.plot(mu_x_test[0:50],mu_y_test[0:50],"^r",label='GP mean'
#                   ,markersize=8)
#observed_x, =plt.plot(x[0:20],y[0:20], "o--g", label='Observed X'
#                      ,markersize=8)
#unobserved_x, =plt.plot(x[21:50],y[21:50],"--^b", label='Not Observed'
#                        ,markersize=8)
#
#plt.legend(handler_map={gp_mean: HandlerLine2D(numpoints=2)},prop={'size': 15})
#plt.xlabel('X Position',fontsize=20)
#plt.ylabel('Y position',fontsize=20)
#plt.title('Kernel=k0+k1+k2+k3, Noise={0} Error={1}'.\
# format(noise,err_magnitude),
#fontsize=12)
#
#plt.show(block=True)





