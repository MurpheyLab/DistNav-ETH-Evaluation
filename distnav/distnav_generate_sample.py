import numpy as np
import numpy.random as npr
import time
from scipy.stats import multivariate_normal as mvn


def generate_sample(robot_mu_x, robot_cov_x, robot_mu_y, robot_cov_y,
                    ped_mu_x, ped_cov_x, ped_mu_y, ped_cov_y,
                    traj_len, num_peds, num_samples, nonzero_ped_idx, scale):
    num_peds_nonzero = len(nonzero_ped_idx)
    npr.seed(int(time.time()))

    agents_sample_x = np.zeros(((num_peds + 1) * num_samples, traj_len), dtype=np.float32)
    agents_sample_y = np.zeros(((num_peds + 1) * num_samples, traj_len), dtype=np.float32)
    agents_pdf_x = np.zeros((num_peds + 1, num_samples))
    agents_pdf_y = np.zeros((num_peds + 1, num_samples))

    agents_sample_x_nonzero = np.zeros(((num_peds_nonzero + 1) * num_samples, traj_len), dtype=np.float32)
    agents_sample_y_nonzero = np.zeros(((num_peds_nonzero + 1) * num_samples, traj_len), dtype=np.float32)
    agents_pdf_x_nonzero = np.zeros((num_peds_nonzero + 1, num_samples))
    agents_pdf_y_nonzero = np.zeros((num_peds_nonzero + 1, num_samples))

    for i in range(num_peds):
        rv_x = mvn(mean=ped_mu_x[i], cov=ped_cov_x[i] * scale, allow_singular=True)
        rv_y = mvn(mean=ped_mu_y[i], cov=ped_cov_y[i] * scale, allow_singular=True)
        rv_samples_x = rv_x.rvs(size=num_samples)
        rv_samples_y = rv_y.rvs(size=num_samples)
        rv_samples_x[0] = ped_mu_x[i].copy()
        rv_samples_y[0] = ped_mu_y[i].copy()  # replace one sample with GP mean
        agents_sample_x[num_samples * i:num_samples * (i + 1)] = rv_samples_x.copy()
        agents_sample_y[num_samples * i:num_samples * (i + 1)] = rv_samples_y.copy()
        agents_pdf_x[i] = rv_x.pdf(rv_samples_x)
        agents_pdf_y[i] = rv_y.pdf(rv_samples_y)

        nonzero_idx = np.where(nonzero_ped_idx == i)[0]
        if len(nonzero_idx) > 0:
            idx = nonzero_idx[0]
            agents_sample_x_nonzero[num_samples * idx:num_samples * (idx + 1)] = rv_samples_x.copy()
            agents_sample_y_nonzero[num_samples * idx:num_samples * (idx + 1)] = rv_samples_y.copy()
            agents_pdf_x_nonzero[idx] = agents_pdf_x[i].copy()
            agents_pdf_y_nonzero[idx] = agents_pdf_y[i].copy()

    # sample robot trajetory separately
    rv_x = mvn(mean=robot_mu_x, cov=robot_cov_x * scale, allow_singular=True)
    rv_samples_x = rv_x.rvs(size=num_samples)
    rv_samples_x[0] = robot_mu_x.copy()  # replace one sample with GP mean
    agents_sample_x[num_samples * num_peds:num_samples * (num_peds + 1)] = rv_samples_x.copy()
    agents_sample_x_nonzero[num_samples * num_peds_nonzero:num_samples * (num_peds_nonzero + 1)] = rv_samples_x.copy()
    agents_pdf_x[-1] = rv_x.pdf(rv_samples_x)
    agents_pdf_x_nonzero[-1] = agents_pdf_x[-1].copy()

    rv_y = mvn(mean=robot_mu_y, cov=robot_cov_y * scale, allow_singular=True)
    rv_samples_y = rv_y.rvs(size=num_samples)
    rv_samples_y[0] = robot_mu_y.copy()
    agents_sample_y[num_samples * num_peds:num_samples * (num_peds + 1)] = rv_samples_y
    agents_sample_y_nonzero[num_samples * num_peds_nonzero:num_samples * (num_peds_nonzero + 1)] = rv_samples_y.copy()
    agents_pdf_y[-1] = rv_y.pdf(rv_samples_y)
    agents_pdf_y_nonzero[-1] = agents_pdf_y[-1].copy()

    return agents_sample_x, agents_sample_y, agents_sample_x_nonzero, agents_sample_y_nonzero, \
           agents_pdf_x, agents_pdf_y, agents_pdf_x_nonzero, agents_pdf_y_nonzero