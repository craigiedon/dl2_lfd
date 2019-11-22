import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle
from math import ceil
import torch
import os
from os.path import join

def interpolated_path(recorded_ys, dt, T):
    demos, num_points, dims = recorded_ys.shape
    x = np.linspace(0, 1, num_points)

    path_gen = interp1d(x, recorded_ys, axis=1)
    path = path_gen([t*dt for t in range(T)])
    return path

def load_dmp_demos(demos_folder):
    start_state_paths = sorted([d for d in os.listdir(demos_folder) if "start-state" in d])
    rollout_paths = sorted([d for d in os.listdir(demos_folder) if "rollout" in d])

    start_states = np.stack([np.loadtxt(join(demos_folder, sp), ndmin=2) for sp in start_state_paths])
    rollouts = np.stack([np.loadtxt(join(demos_folder, rp), ndmin=2) for rp in rollout_paths])

    return start_states, rollouts

class CanonicalSystem():

    def __init__(self, dt, start_x=1.0, ax=5.0):
        # self.ax = -np.log(cutoff)
        self.ax = ax
        self.dt = dt
        self.start_x = start_x
        self.T = timesteps(dt)



def canonical_rollout(start_val, ax, dt, tau=1.0):
    T = timesteps(dt)
    x_t = np.zeros(T)
    x_t[0] = start_val

    for t in range(1, T):
        x_t[t] = canonical_step(x_t[t-1], ax, dt, tau)
    
    return x_t

def canonical_step(x, ax, dt, tau=1.0):
    return x - (ax * x * dt) * tau


def timesteps(dt):
    return int(1.0 / dt)


def rbf(x, h, c):
    return np.exp(-h * (x[:,None] - c)**2)

def rbf_torch(x, h, c):
    return torch.exp(-h * (x - c)**2)

class DMP():
    def __init__(self, num_basis_funcs, dt, d):
        self.ay = 25
        self.by = self.ay / 4.0

        self.dt = dt
        self.T = timesteps(dt)
        self.n_basis_funcs = num_basis_funcs
        self.dims = d

        self.cs = CanonicalSystem(dt=self.dt)

        self.c = np.exp(-self.cs.ax * np.linspace(0, 1, self.n_basis_funcs))
        self.h = np.square(np.diff(self.c) * 0.55)
        self.h = 1.0 / (np.append(self.h, self.h[-1]))

        # Torch equivalents
        self.t_h = torch.from_numpy(self.h).to(dtype=torch.float, device=torch.device("cuda"))
        self.t_c = torch.from_numpy(self.c).to(dtype=torch.float, device=torch.device("cuda"))

        # Start and goal points
        # self.goal = goal
        # self.y0 = y0


    def step(self, x, y, dy, tau=1.0):
        # step canonical system
        x_next = canonical_step(x, self.cs.ax, self.cs.dt, tau)

        # generate basis function activation
        psi = rbf(np.array([x_next]), self.h, self.c)

        f = np.zeros(self.dims)
        for d in range(self.dims):
            # generate the forcing term
            f[d] = x_next * (np.dot(psi, self.weights[d])) / np.sum(psi)

        # DMP acceleration
        # Sugar notation to agree with Pastor (2008) notation
        K = self.ay * self.by
        D = self.ay

        # Modified generalized
        ddy_next = (K * (self.goal - y) - D * dy / tau - K * (self.goal - self.y0) * x_next + K * f) * tau

        # Original Form
        # ddy_next = K * (self.goal - y) - D * dy + (self.goal - self.y0) * f

        # DMP Velocity
        dy_next = dy + (ddy_next * self.dt) * tau
        y_next = y + dy_next * self.dt

        return x_next, y_next, dy_next, ddy_next

    def step_torch(self, starts, goals, x, y, dy, weights, tau=1.0):
        # step canonical system
        x_next = canonical_step(x, self.cs.ax, self.cs.dt, tau)

        # generate basis function activation
        psi = rbf_torch(x_next, self.t_h, self.t_c)

        f = x_next * torch.matmul(weights, psi) / torch.sum(psi)

        # DMP acceleration
        # Sugar notation to agree with Pastor (2008) notation
        # K = torch.from_numpy(self.ay * self.by).to(dtype=torch.float, device=torch.device("cuda"))
        # D = torch.from_numpy(self.ay).to(dtype=torch.float, device=torch.device("cuda"))
        K = self.ay * self.by
        D = self.ay

        # Modified generalized
        ddy_next = (K * (goals - y) - D * dy / tau - K * (goals - starts) * x_next + K * f) * tau

        # Original Form
        # ddy_next = K * (self.goal - y) - D * dy + (self.goal - self.y0) * f

        # DMP Velocity
        dy_next = dy + (ddy_next * self.dt) * tau
        y_next = y + dy_next * self.dt

        return x_next, y_next, dy_next, ddy_next


    def rollout(self, tau=1.0):
        scaled_time = int(self.T / tau)
        # set up tracking vectors
        y_track = np.zeros((scaled_time, self.dims))
        dy_track = np.zeros((scaled_time, self.dims))
        ddy_track = np.zeros((scaled_time, self.dims))

        y_track[0] = self.y0

        x = self.cs.start_x

        for t in range(1, scaled_time):
            x, y_track[t], dy_track[t], ddy_track[t] = self.step(x, y_track[t-1], dy_track[t-1], tau)

        return y_track, dy_track, ddy_track

    def rollout_torch(self, starts, goals, weights, tau=1.0):
        scaled_time = int(self.T / tau)
        batch_size = starts.shape[0]
        y_track = torch.zeros((batch_size, scaled_time, self.dims), device=torch.device("cuda"))
        dy_track = torch.zeros((batch_size, scaled_time, self.dims), device=torch.device("cuda"))
        ddy_track = torch.zeros((batch_size, scaled_time, self.dims), device=torch.device("cuda"))

        y_track[:, 0] = starts.reshape(batch_size, -1)

        x = self.cs.start_x

        for t in range(1, scaled_time):
            x, y_track[:, t], dy_track[:, t], ddy_track[:,t] = self.step_torch(
                starts.view(batch_size, -1),
                goals.view(batch_size, -1),
                x,
                y_track[:, t-1],
                dy_track[:, t-1],
                weights,
                tau)

        return y_track, dy_track, ddy_track


    
def interpolated_path(recorded_ys, dt, T):
    num_points, dims = recorded_ys.shape[0], recorded_ys.shape[1]
    path = np.zeros((dims, T))
    x = np.linspace(0, 1, num_points)

    for d in range(dims):
        path_gen = interp1d(x, recorded_ys[:, d])
        path[d] = path_gen([t*dt for t in range(T)])

    return path

def imitate_path(y_d, dmp):
    # Set initial state and goal
    y_start = y_d[0, :].copy()
    y_goal = y_d[-1, :].copy()

    x_track = canonical_rollout(dmp.cs.start_x, dmp.cs.ax, dmp.dt)

    # generate function to interpolate the desired trajectory
    path = interpolated_path(y_d, dmp.dt, dmp.T)
    dims =  path.shape[0]

    # Calculate the velocity of y_des
    dy_d = np.diff(path) / dmp.dt

    # Add zero to the beginning of every row
    dy_d = np.hstack((np.zeros((dims, 1)), dy_d))

    # calculate the acceleration of y_des
    ddy_d = np.diff(dy_d) / dmp.dt
    ddy_d = np.hstack((np.zeros((dims, 1)), ddy_d))

    # find the force required to move along this trajectory
    K = dmp.ay * dmp.by
    D = dmp.ay

    # Original form...
    # f_target = (ddy_d.T - (K * (y_goal - path.T) - D * dy_d.T))  / (dmp.goal - dmp.y0)

    # Modified general one
    f_target = (ddy_d.T + D * dy_d.T) / K - (dmp.goal - path.T) + (dmp.goal - dmp.y0) * np.tile(x_track, (dims, 1)).T

    # efficiently generate weights to realize f_target
    weights = gen_weights(f_target, y_start, y_goal, dmp) 

    return path, weights


def gen_weights(f_target, y_start, y_goal, dmp):

    # calculate x and psi
    x_track = canonical_rollout(dmp.cs.start_x, dmp.cs.ax, dmp.dt)
    psi_track = rbf(x_track, dmp.h, dmp.c)

    # efficiently calculate BF weights using weighted linear regression
    weights = np.zeros((dmp.dims, dmp.n_basis_funcs))
    for d in range(dmp.dims):
        # spatial scaling term
        # k = (y_goal[d] - y_start[d])
        for b in range(dmp.n_basis_funcs):
            weights[d,b] = (np.sum(x_track * psi_track[:, b] * f_target[:, d])) / (np.sum(x_track**2*psi_track[:, b]))
    weights = np.nan_to_num(weights)
    return weights


def save_dmp(dmp, dmp_path):
    with open(dmp_path, "wb") as f:
        pickle.dump(dmp, f, 2)


def load_dmp(dmp_path):
    with open(dmp_path,"rb") as f:
        dmp = pickle.load(f)
    return dmp


def plot_rollout(dmp_rollout, raw_path = None):
    dims = dmp_rollout.shape[1]
    for d in range(dims):
        plt.subplot(2,ceil(dims / 2),d + 1)
        dmp_timescale = np.linspace(0, 1, dmp_rollout.shape[0])
        plt.plot(dmp_timescale, dmp_rollout[:, d], label="DMP")

        if raw_path is not None:
            raw_timescale = np.linspace(0, 1, raw_path.shape[0])
            plt.plot(raw_timescale, raw_path[:, d], label="Raw")
        plt.legend()
    plt.show()
