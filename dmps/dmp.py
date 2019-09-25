import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class CanonicalSystem():

    def __init__(self, dt, start_x=1.0, ax=1.0):
        self.ax = ax
        self.dt = dt
        self.start_x = start_x



def canonical_rollout(start_val, ax, dt):
    T = timesteps(dt)
    x_t = np.zeros(T)
    x_t[0] = start_val

    for t in range(1, T):
        x_t[t] = canonical_step(x_t[t-1], ax, dt)
    
    return x_t

def canonical_step(x, ax, dt):
    return x - ax * x * dt


def timesteps(dt):
    return int(1.0 / dt)


def rbf(x, h, c):
    return np.exp(-h * (x[:,None] - c)**2)

class DMP():
    def __init__(self, y0, goal, num_basis_funcs=5, dt=0.05, d=2, jnames=None):
        self.ay = np.ones(d) * 25
        self.by = self.ay / 4.0
        self.dt = dt
        self.n_basis_funcs = num_basis_funcs
        self.dims = d

        self.joint_names = jnames if jnames is not None else []

        self.cs = CanonicalSystem(dt=self.dt)
        self.T = timesteps(dt)

        # Spacing centres out equally isn't great if x is decaying non_linearly, so instead try exponential spacing
        self.c = np.exp(-self.cs.ax * np.linspace(0, 1, self.n_basis_funcs))
        self.h = np.ones(self.n_basis_funcs) * self.n_basis_funcs**1.5 / (self.c * self.cs.ax)

        # Start and goal points
        self.goal = goal
        self.y0 = y0

        self.weights = None


    def step(self, x, y, dy):
        # step canonical system
        x_next = canonical_step(x, self.cs.ax, self.cs.dt)

        # generate basis function activation
        psi = rbf(np.array([x_next]), self.h, self.c)

        f = np.zeros(self.dims)
        for d in range(self.dims):
            # generate the forcing term
            f[d] = (x_next * (self.goal[d] - self.y0[d]) * (np.dot(psi, self.weights[d])) / np.sum(psi))

        # DMP acceleration
        ddy_next = (self.ay * (self.by * (self.goal - y) - dy) + f)

        # DMP Velocity
        dy_next = dy + ddy_next * self.dt
        y_next = y + dy_next * self.dt

        return x_next, y_next, dy_next, ddy_next

    def rollout(self):
        # set up tracking vectors
        y_track = np.zeros((self.T, self.dims))
        dy_track = np.zeros((self.T, self.dims))
        ddy_track = np.zeros((self.T, self.dims))

        y_track[0] = self.y0

        x = self.cs.start_x

        for t in range(1, self.T):
            x, y_track[t], dy_track[t], ddy_track[t] = self.step(x, y_track[t-1], dy_track[t-1])

        return y_track, dy_track, ddy_track

    
def interpolated_path(recorded_ys, dt):
    T = timesteps(dt)
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

    # generate function to interpolate the desired trajectory
    path = interpolated_path(y_d, dmp.dt)
    dims =  path.shape[0]

    # Calculate the velocity of y_des
    dy_d = np.diff(path) / dmp.dt

    # Add zero to the beginning of every row
    dy_d = np.hstack((np.zeros((dims, 1)), dy_d))

    # calculate the acceleration of y_des
    ddy_d = np.diff(dy_d) / dmp.dt
    ddy_d = np.hstack((np.zeros((dims, 1)), ddy_d))

    # find the force required to move along this trajectory
    f_target = ddy_d.T - dmp.ay * (dmp.by * (y_goal - path.T) - dy_d.T)

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
        k = (y_goal[d] - y_start[d])
        for b in range(dmp.n_basis_funcs):
            weights[d,b] = (np.sum(x_track * psi_track[:, b] * f_target[:, d])) / (np.sum(x_track**2*psi_track[:, b]) * k)
    weights = np.nan_to_num(weights)
    return weights


def save_dmp(dmp, dmp_path):
    with open(dmp_path, "wb") as f:
        pickle.dump(dmp, f, pickle.HIGHEST_PROTOCOL)


def load_dmp(dmp_path):
    with open(dmp_path,"rb") as f:
        dmp = pickle.load(f)
    return dmp


def plot_rollout(dmp_rollout, raw_path):
    plt.subplot(1, 2, 1)
    plt.cla()
    plt.plot(dmp_rollout)
    plt.subplot(1, 2, 2)
    plt.cla()
    plt.plot(raw_path)
    plt.draw()
    plt.pause(0.1)