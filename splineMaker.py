import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy import optimize
from helper_funcs.utils import t_stamp
import os
from os.path import join


demo_dir = "demos/splines2D-{}".format(t_stamp())
os.mkdir(demo_dir)
ts = np.array([0.0, 0.25, 0.75, 1.0])

fig, ax = plt.subplots()
for i in range(1):
    c1 = np.random.uniform([0.6, 0.1], [0.8, 0.35])
    c2 = np.random.uniform([0.2, 0.5], [0.3, 0.8])
    xy_data = np.array([[0.0, 0.0], c1, c2, [1.0, 1.0]])
    spline = CubicSpline(ts, xy_data)
    xys = spline(np.linspace(0, 1, 100))

    ax.plot(xys[:, 0], xys[:, 1])
    ax.scatter(xy_data[:, 0], xy_data[:, 1], marker='x', c='r')

    # np.savetxt("{}/start-state-{}.txt".format(demo_dir, i), xy_data, fmt='%5f')
    # np.savetxt("{}/rollout-{}.txt".format(demo_dir, i), xys, fmt='%5f')

plt.show()