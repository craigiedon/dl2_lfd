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

c1 = np.random.uniform([0.6, 0.1], [0.8, 0.35])
c2 = np.random.uniform([0.2, 0.5], [0.3, 0.8])
xy_data = np.array([[0.0, 0.0], c1, c2, [1.0, 1.0]])
spline = CubicSpline(ts, xy_data)
xys = spline(np.linspace(0, 1, 100))


# Just Plot Start and End
plt.scatter(xy_data[0, 0], xy_data[0, 1], marker='o', c='green', s=100)
plt.text(xy_data[0,0] + 0.025, xy_data[0, 1], "Start", fontsize=16)
plt.scatter(xy_data[3, 0], xy_data[3, 1], marker='o', c='green', s=100)
plt.text(xy_data[3,0] + 0.025, xy_data[3, 1], "Goal", fontsize=16)
plt.plot([xy_data[0,0], xy_data[3, 0]], [xy_data[0, 1], xy_data[3, 1]], '--', c="black", linewidth=2, alpha=0.3)
plt.show()

# Plot the Collision Points too
plt.scatter(xy_data[0, 0], xy_data[0, 1], marker='o', c='green', s=100)
plt.scatter(xy_data[1, 0], xy_data[1, 1], marker='x', c='r', s=50)
plt.scatter(xy_data[2, 0], xy_data[2, 1], marker='x', c='r', s=50)
plt.scatter(xy_data[3, 0], xy_data[3, 1], marker='o', c='green', s=100)
plt.show()

plt.scatter(xy_data[0, 0], xy_data[0, 1], marker='o', c='green', s=100)
plt.scatter(xy_data[1, 0], xy_data[1, 1], marker='x', c='r', s=50)
plt.scatter(xy_data[2, 0], xy_data[2, 1], marker='x', c='r', s=50)
plt.scatter(xy_data[3, 0], xy_data[3, 1], marker='o', c='green', s=100)
plt.plot(xys[:, 0], xys[:, 1])
plt.show()