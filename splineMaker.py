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
for i in range(10):

    start = np.random.uniform([0.0, 0.0], [0.1, 0.1])
    goal = np.random.uniform([0.9, 0.9], [1.0, 1.0])
    c1 = np.random.uniform([0.5, 0.05], [0.95, 0.5])
    c2 = np.random.uniform([0.05, 0.5], [0.45, 0.95])

    distractor_1 = np.random.uniform([0.1, 0.1], [0.9, 0.9])

    relevant_keypoints = np.array([start, c1, c2, goal])
    all_keypoints = np.array([start, c1, c2, distractor_1, goal])
    spline = CubicSpline(ts, relevant_keypoints)

    true_xys = spline(np.linspace(0, 1, 100))
    noisy_xys =  true_xys + np.random.randn(100, 2) * 0.02
    # print(xys)

    ax.plot(true_xys[:, 0], true_xys[:, 1])
    ax.scatter(noisy_xys[:, 0], noisy_xys[:, 1], alpha=0.25)
    ax.scatter(all_keypoints[:, 0], all_keypoints[:, 1], marker='x', c='r')

    np.savetxt("{}/start-state-{}.txt".format(demo_dir, i), all_keypoints, fmt='%5f')
    np.savetxt("{}/rollout-{}.txt".format(demo_dir, i), noisy_xys, fmt='%5f')

plt.show()