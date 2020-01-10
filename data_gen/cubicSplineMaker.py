import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform as unif
from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy import optimize
from helper_funcs.utils import t_stamp
import os
from os.path import join
from curve_funcs import equally_space_curve



demo_dir = "demos/splines2D-{}".format(t_stamp())

os.mkdir(demo_dir)
os.mkdir(demo_dir + "/train")
os.mkdir(demo_dir + "/val")

ts = np.array([0.0, 0.25, 0.75, 1.0])

fig, ax = plt.subplots()

# Training Data
for i in range(70):
    start = unif([0.0, 0.0], [0.1, 0.1])
    goal = unif([0.9, 0.9], [1.0, 1.0])
    c1 = unif([0.5, 0.05], [0.65, 0.25])
    c2 = unif([0.05, 0.5], [0.25, 0.65])
    distractor_1 = unif([0.3, 0.3], [0.7, 0.7])

    relevant_keypoints = np.array([start, c1, c2, goal])
    all_keypoints = np.array([start, c1, c2, distractor_1, goal])
    spline = CubicSpline(ts, relevant_keypoints)

    true_xys = spline(np.linspace(0, 1, 100))
    unif_xys = equally_space_curve(spline, 100)

    # noisy_xys =  true_xys + np.random.randn(100, 2) * 0.02
    #ax.scatter(true_xys[:, 0], true_xys[:, 1], alpha=0.4, label='squashed')
    # ax.scatter(noisy_xys[:, 0], noisy_xys[:, 1], alpha=0.25)

    ax.plot(unif_xys[:, 0], unif_xys[:, 1], label='uniform')
    ax.scatter(all_keypoints[:, 0], all_keypoints[:, 1], marker='x', c='r')

    np.savetxt("{}/train/start-state-{}.txt".format(demo_dir, i), all_keypoints, fmt='%5f')
    np.savetxt("{}/train/rollout-{}.txt".format(demo_dir, i), unif_xys, fmt='%5f')

plt.show()

fig, ax = plt.subplots()
# Validation Data
for i in range(30):
    start = unif([0.0, 0.0], [0.1, 0.1])
    goal = unif([0.9, 0.9], [1.0, 1.0])
    c1 = unif([0.65, 0.25], [0.95, 0.5])
    c2 = unif([0.25, 0.65], [0.5, 0.95])
    distractor_1 = unif([0.1, 0.1], [0.9, 0.9])

    relevant_keypoints = np.array([start, c1, c2, goal])
    all_keypoints = np.array([start, c1, c2, distractor_1, goal])
    spline = CubicSpline(ts, relevant_keypoints)

    true_xys = spline(np.linspace(0, 1, 100))
    unif_xys = equally_space_curve(spline, 100)

    # noisy_xys =  true_xys + np.random.randn(100, 2) * 0.02
    # ax.plot(true_xys[:, 0], true_xys[:, 1])
    # ax.scatter(noisy_xys[:, 0], noisy_xys[:, 1], alpha=0.25)
    ax.plot(unif_xys[:, 0], unif_xys[:, 1], label='uniform')
    ax.scatter(all_keypoints[:, 0], all_keypoints[:, 1], marker='x', c='r')

    np.savetxt("{}/val/start-state-{}.txt".format(demo_dir, i), all_keypoints, fmt='%5f')
    np.savetxt("{}/val/rollout-{}.txt".format(demo_dir, i), unif_xys, fmt='%5f')

plt.show()