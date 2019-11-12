import os
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
from helper_funcs.utils import t_stamp
import numpy as np
import matplotlib.pyplot as plt


def rbf(x, c, h):
    return np.exp(-h * (x - c)**2)

def force_func(start, goal, c1, c2, x):
    return start + x * (goal - start)  + rbf(x, c1, 100) - rbf(x, c2, 150)

def force_func_2d(start, goal, c1, t):
    straights = start + (goal - start) * t[:, None]
    return straights + rbf(straights, c1, 100)



demo_dir = "demos/wave_combos-{}".format(t_stamp())
os.mkdir(demo_dir)

# Ok so how about you generate an x and y?
ts = np.linspace(0, 1, 100)

for i in range(1):
    start = np.array([0.0, 0.0])
    goal = np.array([1.0, 1.0])
    c1 = np.array([0.5, 0.75])
    ps = force_func_2d(start, goal, c1, ts)
    plt.scatter(ps[:, 0], ps[:, 1])

    # start = np.random.uniform(0, 0.1)
    # goal = np.random.uniform(0.9, 1.0)
    # c1 = np.random.uniform(0.15, 0.45)
    # c2 = np.random.uniform(0.55, 0.85)
    # start_state = np.stack([start, goal, c1, c2])
    # ys = force_func(start, goal, c1, c2, xs)
    # plt.scatter(xs, ys, alpha=0.5, marker='o', s=20)
    # plt.scatter([c1, c2], [1, 1], marker='x')
    # np.savetxt("{}/start-state-{}.txt".format(demo_dir, i), start_state)
    # np.savetxt("{}/rollout-{}.txt".format(demo_dir, i), ys, fmt='%5f')

plt.show()
plt.close()