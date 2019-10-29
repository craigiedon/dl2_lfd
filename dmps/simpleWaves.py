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


xs = np.linspace(0, 1, 100)

demo_dir = "demos/wave_combos-{}".format(t_stamp())
os.mkdir(demo_dir)

for i in range(250):
    start = np.random.uniform(0, 2)
    goal = np.random.uniform(4, 6)
    c1 = np.random.uniform(0.1, 0.45)
    c2 = np.random.uniform(0.55, 0.9)
    start_state = np.stack([start, goal, c1, c2])
    ys = force_func(start, goal, c1, c2, xs)
    plt.scatter(xs, ys, alpha=0.5, marker='o', s=20)
    plt.scatter([c1, c2], [1, 1], marker='x')
    np.savetxt("{}/start-state-{}.txt".format(demo_dir, i), start_state)
    np.savetxt("{}/rollout-{}.txt".format(demo_dir, i), ys, fmt='%5f')

plt.show()
plt.close()