import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform as unif
from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy import optimize
from helper_funcs.utils import t_stamp
import os
from os.path import join
import data_gen.curve_funcs as cf

def gen_multiple_curves(demo_name, num_demos, curve_fun, *curve_args):
    # demo_dir = "demos/{}-{}".format(demo_name, t_stamp())
    demo_dir = "demos/{}".format(demo_name)
    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(demo_dir + "/train", exist_ok=True)
    os.makedirs(demo_dir + "/val", exist_ok=True)

    # Train
    for i in range(num_demos):
        while True:
            start_features, curve_rollout = curve_fun(*curve_args)
            if len(curve_rollout) == 100:
                break

        np.savetxt("{}/train/start-state-{}.txt".format(demo_dir, i), start_features, fmt='%5f')
        np.savetxt("{}/train/rollout-{}.txt".format(demo_dir, i), curve_rollout, fmt='%5f')

    plt.show()

    # Validation
    for i in range(num_demos):
        while True:
            start_features, curve_rollout = curve_fun(*curve_args)
            if len(curve_rollout) == 100:
                break

        np.savetxt("{}/val/start-state-{}.txt".format(demo_dir, i), start_features, fmt='%5f')
        np.savetxt("{}/val/rollout-{}.txt".format(demo_dir, i), curve_rollout, fmt='%5f')

    plt.show()


gen_multiple_curves("avoid", 100, cf.simpleCurveWithAvoidPoint, ([0.0, 0.0], [0.1,0.1]), ([0.9, 0.9], [1.0, 1.0]), ([0.25, 0.25], [0.8, 0.8]), False)
gen_multiple_curves("patrol", 100, cf.simpleCurveWithTwoPatrols, ([0.0, 0.0], [0.1, 0.1]), ([0.9, 0.9], [1.0, 1.0]), ([0.2, 0.2], [0.4, 0.4]), ([0.6, 0.6], [0.8, 0.8]), False)
gen_multiple_curves("stable", 100, cf.movingSinWave, ([0.0, 0.45], [0.0, 0.55]), ([1.0, 0.4], [1.0, 0.6]), ([0.2, 0.3], [0.4, 0.7]), False)
gen_multiple_curves("slow", 100, cf.unevenSpeeds, ([0.0, 0.0], [0.1,0.1]), ([0.9, 0.9], [1.0, 1.0]), ([0.25, 0.25], [0.8, 0.8]), False)
