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
from data_gen.curve_funcs import equally_space_curve, simpleCurveWithAvoidPoint

# Generate Curve

demo_dir = "demos/splines2D-{}".format(t_stamp())
os.mkdir(demo_dir)
os.mkdir(demo_dir + "/train")
os.mkdir(demo_dir + "/val")

# Gen Curve with an added variable that you have to avoid
for i in range(1):
    start_features, curve_rollout = simpleCurveWithAvoidPoint(([0.0, 0.0], [0.0,0.0]), ([1.0, 1.0], [1.0, 1.0]), [0.25, 0.8], [0.25, 0.8])

    # Gen Curve with an added variable that you have to reach (not reached in demo)
    # Gen Curve with two variables you have to reach (not reached in demo)
    # Generate Multi-Dim Wobbly Sinusoid (wobbles of different amplitudes, freq? Maybe one just bulges or has noise?) This is going to be for "keep things steady"
    # Generate uneven velocity curve (for "non-jerky" constraint)

    # Save everything to file
    np.savetxt("{}/train/start-state-{}.txt".format(demo_dir, i), start_features, fmt='%5f')
    np.savetxt("{}/train/rollout-{}.txt".format(demo_dir, i), curve_rollout, fmt='%5f')

plt.show()