import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from dmps.dmp import DMP, imitate_path, load_dmp, save_dmp
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from glob import glob
from math import ceil
from load_data import quat_pose_to_rpy
import pickle

def load_pose_history(demos_folder, demo_num, ee_name):
    demo_path = os.listdir(demos_folder)[demo_num]
    ee_template = join(demos_folder, demo_path, "{}*".format(ee_name))
    sorted_pose_paths = sorted(glob(ee_template))
    pose_history = np.stack([quat_pose_to_rpy(np.genfromtxt(p)) for p in sorted_pose_paths])
    return pose_history

def fit_dmp(pose_path):
    start_pose, goal_pose = pose_path[0, :], pose_path[-1, :]
    dims = pose_path.shape[1]

    dmp = DMP(start_pose, goal_pose, num_basis_funcs=500, dt=0.01, d=dims)
    _, weights = imitate_path(pose_path, dmp)
    dmp.weights = weights

    y_r, _, _ = dmp.rollout()

    print("Rollout shapes: {}".format(y_r.shape))
    print("Raw pose shapes: {}".format(pose_path.shape))


    for d in range(dims):
        plt.subplot(2,ceil(dims / 2),d + 1)
        dmp_timescale = np.linspace(0, 1, y_r.shape[0])
        plt.plot(dmp_timescale, y_r[:, d], label="DMP")

        raw_timescale = np.linspace(0, 1, pose_path.shape[0])
        plt.plot(raw_timescale, pose_path[:, d], label="Raw")
        plt.legend()
    plt.show()

    # Plot this thing out here...
    return dmp



if __name__ == "__main__":
    dmp = load_dmp("saved_dmps/dmp-0-l_wrist_roll_link.npy")
    print(dmp)
    y_r = dmp.rollout()[0]
    for d in range(dmp.dims):
        plt.subplot(2,ceil(dmp.dims / 2),d + 1)
        dmp_timescale = np.linspace(0, 1, y_r.shape[0])
        plt.plot(dmp_timescale, y_r[:, d], label="DMP")
        plt.legend()
    plt.show()

    # demo_folder = "demos/gear_good"
    # demo_num = 0
    # ee_name = "l_wrist_roll_link"

    # pose_hist = load_pose_history(demo_folder, demo_num, "l_wrist_roll_link")
    # dmp = fit_dmp(pose_hist)

    # with open("./saved_dmps/dmp-{}-{}.npy".format(demo_num, ee_name), "wb") as f:
    #     pickle.dump(dmp, f, pickle.HIGHEST_PROTOCOL)