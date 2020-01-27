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
import pickle
from scipy.spatial.transform import Rotation as R

def quat_pose_to_rpy(quat_pose, normalize):
    pos, quat = quat_pose[0:3], quat_pose[3:]
    rpy = R.from_quat(quat).as_euler("xyz")
    if normalize:
        rpy = rpy / np.pi

    return np.concatenate((pos, rpy))

def load_pose_history(demos_folder, demo_num, ee_name, convert_to_rpy=True):
    demo_path = os.listdir(demos_folder)[demo_num]
    ee_template = join(demos_folder, demo_path, "{}*".format(ee_name))
    sorted_pose_paths = sorted(glob(ee_template))

    if convert_to_rpy:
        pose_history = np.stack([quat_pose_to_rpy(np.genfromtxt(p), False) for p in sorted_pose_paths])
    else:
        pose_history = np.stack([np.genfromtxt(p) for p in sorted_pose_paths])

    return pose_history


def fit_dmp(pose_path):
    start_pose, goal_pose = pose_path[0, :], pose_path[-1, :]
    dims = pose_path.shape[1]

    dmp = DMP(start_pose, goal_pose, num_basis_funcs=50, dt=0.01, d=dims)
    _, weights = imitate_path(pose_path, dmp)
    dmp.weights = weights

    # print("DMP Goal Original: {}", dmp.goal)

    y_r = dmp.rollout()[0]

    # ## Testing the generalization properties:
    # dmp.y0 = dmp.y0 + np.ones(6) * 0.05
    # dmp.goal = dmp.goal - np.ones(6) * 0.05
    # dmp.dt = 0.01
    # dmp.T /= 2
    # dmp.dt *= 2
    tau = 1
    y_r_shifted = dmp.rollout(tau=tau)[0]

    # print("DMP Goal Post y0 Shift: {}", dmp.goal)

    # print("Rollout shapes: {}".format(y_r.shape))
    # print("Raw pose shapes: {}".format(pose_path.shape))



    for d in range(dims):
        plt.subplot(2,ceil(dims / 2),d + 1)
        dmp_timescale = np.linspace(0, 1, y_r.shape[0])
        plt.plot(dmp_timescale, y_r[:, d], label="DMP")
        plt.plot()
        # plt.plot(y_r[:, d], label="DMP")

        raw_timescale = np.linspace(0, 1, pose_path.shape[0])
        plt.plot(raw_timescale, pose_path[:, d], label="Raw")
        # plt.scatter(1 - dmp.c, np.ones(dmp.c.shape[0]) * np.mean(pose_path[:, d]), alpha=0.4)
        # plt.plot(pose_path[:, d], label="Raw")

        # dmp_shifted_timescale = np.linspace(0, tau, y_r_shifted.shape[0])
        # plt.plot( y_r_shifted[:, d], label="DMP Shifted")

        # plt.plot(0, [dmp.y0[d]], marker='o', markersize=3, color="red")
        # plt.plot(tau, [dmp.goal[d]], marker='o', markersize=3, color="red")
        plt.legend()
    plt.show()


    # Plot this thing out here...
    return dmp



if __name__ == "__main__":
    # dmp = load_dmp("saved_dmps/dmp-0-l_wrist_roll_link.npy")
    # print(dmp)
    # y_r = dmp.rollout()[0]
    # for d in range(dmp.dims):
    #     plt.subplot(2,ceil(dmp.dims / 2),d + 1)
    #     dmp_timescale = np.linspace(0, 1, y_r.shape[0])
    #     plt.plot(dmp_timescale, y_r[:, d], label="DMP")
    #     plt.legend()
    # plt.show()
    if len(sys.argv) != 4:
        print("Usage: python fit_dmp_pose_saved.py <demo_folder> <demo_num> <ee-name>")
        sys.exit(0)

    demo_folder, demo_num, ee_name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    print(demo_folder.split('/'))

    # demo_folder = "demos/gear_good"
    # demo_num = 0
    # ee_name = "l_wrist_roll_link"

    pose_hist = load_pose_history(demo_folder, demo_num, "l_wrist_roll_link")
    dmp = fit_dmp(pose_hist)

    save_dmp(dmp, "./saved_dmps/dmp-{}-{}.npy".format(demo_folder.split('/')[-1], demo_num))