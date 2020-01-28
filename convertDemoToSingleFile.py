import os
from glob import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
from os.path import join, split
import re

def get_pose(file_path, ee_name):
    folder_path, f_name = split(file_path)
    ts_reg = re.compile(r'.*_(\d+)\.txt')
    repl_form = r'{}_\1.txt'.format(ee_name)
    pose_path = join(folder_path, ts_reg.sub(repl_form, f_name))
    return np.genfromtxt(pose_path)

def quat_pose_to_rpy(quat_pose):
    pos, quat = quat_pose[0:3], quat_pose[3:]
    rpy = R.from_quat(quat).as_euler("xyz") / np.pi
    return np.concatenate((pos, rpy))


demo_path = "demos/wobblyPour"
file_glob = "r_wrist_roll_link_*.txt"

# Get all the time-stamped right end-effector files (7DOFs-x y z q-x q-y q-z q-w, 7 lines each file)
demo_ee_points = sorted(glob(join(demo_path, file_glob)))

right_poses = []
left_poses = []
for f_id in demo_ee_points:
    right_ee_pose_q = get_pose(f_id, "r_wrist_roll_link")
    left_ee_pose_q = get_pose(f_id, "l_wrist_roll_link")

    right_poses.append(quat_pose_to_rpy(right_ee_pose_q))
    left_poses.append(quat_pose_to_rpy(left_ee_pose_q))

right_poses = np.stack(right_poses)
left_poses = np.stack(left_poses)

# So, the rollout file is literally just a save of the left-hand poses
# The start file is: The start pos of left_poses (i.e., left_poses[0]), the end pose (left_poses[-1]), and also any "object inputs". I.e., optional flag to add in right_poses[0]
output_path = "{}-flat".format(demo_path)
os.makedirs(join(output_path, "train"), exist_ok=True)
os.makedirs(join(output_path, "val"), exist_ok=True)
np.savetxt(join(output_path, "train/rollout-0.txt"), left_poses)

start_info = np.stack((left_poses[0], right_poses[0], left_poses[-1]))
np.savetxt(join(output_path, "train/start-state-0.txt"), start_info)