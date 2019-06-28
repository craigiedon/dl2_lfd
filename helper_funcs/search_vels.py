import numpy as np
from glob import glob
import re

joint_poses = glob("demos/**/joint_pos*", recursive=True)

for v in joint_poses:
    pos = np.genfromtxt(v)
    threshold = np.pi * 2
    if len(pos[pos > threshold]) > 0:
        joint_name_file = re.sub(r'joint_pos', r'joint_names', v)
        arm_joint_names = np.genfromtxt(joint_name_file, np.str)
        # Find index of value with 
        # print(v)
        # print(pos)
        print("Name of violation: ", arm_joint_names[pos > threshold], pos[pos > threshold])
        # print(arm_joint_names)
        # print(pos)


