#%%
import numpy as np
from helper_funcs.rm import RobotModel

"""
This script takes the RobotModel helper functions and uses
it to save the required joint names to a txt file.

This was done because rm.py is python2 dependent. Saving the
required info to a text file cuts this dependency and allows the
pytorch code to leverage python3
"""

k_kinect = np.array([366.096588, 0, 268, 0, 366.096588, 192, 0, 0, 1]) \
    .reshape(3, 3)
robot_model = RobotModel("./pr2.xml", "base_link", "r_gripper_tool_frame", k_kinect)
arm_joint_names = robot_model.kdl_kin.get_joint_names()

SAVE_PATH = "./arm_joint_names.txt"
with open(SAVE_PATH, 'w') as file_handler:
        file_handler.writelines("{}\n".format(item) for item in arm_joint_names)


print("Joint Names:")
print(arm_joint_names)
print("Saved to:", SAVE_PATH)