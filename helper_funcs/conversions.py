import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def quat_np_pose_to_rpy(quat_pose, normalize):
    pos, quat = quat_pose[0:3], quat_pose[3:]
    rpy = R.from_quat(quat).as_euler("xyz")
    if normalize:
        rpy = rpy / np.pi

    return np.concatenate((pos, rpy))


def np_to_pgpu(np_array):
    return torch.from_numpy(np_array).to(dtype=torch.float, device=torch.device("cuda"))