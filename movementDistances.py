from load_data import image_demo_paths, PoseAndGoal
from helper_funcs.utils import load_json
import torch

exp_config = load_json("config/experiment_config.json")
im_params = exp_config["image_config"]

train_paths = image_demo_paths(exp_config["demo_folder"], im_params["file_glob"], from_demo=0, to_demo=60)
train_set = PoseAndGoal(train_paths, "l_wrist_roll_link", "r_wrist_roll_link", skip_count=10)

dists = torch.stack([torch.dist(c_pose[0:3], n_pose[0:3]) for c_pose, g_pose, n_pose in train_set])
print("Mean: ", dists.mean())
print("Median: ", dists.median())
print("Max: ", dists.max())
print("Min: ", dists.min())
print("Sorted: ", dists.sort(descending=True))
print("Shape: ", dists.shape)