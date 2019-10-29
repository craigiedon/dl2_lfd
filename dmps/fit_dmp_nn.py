#%%
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.interpolate import interp1d
from dmps.dmp import DMP, imitate_path, load_dmp, save_dmp
from helper_funcs.utils import t_stamp
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from glob import glob
from math import ceil
import pickle
from scipy.spatial.transform import Rotation as R

# Load up the poses
# Create an initial DMP with parameters
# Create a Standard multi-layer ff neural network - inputs the start and goal poses, outpout the basis weights
# Run the start / goal poses through the network to get weights
# Calculate the error function, this is done by first interpolating the DMP with the dt=0.001 metric
# Then, roll out the DMP. You should now just be able to back-propagate this no problem...
# Roll out final one once converged and chart

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

def load_single_dim_demos(demos_folder):
    start_state_paths = sorted([d for d in os.listdir(demos_folder) if "start-state" in d])
    rollout_paths = sorted([d for d in os.listdir(demos_folder) if "rollout" in d])

    start_states = np.stack([np.loadtxt(join(demos_folder, sp)) for sp in start_state_paths])
    rollouts = np.stack([np.loadtxt(join(demos_folder, rp)).reshape(-1, 1) for rp in rollout_paths])

    return start_states, rollouts


def interpolated_path(recorded_ys, dt, T):
    demos, num_points, dims = recorded_ys.shape
    x = np.linspace(0, 1, num_points)

    path_gen = interp1d(x, recorded_ys, axis=1)
    path = path_gen([t*dt for t in range(T)])
    return path


def dmp_nn(in_dim, hidden_dim, out_dims):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dims)
    )


in_dim = 2
hidden_dim = 100
n_basis_funcs = 30
out_dims = n_basis_funcs * 1

start_states, pose_hists = load_single_dim_demos("demos/wave_combos-2019-10-28-17-43-46")
dt = 0.01
T = int(1 / dt)
# torch_orig_rollouts = torch.stack([torch.tensor(interpolated_path(p, dt, T)).T.to(dtype=torch.float) for p in pose_hists])

model = nn.Sequential(
    nn.Linear(in_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, out_dims)
)

optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()


# So data loader needs to load the start states and also the trajectories for each example. Joint tuple batch?
# Check out the TensorDataset constructor and have a play around...
print("Starts {}, Pose_hists: {}".format(start_states.shape, pose_hists.shape))
data_set = TensorDataset(torch.from_numpy(start_states).to(dtype=torch.float),
                         torch.from_numpy(interpolated_path(pose_hists, dt, T)).to(dtype=torch.float))

data_loader = DataLoader(data_set, shuffle=True, batch_size=16)



#%%

results_folder = "logs/synth-wave-{}".format(t_stamp())
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


#%%

for epoch in range(100):
    avg_loss = []
    for batch in data_loader:
        # in_poses = torch.cat((start_pose, goal_pose)).unsqueeze(0)
        in_poses = batch[0][:, [0, -1]]
        # print("In poses shape: {}".format(in_poses.shape))
        learned_weights = model(in_poses.reshape(in_poses.shape[0], -1))
        # print("Learned weights shape: {}".format(learned_weights.shape))

        # This part needs to be "vectorized"
        learned_dmp_rollouts = []
        for i, path in enumerate(batch[0]):
            dmp = DMP(in_poses[i][0],
                      in_poses[i][1],
                      num_basis_funcs=n_basis_funcs,
                      dt=0.01,
                      d=pose_hists.shape[2],
                      weights=learned_weights[i].reshape(1, n_basis_funcs))

            learned_dmp_rollouts.append(dmp.rollout_torch()[0])

        learned_dmp_rollouts = torch.stack(learned_dmp_rollouts)
        # print("LDMP Shape: {}, Batch0 Shape: {}".format(learned_dmp_rollouts.shape, batch[0].shape))
        loss = loss_fn(learned_dmp_rollouts, batch[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("e{} : {}".format(epoch, loss))
    if epoch % 10 == 0:
        torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_{}.pt".format(epoch)))
        
torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_final.pt"))


#%%
demo_num = 51
model.load_state_dict(
    torch.load(join(results_folder, "learned_model_epoch_final.pt"))
)
in_poses = torch_orig_rollouts[demo_num, [0, -1]]
learned_weights = model(in_poses.reshape(1, 2))

dmp = DMP(in_poses[0],
            in_poses[1],
            num_basis_funcs=n_basis_funcs,
            dt=0.01,
            d=pose_hists.shape[2],
            weights=learned_weights[0].reshape(1, n_basis_funcs))
learned_dmp_rollout = dmp.rollout_torch()[0]
dims = 1

for d in range(dims):
    plt.subplot(2,ceil(dims / 2),d + 1)
    dmp_timescale = np.linspace(0, 1, learned_dmp_rollout.shape[0])

    plt.plot(dmp_timescale, learned_dmp_rollout[:, d].detach(), label="DMP")
    print(torch_orig_rollouts.shape)
    plt.scatter(dmp_timescale, torch_orig_rollouts[demo_num, :, d].detach(), label="Raw")
    plt.legend()
plt.show()


#%%
