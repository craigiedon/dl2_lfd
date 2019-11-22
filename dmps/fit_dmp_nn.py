#%%
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.interpolate import interp1d
from dmps.dmp import DMP, imitate_path, load_dmp, save_dmp
from helper_funcs.utils import t_stamp, temp_print
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from glob import glob
from math import ceil
import pickle
from IPython.display import display, clear_output, update_display
from time import time



def load_demos(demos_folder):
    start_state_paths = sorted([d for d in os.listdir(demos_folder) if "start-state" in d])
    rollout_paths = sorted([d for d in os.listdir(demos_folder) if "rollout" in d])

    start_states = np.stack([np.loadtxt(join(demos_folder, sp), ndmin=2) for sp in start_state_paths])
    rollouts = np.stack([np.loadtxt(join(demos_folder, rp), ndmin=2) for rp in rollout_paths])

    return start_states, rollouts




def dmp_nn(in_dim, hidden_dim, out_dims):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dims)
    )


start_states, pose_hists = load_demos("demos/splines2D-2019-11-07-14-50-13")
start_states = torch.from_numpy(start_states).to(dtype=torch.float, device=torch.device("cuda")) 
pose_hists = torch.from_numpy(pose_hists).to(dtype=torch.float, device=torch.device("cuda"))

in_dim = start_states[0].numel()
hidden_dim = 512
n_basis_funcs = 50
out_dims = n_basis_funcs * len(pose_hists[0][0])
dt = 0.01
T = int(1 / dt)

print("Rollouts dims: {}".format(pose_hists.shape))

model = nn.Sequential(
    nn.Linear(in_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, out_dims)
)

model.to(torch.device("cuda"))

optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()


train_num = 20
train_set = TensorDataset(start_states[:train_num], pose_hists[:train_num])
val_set = TensorDataset(start_states[train_num:], pose_hists[train_num:])
print("Train Set Size: {}, Val Set Size: {}".format(len(train_set), len(val_set)))

train_loader = DataLoader(train_set, shuffle=True, batch_size=32)
val_loader = DataLoader(val_set, shuffle=False, batch_size=32)



#%%

results_folder = "logs/synth-wave-{}".format(t_stamp())
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


#%%

# display("", display_id="batch_progress")
for epoch in range(500):
    ## Training Loop
    train_losses = []
    for batch_idx, (ss_batch, rollout_batch) in enumerate(train_loader):
        dims = rollout_batch.shape[2]

        learned_weights = model(ss_batch.view(ss_batch.shape[0], -1)).view(-1, dims, n_basis_funcs)
        dmp = DMP(num_basis_funcs=n_basis_funcs, dt=dt, d=dims, weights=learned_weights)
        learned_dmp_rollouts, _, _ = dmp.rollout_torch(ss_batch[:, 0], ss_batch[:, 1])
        loss = loss_fn(learned_dmp_rollouts, rollout_batch)
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_train_loss = np.mean(train_losses)


    ## Validation Loop
    val_losses = []
    with torch.no_grad():
        for batch_idx, (ss_batch, rollout_batch) in enumerate(val_loader):
            dims = rollout_batch.shape[2]

            learned_weights = model(ss_batch.view(ss_batch.shape[0], -1)).view(-1, dims, n_basis_funcs)
            dmp = DMP(num_basis_funcs=n_basis_funcs, dt=dt, d=dims, weights=learned_weights)
            learned_dmp_rollouts = dmp.rollout_torch(ss_batch[:, 0], ss_batch[:, 1])[0]
            loss = loss_fn(learned_dmp_rollouts, rollout_batch)
            val_losses.append(loss.item())

    avg_val_loss = np.mean(val_losses)

    print("e{}\t t: {:.6}\t v: {:.6}".format(epoch, avg_train_loss, avg_val_loss))
    if epoch % 10 == 0:
        torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_{}.pt".format(epoch)))
        
torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_final.pt"))


#%%
model.load_state_dict(
    torch.load(join(results_folder, "learned_model_epoch_final.pt"))
)

%matplotlib auto
plt.style.use('seaborn')
for d in range(5, 6):
    demo_num = d
    # plt.subplot(3,3, d)
    val_starts, val_rollout = val_set[demo_num]
    learned_weights = model(val_starts.view(-1))

    dmp = DMP(num_basis_funcs=n_basis_funcs,
            dt=dt,
            d=pose_hists.shape[2],
            weights=learned_weights.reshape(-1, pose_hists.shape[2], n_basis_funcs))
    learned_dmp_rollout = dmp.rollout_torch(
        val_starts[0].unsqueeze(0),
        val_starts[1].unsqueeze(0)
    )[0][0].detach().cpu()

    dmp_timescale = np.linspace(0, 1, learned_dmp_rollout.shape[0])

    plt.plot(learned_dmp_rollout[:, 0], learned_dmp_rollout[:, 1], label="DMP")
    plt.plot(val_rollout.detach().cpu()[:, 0], val_rollout.detach().cpu()[:, 1], label="Raw", c='orange')
    plt.scatter(val_starts[:, 0].detach().cpu(), val_starts[:, 1].detach().cpu(), c='r', marker='x')
    plt.xlabel("X")
    plt.xlim(-0.1, 1.1)
    plt.ylabel("Y")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.show()


#%%
