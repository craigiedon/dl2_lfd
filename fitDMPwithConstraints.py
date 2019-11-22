#%%
import torch
from torch import nn, optim
from ltl_diff import constraints, oracle
import os
from os.path import join
from nns.dmp_nn import DMPNN
from dmps.dmp import load_dmp_demos, DMP
from helper_funcs.conversions import np_to_pgpu
from helper_funcs.utils import t_stamp
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Load the start states and pose hists
start_states, pose_hists = load_dmp_demos("demos/splines2D-2019-11-07-14-50-13")
start_states = np_to_pgpu(start_states)
pose_hists = np_to_pgpu(pose_hists)

# Create the model and push to gpu
in_dim = start_states[0].numel()
basis_fs = 30
dt = 0.01
model = DMPNN(in_dim, 100, pose_hists.shape[2], basis_fs)
model.to(torch.device("cuda"))
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()


# Create train/validation split and loaders
train_num = 70
train_set = TensorDataset(start_states[:train_num], pose_hists[:train_num])
train_loader = DataLoader(train_set, shuffle=True, batch_size=32)

val_set = TensorDataset(start_states[train_num:], pose_hists[train_num:])
val_loader = DataLoader(val_set, shuffle=False, batch_size=32)

# Create time-stamped results folder (possibly move this out to a util func?)
results_folder = "logs/synth-wave-{}".format(t_stamp())
os.makedirs(results_folder, exist_ok=True)


# Construct a constraint for the "eventually reach this point" constraint.
fixed_point = torch.tensor([0.75, 0.2], device=torch.device("cuda"))
constraint = constraints.EventuallyReach(fixed_point, 1e-2)

def batch_learn(data_loader, loss_fn, constraint, optimizer=None):
    losses = []
    for batch_idx, (starts, rollouts) in enumerate(data_loader):
        batch_size, T, dims = rollouts.shape

        learned_weights = model(starts)
        dmp = DMP(basis_fs, dt, dims)
        learned_rollouts = dmp.rollout_torch(starts[:, 0], starts[:, 1], learned_weights)[0]

        main_loss = loss_fn(learned_rollouts, rollouts)
        c_loss, c_sat = oracle.evaluate_constraint(starts, rollouts, constraint, model, dmp.rollout_torch)
        full_loss = 1.0 * main_loss + 0.1 * c_loss

        losses.append([main_loss.item(), c_loss.item(), full_loss.item()])

        if optimizer is not None:
            optimizer.zero_grad()
            full_loss.backward()
            optimizer.step()

    return losses


#%%
for epoch in range(300):
    # Train loop
    train_losses = batch_learn(train_loader, loss_fn, constraint, optimizer)
    avg_train_loss = np.mean(train_losses, 0, keepdims=True)

    # Validation Loop
    val_losses = batch_learn(val_loader, loss_fn, constraint, None)
    avg_val_loss = np.mean(val_losses, 0, keepdims=True)

    print("e{}\t t: {}\t v: {}".format(epoch, avg_train_loss[0, :2], avg_val_loss[0, :2]))
    if epoch % 10 == 0:
        torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_{}.pt".format(epoch)))
        
torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_final.pt"))

#%%
# Visualization stuff (can probably be in separate file / functions...)
%matplotlib auto
plt.style.use('seaborn')
for d in range(1, 9):
    demo_num = d
    plt.subplot(3,3, d)
    val_starts, val_rollout = val_set[demo_num]
    val_y0 = val_starts[0].unsqueeze(0)
    val_goal = val_starts[1].unsqueeze(0)

    learned_weights = model(val_starts.unsqueeze(0))

    dmp = DMP(basis_fs, dt, pose_hists.shape[2])
    learned_dmp_rollout = dmp.rollout_torch(val_y0, val_goal, learned_weights)[0][0].detach().cpu()

    dmp_timescale = np.linspace(0, 1, learned_dmp_rollout.shape[0])

    plt.plot(learned_dmp_rollout[:, 0], learned_dmp_rollout[:, 1], label="DMP")
    plt.plot(val_rollout.detach().cpu()[:, 0], val_rollout.detach().cpu()[:, 1], label="Raw", c='orange')
    # plt.scatter(val_starts[:, 0].detach().cpu(), val_starts[:, 1].detach().cpu(), c='r', marker='x')
    plt.scatter(fixed_point[0].item(), fixed_point[1].item(), c='r', marker='x')
    plt.xlabel("X")
    plt.xlim(-0.1, 1.1)
    plt.ylabel("Y")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.show()

# Chart results with the constraint:
    # Training v Validation Loss
    # Charting constraint loss / constraint satisfaction...
    # Charting learned thing with added constraint. Does it hit the mark?

# %%
