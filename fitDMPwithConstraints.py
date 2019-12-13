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

def rollout_error(output_roll, target_roll):
    return torch.norm(output_roll - target_roll, dim=2).mean()




# Load the start states and pose hists
start_states, pose_hists = load_dmp_demos("demos/splines2D-2019-12-11-16-00-33")
start_states = np_to_pgpu(start_states)
pose_hists = np_to_pgpu(pose_hists)

# Create the model and push to gpu
in_dim = start_states[0].numel()
basis_fs = 30
dt = 0.01
model = DMPNN(in_dim, 1024, pose_hists.shape[2], basis_fs)
model.to(torch.device("cuda"))
optimizer = optim.Adam(model.parameters())
loss_fn = rollout_error


# Create train/validation split and loaders
train_num = 70
train_set = TensorDataset(start_states[:train_num], pose_hists[:train_num])
train_loader = DataLoader(train_set, shuffle=True, batch_size=32)

val_set = TensorDataset(start_states[train_num:], pose_hists[train_num:])
val_loader = DataLoader(val_set, shuffle=False, batch_size=32)

# Create time-stamped results folder (possibly move this out to a util func?)
results_folder = "logs/synth-wave-multicon{}".format(t_stamp())
os.makedirs(results_folder, exist_ok=True)


# Construct a constraint for the "eventually reach this point" constraint.
constraint = constraints.EventuallyReach([1,2,3], 1e-2)

def batch_learn(data_loader, loss_fn, constraint, optimizer=None):
    losses = []
    for batch_idx, (starts, rollouts) in enumerate(data_loader):
        batch_size, T, dims = rollouts.shape
        # print("B: {}".format(batch_idx))

        learned_weights = model(starts)
        dmp = DMP(basis_fs, dt, dims)
        learned_rollouts = dmp.rollout_torch(starts[:, 0], starts[:, -1], learned_weights)[0]

        main_loss = loss_fn(learned_rollouts, rollouts)

        if constraint is None:
            c_loss, c_sat = torch.tensor(0.0), torch.tensor(1.0)
            full_loss = main_loss
        else:
            c_loss, c_sat = oracle.evaluate_constraint(starts, rollouts, constraint, model, dmp.rollout_torch)
            full_loss = 1.0 * main_loss + 0.5 * c_loss

        losses.append([main_loss.item(), c_loss.item(), full_loss.item()])

        if optimizer is not None:
            optimizer.zero_grad()
            full_loss.backward()
            optimizer.step()

    return np.mean(losses, 0, keepdims=True)


#%%
train_losses = []
val_losses = []
for epoch in range(300):

    # Train loop
    avg_train_loss = batch_learn(train_loader, loss_fn, None, optimizer)

    # Validation Loop
    avg_val_loss = batch_learn(val_loader, loss_fn, constraint, None)

    train_losses.append(avg_train_loss[0])
    val_losses.append(avg_val_loss[0])

    print("e{}\t t: {}\t v: {}".format(epoch, avg_train_loss[0, :2], avg_val_loss[0, :2]))
    if epoch % 10 == 0:
        torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_{}.pt".format(epoch)))
        
torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_final.pt"))
np.savetxt(join(results_folder, "train_losses.txt"), train_losses)
np.savetxt(join(results_folder, "val_losses.txt"), val_losses)

#%%
# Visualization stuff (can probably be in separate file / functions...)
%matplotlib auto
plt.style.use('seaborn')
for d in range(1, 9):
    demo_num = d
    plt.subplot(3,3, d)
    val_starts, val_rollout = train_set[demo_num]
    val_y0 = val_starts[0].unsqueeze(0)
    val_goal = val_starts[-1].unsqueeze(0)

    learned_weights = model(val_starts.unsqueeze(0))

    dmp = DMP(basis_fs, dt, pose_hists.shape[2])
    learned_dmp_rollout = dmp.rollout_torch(val_y0, val_goal, learned_weights, 1.0)[0][0].detach().cpu()

    # dmp_timescale = np.linspace(0, 1, learned_dmp_rollout.shape[0])

    plt.plot(learned_dmp_rollout[:, 0], learned_dmp_rollout[:, -1], label="DMP")
    plt.scatter(val_rollout.detach().cpu()[:, 0], val_rollout.detach().cpu()[:, 1], label="Raw", c='orange', alpha=0.5)

    plt.scatter(val_starts[:, 0].detach().cpu(),
                val_starts[:, 1].detach().cpu(),
                c='r', marker='x')

    plt.xlabel("X")
    plt.xlim(-0.1, 1.1)
    plt.ylabel("Y")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.show()


#%%
plot_train_val(join(results_folder, "train_losses.txt"),
                  join(results_folder, "val_losses.txt"))