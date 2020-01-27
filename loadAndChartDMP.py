import torch
from torch import nn, optim, autograd
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

def chartDMP(data_folder, model_folder):
    start_states, pose_hists = load_dmp_demos(demo_folder)
    start_states, pose_hists = np_to_pgpu(start_states), np_to_pgpu(pose_hists)
    data_set = TensorDataset(start_states, pose_hists)
    basis_fs = 30
    dt = 0.01

    model = DMPNN(in_dim, 1024, t_pose_hists.shape[2], basis_fs).cuda()
    model.load_state_dict(torch.load(join(model_folder, "learned_model_epoch_final.pt")))
    model.eval()

    # TODO: Maybe also make it a _grid_ of subplots rather than a single one this time....(check your previous script runner, or look up plotting grids for matplotlib?)
    i = 0
    inputs, rollout = data_set[i]
    y0 = inputs[0].unsqueeze(0)
    goal = inputs[-1].unsqueeze(0)

    learned_weights = model(inputs.unsqueeze(0))

    dmp = DMP(basis_fs, dt, rollout.shape[1])
    learned_rollout = dmp.rollout_torch(y0, goal, learned_weights, 1.0)[0][0].detach().cpu()

    dmp_timescale = np.linspace(0, 1, learned_rollout.shape[0])

    fig, ax = plt.subplots()
    ax.scatter(rollout.detach().cpu()[:, 0], rollout.detach().cpu()[:, 1], label="Demo", c='orange', alpha=0.5)
    ax.scatter(learned_rollout[:, 0], learned_rollout[:, -1], label="DMP + LTL", alpha=0.5)

    ax.scatter(train_starts[:, 0].detach().cpu(),
               train_starts[:, 1].detach().cpu(),
               c='r', marker='x')

    # Steady Plotting Stuff
    # # ax.plot([0.0, 1.0], [0.75, 0.75], c='r')
    # # ax.plot([0.0, 1.0], [0.25, 0.25], c='r')

    # Avoid obstacle plotting stuff
    # # ax.add_patch(plt.Circle(train_starts[1], radius=0.1, color="red", alpha=0.1))

    plt.xlabel("X")
    plt.xlim(0.0, 1.0)
    plt.ylabel("Y")
    plt.ylim(0.0, 1.0)
    plt.legend(prop={"size": 14})
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_folder = sys.argv[1]
    model_folder = sys.argv[2]
    chartDMP(data_folder, model_folder)